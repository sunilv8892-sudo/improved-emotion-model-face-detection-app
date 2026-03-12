"""
Train MinMaxScaler -> LDA -> OneVsRest RBF SVM from the pre-extracted CSV.

CSV layout (2572 columns):
  0..999     -> EfficientNet-B0 features (include_top=True, 1000-d)
  1000..2567 -> HOG features (1568-d)
  X-degree   -> head pose X
  Y-degree   -> head pose Y
  Class      -> emotion label
  Image_Name -> source filename

Pipeline: features -> MinMaxScaler -> LDA(n_components=5) -> OneVsRest(SVC(rbf))

Usage:
  python training/train_from_csv.py
  python training/train_from_csv.py --svm-c 10 --use-pose
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.svm import SVC


# ── Feature layout constants (must match the CSV) ──────────────────────────
EFFICIENTNET_DIM = 1000  # include_top=True -> 1000 ImageNet softmax probs
HOG_DIM = 1568           # 8 bins, 8x8 cells, 2x2 blocks, stride 1, 64x64

LABELS = ["Angry", "Disgust", "Happy", "Neutral", "Sad", "Surprise"]


def load_csv(csv_path: str, use_pose: bool, efn_only: bool = False) -> tuple:
    """Load features and labels from the pre-extracted CSV."""
    print(f"\n=== Loading CSV: {csv_path} ===")
    df = pd.read_csv(csv_path)
    print(f"  Shape: {df.shape}")
    print(f"  Class distribution: {df['Class'].value_counts().to_dict()}")

    # Extract feature columns
    efn_cols = [str(i) for i in range(EFFICIENTNET_DIM)]
    efn_features = df[efn_cols].values.astype(np.float64)

    if efn_only:
        features = efn_features
        hog_dim = 0
        pose_dim = 0
        print(f"  EFN-only mode: using only EfficientNet features")
    else:
        hog_cols = [str(i) for i in range(EFFICIENTNET_DIM, EFFICIENTNET_DIM + HOG_DIM)]
        hog_features = df[hog_cols].values.astype(np.float64)

        if use_pose:
            pose_features = df[["X-degree", "Y-degree"]].values.astype(np.float64)
            features = np.hstack([efn_features, hog_features, pose_features])
            pose_dim = 2
            print(f"  Using pose features (X-degree, Y-degree)")
        else:
            features = np.hstack([efn_features, hog_features])
            pose_dim = 0
            print(f"  Skipping pose features (not available at runtime)")
        hog_dim = HOG_DIM

    labels = df["Class"].to_numpy(dtype=str, na_value="Unknown")
    total_dim = features.shape[1]
    print(f"  Feature dim: EFN={EFFICIENTNET_DIM} + HOG={hog_dim} + Pose={pose_dim if not efn_only else 0} = {total_dim}")

    return features, labels, pose_dim if not efn_only else 0, hog_dim


def train_pipeline(features, labels, svm_c: float, svm_gamma: str, lda_components: int):
    """Train MinMaxScaler -> LDA -> OneVsRest RBF SVM."""

    # ── Train/test split ──
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"\n=== Training Pipeline ===")
    print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # ── MinMaxScaler ──
    print(f"  Fitting MinMaxScaler...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── LDA ──
    n_classes = len(np.unique(y_train))
    lda_dim = min(lda_components, n_classes - 1)
    print(f"  Fitting LDA (output_dim={lda_dim})...")
    lda = LinearDiscriminantAnalysis(n_components=lda_dim)
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    # ── SVM ──
    gamma_value = svm_gamma if svm_gamma == "scale" else float(svm_gamma)
    print(f"  Training OneVsRest RBF SVM (C={svm_c}, gamma={gamma_value})...")
    base_svm = SVC(kernel="rbf", C=svm_c, gamma=gamma_value, probability=False)
    ovr_svm = OneVsRestClassifier(base_svm, n_jobs=-1)

    # Binarize labels for OvR
    y_train_bin = label_binarize(y_train, classes=LABELS)
    ovr_svm.fit(X_train_lda, y_train_bin)

    # ── Evaluate ──
    y_test_bin = label_binarize(y_test, classes=LABELS)
    y_pred_bin = ovr_svm.predict(X_test_lda)
    y_pred_labels = [LABELS[np.argmax(row)] for row in y_pred_bin]

    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"\n=== Results ===")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\n{classification_report(y_test, y_pred_labels, target_names=LABELS)}")

    return scaler, lda, ovr_svm, lda_dim, accuracy


def export_params(scaler, lda, ovr_svm, lda_dim, pose_dim, output_dir, hog_dim=HOG_DIM):
    """Export all parameters to JSON for the Dart runtime."""

    total_dim = EFFICIENTNET_DIM + hog_dim + pose_dim

    # ── Scaler params ──
    # sklearn MinMaxScaler: X_scaled = X * scale_ + min_
    scaler_min = scaler.min_.tolist()
    scaler_scale = scaler.scale_.tolist()

    # ── LDA params ──
    lda_xbar = lda.xbar_.tolist()
    lda_scalings = lda.scalings_[:, :lda_dim].tolist()

    # ── SVM params (per binary model) ──
    binary_models = []
    for idx, estimator in enumerate(ovr_svm.estimators_):
        label = LABELS[idx]
        sv = estimator.support_vectors_.tolist()
        dc = estimator.dual_coef_[0].tolist()
        intercept = float(estimator.intercept_[0])
        gamma = float(estimator._gamma) if hasattr(estimator, "_gamma") else float(estimator.gamma)
        binary_models.append({
            "label": label,
            "support_vectors": sv,
            "dual_coefficients": dc,
            "intercept": intercept,
        })

    # Get gamma from first estimator
    est0 = ovr_svm.estimators_[0]
    if hasattr(est0, "_gamma"):
        gamma_val = float(est0._gamma)
    else:
        gamma_val = 1.0 / (lda_dim * np.var(lda.transform(scaler.transform(
            np.zeros((1, total_dim))
        ))))

    params = {
        "feature_layout": {
            "efficientnet_dim": EFFICIENTNET_DIM,
            "hog_dim": hog_dim,
            "pose_dim": pose_dim,
            "total_dim": total_dim,
        },
        "feature_order": ["efficientnet"] + (["hog"] if hog_dim > 0 else []) + (["pose"] if pose_dim > 0 else []),
        "labels": LABELS,
        "scaler": {
            "min": scaler_min,
            "scale": scaler_scale,
        },
        "lda": {
            "xbar": lda_xbar,
            "scalings": lda_scalings,
            "output_dim": lda_dim,
        },
        "svm": {
            "gamma": gamma_val,
            "binary_models": binary_models,
        },
    }

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "emotion_runtime_params.json")
    with open(json_path, "w") as f:
        json.dump(params, f)
    file_size = os.path.getsize(json_path)
    print(f"\n  Saved JSON params: {json_path} ({file_size/1024:.1f} KB)")
    return json_path


def export_tflite_include_top(output_dir):
    """Export EfficientNetB0 with include_top=True as TFLite (1000-d output)."""
    import tensorflow as tf

    print(f"\n=== Exporting TFLite (include_top=True, 1000-d) ===")
    model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )
    print(f"  Model output shape: {model.output_shape}")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(output_dir, "efficientnet_feature_extractor.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    file_size = os.path.getsize(tflite_path)
    print(f"  Saved TFLite: {tflite_path} ({file_size/1024/1024:.1f} MB)")
    return tflite_path


def main():
    parser = argparse.ArgumentParser(description="Train emotion SVM from pre-extracted CSV")
    parser.add_argument("--csv", default="training/EfficientNetb0_HOG_pose_FM (1).csv",
                        help="Path to the pre-extracted feature CSV")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    parser.add_argument("--svm-c", type=float, default=10.0, help="SVM regularization C")
    parser.add_argument("--svm-gamma", default="scale", help="SVM gamma (or 'scale')")
    parser.add_argument("--lda-components", type=int, default=5, help="LDA output components")
    parser.add_argument("--use-pose", action="store_true", help="Include X/Y pose features")
    parser.add_argument("--efn-only", action="store_true", help="Use only EfficientNet features (skip HOG)")
    parser.add_argument("--skip-tflite", action="store_true", help="Skip TFLite export")
    args = parser.parse_args()

    # 1. Load features from CSV (instant — no model inference needed!)
    features, labels, pose_dim, hog_dim = load_csv(args.csv, args.use_pose, args.efn_only)

    # 2. Train pipeline
    scaler, lda, ovr_svm, lda_dim, accuracy = train_pipeline(
        features, labels, args.svm_c, args.svm_gamma, args.lda_components
    )

    # 3. Export JSON parameters
    json_path = export_params(scaler, lda, ovr_svm, lda_dim, pose_dim, args.output_dir, hog_dim=hog_dim)

    # 4. Export TFLite model (include_top=True)
    if not args.skip_tflite:
        tflite_path = export_tflite_include_top(args.output_dir)

    # 5. Copy to Flutter assets
    assets_dir = "assets/models"
    os.makedirs(assets_dir, exist_ok=True)

    shutil.copy2(json_path, os.path.join(assets_dir, "emotion_runtime_params.json"))
    print(f"  Copied JSON -> {assets_dir}/emotion_runtime_params.json")

    if not args.skip_tflite:
        shutil.copy2(tflite_path, os.path.join(assets_dir, "efficientnet_feature_extractor.tflite"))
        print(f"  Copied TFLite -> {assets_dir}/efficientnet_feature_extractor.tflite")

    print(f"\n=== DONE! Accuracy: {accuracy*100:.2f}% ===")
    print(f"  Run 'flutter clean && flutter run' to test the new model.")


if __name__ == "__main__":
    main()
