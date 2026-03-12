"""
Re-Extract Features & Retrain with EFN + HOG
=============================================

This script solves the HOG mismatch problem by:
1. Re-extracting HOG features from the original training images using the
   Dart-compatible HOG algorithm (hog_compat.py)
2. Keeping the original EfficientNet features from the CSV (they already match)
3. Retraining the pipeline (MinMaxScaler → LDA → SVM) with EFN + HOG features
4. Deploying the retrained model to the Flutter assets

PREREQUISITES:
    - Original training images directory
    - The existing CSV file (for EFN features + labels)
    - Python packages: numpy, pandas, scikit-learn, Pillow, joblib

RUN LOCALLY:
    python training/re_extract_and_retrain.py --images-dir path/to/training_images

RUN ON GOOGLE COLAB (free GPU not needed, CPU is fine):
    1. Upload this script + hog_compat.py + the CSV + your images folder to Colab
    2. !pip install numpy pandas scikit-learn Pillow joblib
    3. !python re_extract_and_retrain.py --images-dir /content/training_images

The script will create a new CSV with Dart-compatible HOG features and retrain
the full EFN+HOG pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Import our Dart-compatible HOG extractor
sys.path.insert(0, str(Path(__file__).parent))
from hog_compat import extract_hog_dart_compat, DESCRIPTOR_LENGTH as HOG_DIM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-extract HOG features using Dart-compatible algorithm and retrain"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing the original training images",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("training/EfficientNetb0_HOG_pose_FM (1).csv"),
        help="Original CSV with EFN features and labels",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("training/efn_hog_dart_compat.csv"),
        help="Output CSV with Dart-compatible HOG features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for trained model files",
    )
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        default=Path("assets/models"),
        help="Flutter assets directory for deployment",
    )
    parser.add_argument("--svm-c", type=float, default=15.0)
    parser.add_argument("--svm-gamma", default="0.1")
    parser.add_argument("--lda-components", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip HOG re-extraction (use existing output CSV)",
    )
    parser.add_argument(
        "--use-pose",
        action="store_true",
        help="Include pose features (X-degree, Y-degree)",
    )
    return parser.parse_args()


def re_extract_hog_features(
    original_csv: Path,
    images_dir: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """Re-extract HOG features and create a new CSV."""

    print(f"Loading original CSV: {original_csv}")
    df = pd.read_csv(original_csv)
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

    # EFN features are columns 0-999
    efn_cols = [str(i) for i in range(1000)]

    # Verify all images exist
    image_names = df["Image_Name"].tolist()
    missing = [name for name in image_names if not (images_dir / name).exists()]
    if missing:
        print(f"\n  WARNING: {len(missing)} images not found in {images_dir}")
        print(f"  First 5 missing: {missing[:5]}")
        if len(missing) > len(image_names) * 0.1:
            raise FileNotFoundError(
                f"{len(missing)}/{len(image_names)} images missing. "
                f"Check --images-dir path."
            )
        # Remove rows with missing images
        df = df[~df["Image_Name"].isin(missing)].reset_index(drop=True)
        print(f"  Continuing with {len(df)} images")

    # Extract Dart-compatible HOG for each image
    print(f"\nExtracting Dart-compatible HOG features from {len(df)} images...")
    hog_features = []
    for i, image_name in enumerate(df["Image_Name"]):
        img_path = images_dir / image_name
        try:
            hog = extract_hog_dart_compat(img_path)
            hog_features.append(hog)
        except Exception as e:
            print(f"  ERROR extracting {image_name}: {e}")
            hog_features.append(np.zeros(HOG_DIM, dtype=np.float32))

        if (i + 1) % 500 == 0 or i == len(df) - 1:
            print(f"  Progress: {i + 1}/{len(df)}")

    hog_array = np.array(hog_features, dtype=np.float32)
    print(f"  HOG shape: {hog_array.shape}")
    print(f"  HOG range: [{hog_array.min():.4f}, {hog_array.max():.4f}]")

    # Build new CSV: EFN (original) + HOG (re-extracted) + metadata
    hog_col_names = [f"hog_{i}" for i in range(HOG_DIM)]
    hog_df = pd.DataFrame(hog_array, columns=hog_col_names)

    new_df = pd.concat(
        [
            df[efn_cols],  # Original EFN features
            hog_df,  # Dart-compatible HOG features
            df[["X-degree", "Y-degree", "Class", "Image_Name"]],
        ],
        axis=1,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_csv, index=False)
    print(f"\nSaved new CSV: {output_csv} ({len(new_df)} rows, {len(new_df.columns)} cols)")
    return new_df


def train_pipeline(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[float, str]:
    """Train the full EFN+HOG pipeline and export."""

    # Feature columns
    efn_cols = [str(i) for i in range(1000)]
    hog_cols = [c for c in df.columns if c.startswith("hog_")]
    feature_cols = efn_cols + hog_cols

    if args.use_pose and "X-degree" in df.columns:
        feature_cols += ["X-degree", "Y-degree"]

    efn_dim = len(efn_cols)
    hog_dim = len(hog_cols)
    pose_dim = 2 if args.use_pose else 0
    total_dim = len(feature_cols)

    print(f"\n{'='*60}")
    print(f"TRAINING: EFN({efn_dim}) + HOG({hog_dim}) + Pose({pose_dim}) = {total_dim}")
    print(f"{'='*60}")

    X = df[feature_cols].astype(np.float32).to_numpy()
    y = df["Class"].astype(str).to_numpy()

    print(f"Classes: {np.unique(y)}")
    print(f"Samples: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # LDA
    lda = LinearDiscriminantAnalysis(n_components=args.lda_components, solver="svd")
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    # SVM
    gamma_val = args.svm_gamma
    try:
        gamma_val = float(gamma_val)
    except ValueError:
        pass  # "scale" or "auto"

    base_svm = SVC(kernel="rbf", C=args.svm_c, gamma=gamma_val)
    svm = OneVsRestClassifier(base_svm)
    svm.fit(X_train_lda, y_train)

    y_pred = svm.predict(X_test_lda)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Export
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(lda, output_dir / "lda.pkl")
    joblib.dump(svm, output_dir / "svm.pkl")

    # Export Flutter runtime JSON
    labels = [str(label) for label in svm.classes_]
    gamma_values = [float(est._gamma) for est in svm.estimators_]
    gamma = float(gamma_values[0])

    runtime = {
        "labels": labels,
        "feature_layout": {
            "efficientnet_dim": efn_dim,
            "hog_dim": hog_dim,
            "pose_dim": pose_dim,
            "total_dim": total_dim,
        },
        "feature_order": {
            "efficientnet": list(efn_cols),
            "hog": list(hog_cols),
            "pose": ["X-degree", "Y-degree"] if pose_dim > 0 else [],
        },
        "scaler": {
            "min": scaler.min_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "lda": {
            "xbar": lda.xbar_.tolist(),
            "scalings": lda.scalings_[:, : lda.n_components].tolist(),
            "output_dim": int(lda.n_components),
        },
        "svm": {
            "kernel": "rbf",
            "strategy": "one_vs_rest",
            "gamma": gamma,
            "binary_models": [
                {
                    "label": label,
                    "support_vectors": est.support_vectors_.tolist(),
                    "dual_coefficients": est.dual_coef_[0].tolist(),
                    "intercept": float(est.intercept_[0]),
                }
                for label, est in zip(labels, svm.estimators_)
            ],
        },
    }

    runtime_path = output_dir / "emotion_runtime_params.json"
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    print(f"\nSaved runtime params: {runtime_path}")

    # Deploy to Flutter assets
    deploy_dir = args.deploy_dir
    if deploy_dir.exists():
        import shutil
        dest = deploy_dir / "emotion_runtime_params.json"
        shutil.copy2(runtime_path, dest)
        print(f"Deployed to: {dest}")
    else:
        print(f"Deploy dir not found: {deploy_dir} (copy manually)")

    return accuracy, report


def main():
    args = parse_args()

    if args.skip_extraction:
        if not args.output_csv.exists():
            print(f"ERROR: --skip-extraction but {args.output_csv} not found")
            sys.exit(1)
        print(f"Loading pre-extracted CSV: {args.output_csv}")
        df = pd.read_csv(args.output_csv)
    else:
        df = re_extract_hog_features(args.csv, args.images_dir, args.output_csv)

    accuracy, report = train_pipeline(df, args)

    print(f"\n{'='*60}")
    print(f"DONE! Accuracy with EFN+HOG: {accuracy:.4f}")
    print(f"{'='*60}")
    print("\nThe model now uses BOTH EfficientNet AND HOG features.")
    print("The Dart runtime will extract HOG features that exactly match")
    print("the training data, so the SVM classifier will work correctly.")
    print("\nRebuild the Flutter app: flutter clean && flutter run")


if __name__ == "__main__":
    main()
