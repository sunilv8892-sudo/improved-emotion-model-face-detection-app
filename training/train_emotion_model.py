from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional until TFLite export is needed
    tf = None


@dataclass(frozen=True)
class FeatureLayout:
    efficientnet_columns: list[str]
    hog_columns: list[str]
    pose_columns: list[str]
    label_column: str
    image_column: str | None

    @property
    def feature_columns(self) -> list[str]:
        return [
            *self.efficientnet_columns,
            *self.hog_columns,
            *self.pose_columns,
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the emotion pipeline described in the paper: "
            "MinMaxScaler -> LDA(5) -> RBF SVM, then export Flutter runtime assets."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the extracted-features CSV dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory where pkl/json/tflite files are written",
    )
    parser.add_argument(
        "--label-column",
        default="emotion_label",
        help="Name of the emotion label column",
    )
    parser.add_argument(
        "--image-column",
        default="image_name",
        help="Name of the image name column if present",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test split ratio",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=10.0,
        help="RBF SVM C parameter",
    )
    parser.add_argument(
        "--svm-gamma",
        default="scale",
        help="RBF SVM gamma value or 'scale'/'auto'",
    )
    parser.add_argument(
        "--efficientnet-prefixes",
        nargs="*",
        default=["efficientnet", "effnet", "cnn", "deep"],
        help="Column prefixes used to detect EfficientNet features",
    )
    parser.add_argument(
        "--hog-prefixes",
        nargs="*",
        default=["hog"],
        help="Column prefixes used to detect HOG features",
    )
    parser.add_argument(
        "--pose-prefixes",
        nargs="*",
        default=["pose", "angle", "euler"],
        help="Column prefixes used to detect pose angle features",
    )
    parser.add_argument(
        "--efficientnet-dim",
        type=int,
        default=None,
        help="Fallback EfficientNet feature count when columns do not have usable prefixes",
    )
    parser.add_argument(
        "--hog-dim",
        type=int,
        default=None,
        help="Fallback HOG feature count when columns do not have usable prefixes",
    )
    parser.add_argument(
        "--pose-cols",
        nargs="*",
        default=None,
        help="Explicit pose column names, e.g. pose_x pose_y",
    )
    parser.add_argument(
        "--ignore-pose",
        action="store_true",
        help="Train and export using only EfficientNet + HOG features",
    )
    parser.add_argument(
        "--export-tflite",
        action="store_true",
        help="Also export EfficientNetB0 feature extractor as TFLite",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="EfficientNet feature extractor input size",
    )
    return parser.parse_args()


def _starts_with_any(column: str, prefixes: Sequence[str]) -> bool:
    normalized = column.lower()
    return any(normalized.startswith(prefix.lower()) for prefix in prefixes)


def resolve_feature_layout(df: pd.DataFrame, args: argparse.Namespace) -> FeatureLayout:
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in CSV")

    excluded = {args.label_column}
    image_column = args.image_column if args.image_column in df.columns else None
    if image_column is not None:
        excluded.add(image_column)

    candidate_columns = [column for column in df.columns if column not in excluded]

    efficientnet_columns = [
        column
        for column in candidate_columns
        if _starts_with_any(column, args.efficientnet_prefixes)
    ]
    hog_columns = [
        column
        for column in candidate_columns
        if _starts_with_any(column, args.hog_prefixes)
    ]

    if args.ignore_pose:
        pose_columns = []
    elif args.pose_cols:
        missing = [column for column in args.pose_cols if column not in df.columns]
        if missing:
            raise ValueError(f"Pose columns not found: {missing}")
        pose_columns = list(args.pose_cols)
    else:
        pose_columns = [
            column
            for column in candidate_columns
            if _starts_with_any(column, args.pose_prefixes)
        ]

    if efficientnet_columns and hog_columns and (pose_columns or args.ignore_pose):
        return FeatureLayout(
            efficientnet_columns=efficientnet_columns,
            hog_columns=hog_columns,
            pose_columns=pose_columns,
            label_column=args.label_column,
            image_column=image_column,
        )

    if args.efficientnet_dim is None or args.hog_dim is None:
        raise ValueError(
            "Could not infer EfficientNet/HOG/pose columns from CSV names. "
            "Pass --efficientnet-dim, --hog-dim and optionally --pose-cols or --ignore-pose."
        )

    ordered = list(candidate_columns)
    eff_end = args.efficientnet_dim
    hog_end = eff_end + args.hog_dim
    if hog_end > len(ordered):
        raise ValueError("Explicit feature dimensions exceed number of available columns")

    if args.ignore_pose:
        explicit_pose_columns = []
    else:
        explicit_pose_columns = pose_columns or ordered[hog_end:hog_end + 2]
        if len(explicit_pose_columns) != 2:
            raise ValueError("Expected exactly 2 pose columns for the runtime pipeline")

    return FeatureLayout(
        efficientnet_columns=ordered[:eff_end],
        hog_columns=ordered[eff_end:hog_end],
        pose_columns=explicit_pose_columns,
        label_column=args.label_column,
        image_column=image_column,
    )


def export_feature_extractor_tflite(output_dir: Path, input_size: int) -> Path:
    if tf is None:
        raise ImportError(
            "TensorFlow is required for TFLite export. Install tensorflow>=2.13 first."
        )

    # The provided CSV contains ~1000 EfficientNet features, so the runtime
    # extractor must emit a 1000-dimensional vector as well. Using include_top=True
    # preserves the ImageNet classification head size of 1000.
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=True,
        weights="imagenet",
        input_shape=(input_size, input_size, 3),
        classes=1000,
    )
    feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.output,
        name="efficientnet_b0_feature_extractor",
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(feature_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    output_path = output_dir / "efficientnet_feature_extractor.tflite"
    output_path.write_bytes(tflite_model)
    return output_path


def export_flutter_runtime_assets(
    output_dir: Path,
    layout: FeatureLayout,
    scaler: MinMaxScaler,
    lda: LinearDiscriminantAnalysis,
    svm: OneVsRestClassifier,
) -> Path:
    labels = [str(label) for label in svm.classes_]
    gamma_values = [float(estimator._gamma) for estimator in svm.estimators_]
    gamma = float(gamma_values[0])

    runtime_payload = {
        "labels": labels,
        "feature_layout": {
            "efficientnet_dim": len(layout.efficientnet_columns),
            "hog_dim": len(layout.hog_columns),
            "pose_dim": len(layout.pose_columns),
            "total_dim": len(layout.feature_columns),
        },
        "feature_order": {
            "efficientnet": layout.efficientnet_columns,
            "hog": layout.hog_columns,
            "pose": layout.pose_columns,
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
                    "support_vectors": estimator.support_vectors_.tolist(),
                    "dual_coefficients": estimator.dual_coef_[0].tolist(),
                    "intercept": float(estimator.intercept_[0]),
                }
                for label, estimator in zip(labels, svm.estimators_)
            ],
        },
    }

    output_path = output_dir / "emotion_runtime_params.json"
    output_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_path)
    layout = resolve_feature_layout(df, args)

    X = df[layout.feature_columns].astype(np.float32).to_numpy()
    y = df[layout.label_column].astype(str).to_numpy()

    if len(np.unique(y)) != 6:
        print(f"Warning: expected 6 emotion classes, found {len(np.unique(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lda = LinearDiscriminantAnalysis(n_components=5, solver="svd")
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    base_svm = SVC(
        kernel="rbf",
        C=args.svm_c,
        gamma=args.svm_gamma,
        decision_function_shape="ovr",
    )
    svm = OneVsRestClassifier(base_svm)
    svm.fit(X_train_lda, y_train)

    y_pred = svm.predict(X_test_lda)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=svm.classes_))

    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(lda, output_dir / "lda.pkl")
    joblib.dump(svm, output_dir / "svm.pkl")

    runtime_json = export_flutter_runtime_assets(output_dir, layout, scaler, lda, svm)
    print(f"Saved Flutter runtime parameters: {runtime_json}")

    if args.export_tflite:
        tflite_path = export_feature_extractor_tflite(output_dir, args.input_size)
        print(f"Saved EfficientNet feature extractor: {tflite_path}")


if __name__ == "__main__":
    main()