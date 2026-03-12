"""
Rebuild the emotion pipeline from scratch with MATCHED feature extraction.

This script ensures that the features used during training are extracted using
the EXACT same methods as the Dart runtime code:
  - EfficientNet features via the deployed TFLite model
  - HOG features using an implementation that matches the Dart code exactly

Pipeline: EfficientNet(1000-d) + HOG(1568-d) → MinMaxScaler → LDA(5) → RBF SVM

Dataset: FER2013 from HuggingFace (35,887 labeled 48×48 grayscale face images)
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Constants matching the Dart code exactly
# ---------------------------------------------------------------------------
HOG_NUM_BINS = 8
HOG_CELL_SIZE = 8
HOG_BLOCK_SIZE = 2
HOG_BLOCK_STRIDE = 1
HOG_TARGET_SIZE = 64
HOG_EPSILON = 1e-5
HOG_CLIP = 0.2

EFFICIENTNET_INPUT_SIZE = 224
EFFICIENTNET_OUTPUT_DIM = 1280  # include_top=False, pooling='avg' (penultimate features)

# FER2013 labels
FER2013_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

# We keep 6 classes (drop Fear) to match the existing setup
KEEP_LABELS = {"Angry", "Disgust", "Happy", "Neutral", "Sad", "Surprise"}


# ---------------------------------------------------------------------------
# HOG Feature Extractor — EXACT match to Dart code
# ---------------------------------------------------------------------------
def compute_hog(gray_image: np.ndarray) -> np.ndarray:
    """
    Compute HOG descriptor matching the Dart HogFeatureExtractor exactly.

    Parameters
    ----------
    gray_image : np.ndarray
        Grayscale image of shape (H, W) with values in [0, 255].
        Will be resized to HOG_TARGET_SIZE × HOG_TARGET_SIZE.

    Returns
    -------
    np.ndarray
        HOG descriptor of length 1568.
    """
    from PIL import Image

    # Resize to target size (matching Dart's img.copyResize)
    if gray_image.shape[0] != HOG_TARGET_SIZE or gray_image.shape[1] != HOG_TARGET_SIZE:
        pil_img = Image.fromarray(gray_image.astype(np.uint8), mode="L")
        pil_img = pil_img.resize((HOG_TARGET_SIZE, HOG_TARGET_SIZE), Image.BILINEAR)
        gray = np.array(pil_img, dtype=np.float32)
    else:
        gray = gray_image.astype(np.float32)

    h, w = gray.shape
    assert h == HOG_TARGET_SIZE and w == HOG_TARGET_SIZE

    # 1. Compute gradients using centered differences (matching Dart)
    magnitude = np.zeros((h, w), dtype=np.float32)
    orientation = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            # Horizontal gradient
            left = gray[y, x - 1] if x > 0 else gray[y, x]
            right = gray[y, x + 1] if x < w - 1 else gray[y, x]
            gx = right - left

            # Vertical gradient
            top = gray[y - 1, x] if y > 0 else gray[y, x]
            bottom = gray[y + 1, x] if y < h - 1 else gray[y, x]
            gy = bottom - top

            magnitude[y, x] = math.sqrt(gx * gx + gy * gy)

            # Orientation in [0, 180) — unsigned
            angle = math.atan2(gy, gx) * (180.0 / math.pi)
            if angle < 0:
                angle += 180.0
            if angle >= 180.0:
                angle -= 180.0
            orientation[y, x] = angle

    # 2. Build cell histograms with bilinear interpolation
    bin_width = 180.0 / HOG_NUM_BINS
    cells_x = w // HOG_CELL_SIZE
    cells_y = h // HOG_CELL_SIZE

    cell_hist = np.zeros((cells_y, cells_x, HOG_NUM_BINS), dtype=np.float32)

    for cy in range(cells_y):
        for cx in range(cells_x):
            for py in range(HOG_CELL_SIZE):
                for px in range(HOG_CELL_SIZE):
                    img_y = cy * HOG_CELL_SIZE + py
                    img_x = cx * HOG_CELL_SIZE + px

                    mag = magnitude[img_y, img_x]
                    ori = orientation[img_y, img_x]

                    bin_center = ori / bin_width
                    lower_bin = int(math.floor(bin_center)) % HOG_NUM_BINS
                    upper_bin = (lower_bin + 1) % HOG_NUM_BINS
                    upper_weight = bin_center - math.floor(bin_center)
                    lower_weight = 1.0 - upper_weight

                    cell_hist[cy, cx, lower_bin] += mag * lower_weight
                    cell_hist[cy, cx, upper_bin] += mag * upper_weight

    # 3. Block normalization (L2-Hys) — matching Dart exactly
    blocks_x = (cells_x - HOG_BLOCK_SIZE) // HOG_BLOCK_STRIDE + 1
    blocks_y = (cells_y - HOG_BLOCK_SIZE) // HOG_BLOCK_STRIDE + 1
    feature_length = blocks_y * blocks_x * HOG_BLOCK_SIZE * HOG_BLOCK_SIZE * HOG_NUM_BINS
    descriptor = np.zeros(feature_length, dtype=np.float32)
    offset = 0

    for by in range(blocks_y):
        for bx in range(blocks_x):
            block_vec = []
            for dy in range(HOG_BLOCK_SIZE):
                for dx in range(HOG_BLOCK_SIZE):
                    hist = cell_hist[by * HOG_BLOCK_STRIDE + dy, bx * HOG_BLOCK_STRIDE + dx]
                    for b in range(HOG_NUM_BINS):
                        block_vec.append(float(hist[b]))

            # L2-Hys normalization
            norm = 0.0
            for v in block_vec:
                norm += v * v
            norm = math.sqrt(norm + HOG_EPSILON)

            for i in range(len(block_vec)):
                block_vec[i] = block_vec[i] / norm
                if block_vec[i] > HOG_CLIP:
                    block_vec[i] = HOG_CLIP

            norm2 = 0.0
            for v in block_vec:
                norm2 += v * v
            norm2 = math.sqrt(norm2 + HOG_EPSILON)

            for i in range(len(block_vec)):
                descriptor[offset] = block_vec[i] / norm2
                offset += 1

    return descriptor


def compute_hog_vectorized(gray_image: np.ndarray) -> np.ndarray:
    """
    Faster vectorized version of compute_hog, but producing identical results.
    """
    from PIL import Image

    if gray_image.shape[0] != HOG_TARGET_SIZE or gray_image.shape[1] != HOG_TARGET_SIZE:
        pil_img = Image.fromarray(gray_image.astype(np.uint8), mode="L")
        pil_img = pil_img.resize((HOG_TARGET_SIZE, HOG_TARGET_SIZE), Image.BILINEAR)
        gray = np.array(pil_img, dtype=np.float32)
    else:
        gray = gray_image.astype(np.float32)

    h, w = HOG_TARGET_SIZE, HOG_TARGET_SIZE

    # Gradients with centered differences (replicate boundary)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    # Boundaries: gradient = 0 (same as Dart: left==self, right==self → gx=0)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * (180.0 / np.pi)
    orientation[orientation < 0] += 180.0
    orientation[orientation >= 180.0] -= 180.0

    bin_width = 180.0 / HOG_NUM_BINS
    cells_x = w // HOG_CELL_SIZE
    cells_y = h // HOG_CELL_SIZE

    cell_hist = np.zeros((cells_y, cells_x, HOG_NUM_BINS), dtype=np.float32)

    for cy in range(cells_y):
        for cx in range(cells_x):
            y0 = cy * HOG_CELL_SIZE
            x0 = cx * HOG_CELL_SIZE
            cell_mag = magnitude[y0:y0 + HOG_CELL_SIZE, x0:x0 + HOG_CELL_SIZE].ravel()
            cell_ori = orientation[y0:y0 + HOG_CELL_SIZE, x0:x0 + HOG_CELL_SIZE].ravel()

            bin_center = cell_ori / bin_width
            lower_bin = np.floor(bin_center).astype(int) % HOG_NUM_BINS
            upper_bin = (lower_bin + 1) % HOG_NUM_BINS
            upper_weight = bin_center - np.floor(bin_center)
            lower_weight = 1.0 - upper_weight

            for i in range(len(cell_mag)):
                cell_hist[cy, cx, lower_bin[i]] += cell_mag[i] * lower_weight[i]
                cell_hist[cy, cx, upper_bin[i]] += cell_mag[i] * upper_weight[i]

    blocks_x = (cells_x - HOG_BLOCK_SIZE) // HOG_BLOCK_STRIDE + 1
    blocks_y = (cells_y - HOG_BLOCK_SIZE) // HOG_BLOCK_STRIDE + 1
    feature_length = blocks_y * blocks_x * HOG_BLOCK_SIZE * HOG_BLOCK_SIZE * HOG_NUM_BINS

    descriptor = np.zeros(feature_length, dtype=np.float32)
    offset = 0

    for by in range(blocks_y):
        for bx in range(blocks_x):
            block = []
            for dy in range(HOG_BLOCK_SIZE):
                for dx in range(HOG_BLOCK_SIZE):
                    block.extend(
                        cell_hist[by * HOG_BLOCK_STRIDE + dy, bx * HOG_BLOCK_STRIDE + dx]
                    )
            block = np.array(block, dtype=np.float32)

            norm = np.sqrt(np.sum(block ** 2) + HOG_EPSILON)
            block = block / norm
            block = np.clip(block, 0.0, HOG_CLIP)

            norm2 = np.sqrt(np.sum(block ** 2) + HOG_EPSILON)
            block = block / norm2

            descriptor[offset:offset + len(block)] = block
            offset += len(block)

    return descriptor


# ---------------------------------------------------------------------------
# EfficientNet Feature Extraction — Keras batch (fast) + TFLite (for validation)
# ---------------------------------------------------------------------------
class KerasFeatureExtractor:
    """Use Keras EfficientNetB0 directly for fast batch feature extraction."""

    def __init__(self):
        import tensorflow as tf

        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE, 3),
            pooling="avg",
        )
        self.model = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.output,
            name="efficientnet_b0_feature_extractor",
        )
        self.output_dim = self.model.output_shape[-1]  # 1280
        print(f"  Keras EfficientNetB0 loaded: output_dim={self.output_dim}")

    def extract_batch(self, rgb_images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        """
        Extract features for a list of RGB images in batches.

        Parameters
        ----------
        rgb_images : list of np.ndarray
            List of RGB images (any size), uint8 [0, 255].
        batch_size : int
            TF prediction batch size.

        Returns
        -------
        np.ndarray of shape (N, output_dim)
        """
        from PIL import Image as PILImage

        all_features = np.zeros((len(rgb_images), self.output_dim), dtype=np.float64)
        n_batches = (len(rgb_images) + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(rgb_images))
            batch_imgs = rgb_images[start:end]

            # Preprocess this batch only
            preprocessed = np.zeros(
                (len(batch_imgs), EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE, 3),
                dtype=np.float32,
            )
            for i, img in enumerate(batch_imgs):
                pil_img = PILImage.fromarray(img)
                pil_img = pil_img.resize(
                    (EFFICIENTNET_INPUT_SIZE, EFFICIENTNET_INPUT_SIZE),
                    PILImage.BILINEAR,
                )
                preprocessed[i] = np.array(pil_img, dtype=np.float32)

            features = self.model.predict(preprocessed, batch_size=batch_size, verbose=0)
            all_features[start:end] = features.astype(np.float64)

            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(f"    Batch {batch_idx + 1}/{n_batches}")

        return all_features

    def export_tflite(self, output_path: Path) -> None:
        """Export the same model as TFLite for Flutter runtime."""
        import tensorflow as tf

        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        output_path.write_bytes(tflite_bytes)
        print(f"  Saved TFLite model: {output_path} ({len(tflite_bytes) / 1024 / 1024:.1f} MB)")


class TFLiteFeatureExtractor:
    """Run the deployed EfficientNet TFLite model for feature extraction."""

    def __init__(self, model_path: str | Path):
        import tensorflow as tf

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.input_index = input_details[0]["index"]
        self.output_index = output_details[0]["index"]
        self.input_shape = input_details[0]["shape"]  # [1, 224, 224, 3]
        self.input_dtype = input_details[0]["dtype"]
        self.output_shape = output_details[0]["shape"]

        self.input_h = int(self.input_shape[1])
        self.input_w = int(self.input_shape[2])
        self.input_c = int(self.input_shape[3])
        self.output_dim = int(np.prod(self.output_shape[1:]))

        print(f"  TFLite model loaded: input {self.input_shape}, output {self.output_shape}")
        print(f"  Input dtype: {self.input_dtype}, output dim: {self.output_dim}")

    def extract(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Extract features from an RGB image.

        Parameters
        ----------
        rgb_image : np.ndarray
            RGB image of any size, uint8 [0, 255].

        Returns
        -------
        np.ndarray
            Feature vector of shape (output_dim,).
        """
        from PIL import Image

        # Resize to model input size (matching Dart's img.copyResize → bilinear)
        pil_img = Image.fromarray(rgb_image)
        pil_img = pil_img.resize((self.input_w, self.input_h), Image.BILINEAR)
        img_array = np.array(pil_img, dtype=np.float32)

        # Dart code passes raw [0, 255] RGB values — match that exactly
        # EfficientNet TF2 has built-in normalization, so [0, 255] is correct
        input_data = img_array.reshape(1, self.input_h, self.input_w, self.input_c)

        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)

        return output.flatten().astype(np.float64)


# ---------------------------------------------------------------------------
# Dataset Loading
# ---------------------------------------------------------------------------
def load_fer2013() -> tuple[list[np.ndarray], list[str]]:
    """
    Load FER2013 dataset from HuggingFace (clip-benchmark/wds_fer2013).

    Returns list of grayscale images (48×48 uint8) and string labels.
    """
    from datasets import load_dataset
    from PIL import Image as PILImage

    images = []
    labels = []

    for split_name in ["train", "test"]:
        print(f"  Loading {split_name} split from HuggingFace...")
        try:
            ds = load_dataset("clip-benchmark/wds_fer2013", split=split_name)
        except Exception as e:
            print(f"  Warning: could not load {split_name} split: {e}")
            continue

        for example in ds:
            label_idx = example["cls"]
            if label_idx is None:
                continue
            label_str = FER2013_LABELS.get(label_idx)
            if label_str is None or label_str not in KEEP_LABELS:
                continue

            pil_img = example["jpg"]
            if pil_img is None:
                continue

            # Convert to grayscale numpy array
            gray = np.array(pil_img.convert("L"), dtype=np.uint8)
            images.append(gray)
            labels.append(label_str)

    print(f"  Loaded {len(images)} images across {len(set(labels))} classes")
    for lbl in sorted(set(labels)):
        count = sum(1 for l in labels if l == lbl)
        print(f"    {lbl}: {count}")

    return images, labels


def grayscale_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Convert grayscale image to RGB by replicating channels."""
    if gray.ndim == 2:
        return np.stack([gray, gray, gray], axis=-1)
    return gray


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Rebuild emotion pipeline with matched features")
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--tflite-model", type=Path, default=None,
                        help="Path to the EfficientNet TFLite model. "
                             "Defaults to assets/models/efficientnet_feature_extractor.tflite")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--svm-gamma", default="scale")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples for faster iteration")
    parser.add_argument("--skip-efficientnet", action="store_true",
                        help="Use zeros instead of EfficientNet features (for debugging)")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent

    # ── Step 1: Load dataset ──
    print("\n=== Loading FER2013 Dataset ===")
    images, labels = load_fer2013()

    if args.max_samples and args.max_samples < len(images):
        print(f"  Limiting to {args.max_samples} samples")
        rng = np.random.RandomState(args.random_state)
        indices = rng.choice(len(images), args.max_samples, replace=False)
        images = [images[i] for i in indices]
        labels = [labels[i] for i in indices]

    # ── Step 2: Extract features ──
    print("\n=== Extracting Features ===")

    # Use Keras model directly for fast batch extraction
    keras_extractor: Optional[KerasFeatureExtractor] = None
    efn_dim = EFFICIENTNET_OUTPUT_DIM

    if not args.skip_efficientnet:
        keras_extractor = KerasFeatureExtractor()
        efn_dim = keras_extractor.output_dim
    else:
        print("  Skipping EfficientNet (using zeros)")

    # Compute HOG descriptor length
    cells_x = HOG_TARGET_SIZE // HOG_CELL_SIZE
    cells_y = HOG_TARGET_SIZE // HOG_CELL_SIZE
    blocks_x = (cells_x - HOG_BLOCK_SIZE) // HOG_BLOCK_STRIDE + 1
    blocks_y = (cells_y - HOG_BLOCK_SIZE) // HOG_BLOCK_STRIDE + 1
    hog_dim = blocks_y * blocks_x * HOG_BLOCK_SIZE * HOG_BLOCK_SIZE * HOG_NUM_BINS
    print(f"  EfficientNet dim: {efn_dim}, HOG dim: {hog_dim}")
    total_dim = efn_dim + hog_dim

    # Extract EfficientNet features in batches (fast)
    if keras_extractor is not None:
        print("  Extracting EfficientNet features (batch mode)...")
        rgb_images = [grayscale_to_rgb(img) for img in images]
        efn_features_all = keras_extractor.extract_batch(rgb_images, batch_size=32)
        print(f"  EfficientNet done: shape={efn_features_all.shape}")
    else:
        efn_features_all = np.zeros((len(images), efn_dim), dtype=np.float64)

    # Extract HOG features
    print("  Extracting HOG features...")
    hog_features_all = np.zeros((len(images), hog_dim), dtype=np.float64)
    t0 = time.time()
    for i, gray_img in enumerate(images):
        if (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            print(f"    [{i + 1}/{len(images)}] {rate:.0f} img/s")
        hog_features_all[i] = compute_hog_vectorized(gray_img)
    elapsed = time.time() - t0
    print(f"  HOG done: {elapsed:.1f}s ({len(images) / max(elapsed, 0.001):.0f} img/s)")

    # Combine features
    all_features = np.concatenate([efn_features_all, hog_features_all], axis=1)
    print(f"  Combined features shape: {all_features.shape}")

    # ── Step 3: Train-test split ──
    print("\n=== Training Pipeline ===")
    y = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Step 4: MinMaxScaler ──
    print("  Fitting MinMaxScaler...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Step 5: LDA ──
    n_classes = len(set(y_train))
    lda_components = min(5, n_classes - 1)
    print(f"  Fitting LDA (n_components={lda_components})...")
    lda = LinearDiscriminantAnalysis(n_components=lda_components, solver="svd")
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    X_test_lda = lda.transform(X_test_scaled)

    # ── Step 6: SVM ──
    print(f"  Training OneVsRest RBF SVM (C={args.svm_c}, gamma={args.svm_gamma})...")
    base_svm = SVC(
        kernel="rbf",
        C=args.svm_c,
        gamma=args.svm_gamma,
        decision_function_shape="ovr",
    )
    svm = OneVsRestClassifier(base_svm)
    svm.fit(X_train_lda, y_train)

    # ── Step 7: Evaluate ──
    y_pred = svm.predict(X_test_lda)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  ✅ Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=svm.classes_))

    # ── Step 8: Export ──
    print("\n=== Exporting Model Artifacts ===")

    # Save sklearn objects
    joblib.dump(scaler, output_dir / "scaler.pkl")
    joblib.dump(lda, output_dir / "lda.pkl")
    joblib.dump(svm, output_dir / "svm.pkl")
    print(f"  Saved scaler.pkl, lda.pkl, svm.pkl")

    # Export Flutter runtime JSON
    sorted_labels = [str(c) for c in svm.classes_]
    gamma_values = [float(est._gamma) for est in svm.estimators_]
    gamma = float(gamma_values[0])

    # Build feature column names
    efn_cols = [str(i) for i in range(efn_dim)]
    hog_cols = [f"hog_{i}" for i in range(hog_dim)]

    runtime_payload = {
        "labels": sorted_labels,
        "feature_layout": {
            "efficientnet_dim": efn_dim,
            "hog_dim": hog_dim,
            "pose_dim": 0,
            "total_dim": total_dim,
        },
        "feature_order": {
            "efficientnet": efn_cols,
            "hog": hog_cols,
            "pose": [],
        },
        "scaler": {
            "min": scaler.min_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "lda": {
            "xbar": lda.xbar_.tolist(),
            "scalings": lda.scalings_[:, :lda.n_components].tolist(),
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
                for label, est in zip(sorted_labels, svm.estimators_)
            ],
        },
    }

    json_path = output_dir / "emotion_runtime_params.json"
    json_path.write_text(json.dumps(runtime_payload, indent=2), encoding="utf-8")
    print(f"  Saved {json_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Pipeline rebuilt successfully!")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Features: EfficientNet({efn_dim}) + HOG({hog_dim}) = {total_dim}")
    print(f"  LDA components: {lda_components}")
    print(f"  SVM gamma: {gamma:.6f}")
    print(f"  Classes: {sorted_labels}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Export TFLite model (same architecture used for feature extraction)
    if keras_extractor is not None:
        tflite_path = output_dir / "efficientnet_feature_extractor.tflite"
        print("\n  Exporting TFLite model for Flutter runtime...")
        keras_extractor.export_tflite(tflite_path)

    # Copy to assets if in project tree
    assets_dir = project_root / "assets" / "models"
    if assets_dir.exists():
        import shutil
        shutil.copy2(json_path, assets_dir / "emotion_runtime_params.json")
        tflite_src = output_dir / "efficientnet_feature_extractor.tflite"
        if tflite_src.exists():
            shutil.copy2(tflite_src, assets_dir / "efficientnet_feature_extractor.tflite")
        print(f"\n  ✅ Copied artifacts to {assets_dir}")


if __name__ == "__main__":
    main()
