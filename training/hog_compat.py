"""
scikit-image Compatible HOG Feature Extractor (Python)
======================================================

This module implements HOG (Histogram of Oriented Gradients) in pure Python/NumPy
that produces **numerically identical** output to both:
    1. skimage.feature.hog (the original training library)
    2. lib/modules/hog_feature_extractor.dart (the Dart runtime)

The Dart HOG code was rewritten to match skimage exactly. This Python file
provides the same algorithm for validation and future re-extraction.

PARAMETERS (matching training notebook Mood Prediction.ipynb):
    - Target image size: 256×256 grayscale
    - Cell size: 32×32 pixels
    - Block size: 2×2 cells
    - Block stride: 1 cell
    - Orientation bins: 8 (unsigned, 0°-180°)
    - Block normalization: L2-Hys (eps²=1e-10, clip at 0.2)
    - Output dimension: 7×7 blocks × 2×2 cells × 8 bins = 1568

Algorithm details (matching skimage exactly):
    - Gradients: central differences, 0 at borders
    - Cell histograms: hard spatial assignment (NOT tri-linear), averaged by cell area
    - Orientation: hard bin assignment (one bin per pixel)
    - L2-Hys: sqrt(sum(block²) + eps²) where eps=1e-5, eps²=1e-10

Usage:
    from hog_compat import extract_hog_skimage_compat
    features = extract_hog_skimage_compat(image_path)  # np.ndarray of shape (1568,)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

# === Parameters matching skimage hog and Dart hog_feature_extractor.dart ===
NUM_BINS = 8
CELL_SIZE = 32
BLOCK_SIZE = 2
TARGET_SIZE = 256
EPS = 1e-5
EPS_SQUARED = EPS * EPS  # 1e-10
DEG_PER_BIN = 180.0 / NUM_BINS  # 22.5°
INV_CELL_AREA = 1.0 / (CELL_SIZE * CELL_SIZE)  # 1/1024

NUM_CELLS = TARGET_SIZE // CELL_SIZE  # 8
NUM_BLOCKS = NUM_CELLS - BLOCK_SIZE + 1  # 7
DESCRIPTOR_LENGTH = NUM_BLOCKS * NUM_BLOCKS * BLOCK_SIZE * BLOCK_SIZE * NUM_BINS  # 1568


def extract_hog_skimage_compat(image_input) -> np.ndarray:
    """Extract HOG features matching skimage.feature.hog exactly.

    Args:
        image_input: One of:
            - str or Path: file path to an image
            - np.ndarray (H, W, 3): RGB image (uint8 or float)
            - np.ndarray (H, W): grayscale image
            - PIL.Image: PIL image

    Returns:
        np.ndarray of shape (1568,) - the HOG descriptor
    """
    gray = _to_grayscale_resized(image_input)
    mag, ori = _compute_gradients(gray)
    cell_hist = _build_cell_histograms(mag, ori)
    descriptor = _block_normalize(cell_hist)
    return descriptor


# Keep old name as alias for backward compatibility
extract_hog_dart_compat = extract_hog_skimage_compat


def _to_grayscale_resized(image_input) -> np.ndarray:
    """Convert to grayscale and resize to 256×256.

    Uses cv2.COLOR_RGB2GRAY formula: 0.299*R + 0.587*G + 0.114*B
    Returns float64 array with values in [0, 255].
    """
    if isinstance(image_input, (str, Path)):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            pil_img = Image.fromarray(image_input.astype(np.uint8), mode="L").convert("RGB")
        else:
            pil_img = Image.fromarray(image_input.astype(np.uint8), mode="RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    pil_img = pil_img.resize((TARGET_SIZE, TARGET_SIZE), Image.BILINEAR)

    rgb = np.array(pil_img, dtype=np.float64)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray


def _compute_gradients(gray: np.ndarray):
    """Compute gradient magnitude and orientation.

    Matches skimage's _hog_channel_gradient exactly:
        g_row[1:-1,:] = image[2:,:] - image[:-2,:]  (0 at borders)
        g_col[:,1:-1] = image[:,2:] - image[:,:-2]   (0 at borders)
    """
    H, W = TARGET_SIZE, TARGET_SIZE
    g_row = np.zeros((H, W), dtype=np.float64)
    g_col = np.zeros((H, W), dtype=np.float64)
    g_row[1:-1, :] = gray[2:, :] - gray[:-2, :]
    g_col[:, 1:-1] = gray[:, 2:] - gray[:, :-2]

    mag = np.hypot(g_col, g_row)
    ori = np.rad2deg(np.arctan2(g_row, g_col)) % 180.0
    return mag, ori


def _build_cell_histograms(mag: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """Build cell histograms with hard assignment, averaged by cell area.

    Matches skimage's Cython hog_histograms function exactly:
    - Each pixel votes into one cell and one orientation bin
    - Histogram values divided by (cell_rows × cell_columns)
    """
    cell_hist = np.zeros((NUM_CELLS, NUM_CELLS, NUM_BINS), dtype=np.float64)

    for y in range(TARGET_SIZE):
        cy = min(y // CELL_SIZE, NUM_CELLS - 1)
        for x in range(TARGET_SIZE):
            cx = min(x // CELL_SIZE, NUM_CELLS - 1)
            m = mag[y, x]
            o = ori[y, x]
            b = int(o / DEG_PER_BIN)
            if b >= NUM_BINS:
                b = NUM_BINS - 1
            cell_hist[cy, cx, b] += m

    cell_hist *= INV_CELL_AREA
    return cell_hist


def _block_normalize(cell_hist: np.ndarray) -> np.ndarray:
    """L2-Hys block normalization matching skimage's _hog_normalize_block.

    Uses eps²=1e-10 (not eps=1e-5) in the denominator:
        out = block / sqrt(sum(block²) + eps²)
        out = min(out, 0.2)
        out = out / sqrt(sum(out²) + eps²)
    """
    descriptor = np.zeros(DESCRIPTOR_LENGTH, dtype=np.float64)
    offset = 0

    for by in range(NUM_BLOCKS):
        for bx in range(NUM_BLOCKS):
            block = cell_hist[by:by + BLOCK_SIZE, bx:bx + BLOCK_SIZE, :].ravel()

            out = block / np.sqrt(np.sum(block ** 2) + EPS_SQUARED)
            out = np.minimum(out, 0.2)
            out = out / np.sqrt(np.sum(out ** 2) + EPS_SQUARED)

            n = len(out)
            descriptor[offset:offset + n] = out
            offset += n

    return descriptor.astype(np.float32)


# ========== Batch extraction helper ==========

def extract_hog_from_directory(
    image_dir: str | Path,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
) -> dict[str, np.ndarray]:
    """Extract HOG features from all images in a directory.

    Returns:
        dict mapping filename → 1568-d HOG feature vector
    """
    image_dir = Path(image_dir)
    results = {}

    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    for i, img_path in enumerate(image_files):
        try:
            features = extract_hog_dart_compat(img_path)
            results[img_path.name] = features
            if (i + 1) % 100 == 0:
                print(f"  Extracted {i + 1}/{len(image_files)} images...")
        except Exception as e:
            print(f"  WARNING: Failed to extract HOG from {img_path.name}: {e}")

    return results


if __name__ == "__main__":
    # Quick self-test
    print(f"Descriptor length: {DESCRIPTOR_LENGTH}")

    # Test with a random image
    np.random.seed(42)
    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    desc = extract_hog_dart_compat(test_img)
    print(f"Output shape: {desc.shape}")
    print(f"Range: [{desc.min():.6f}, {desc.max():.6f}]")
    print(f"Mean: {desc.mean():.6f}")
    print("Self-test PASSED")
