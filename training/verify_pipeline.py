"""Verify all dimensions and end-to-end pipeline math."""
import json, numpy as np, math, os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

with open("assets/models/emotion_runtime_params.json") as f:
    params = json.load(f)

layout = params["feature_layout"]
print("Feature layout:", layout)
print("Labels:", params["labels"])

# Verify scaler dimensions
scaler_min = params["scaler"]["min"]
scaler_scale = params["scaler"]["scale"]
print(f"Scaler min length: {len(scaler_min)}, scale length: {len(scaler_scale)}")
assert len(scaler_min) == layout["total_dim"]
assert len(scaler_scale) == layout["total_dim"]

# Verify LDA dimensions
lda_xbar = params["lda"]["xbar"]
lda_scalings = params["lda"]["scalings"]
lda_out = params["lda"]["output_dim"]
print(f"LDA xbar length: {len(lda_xbar)}, scalings: {len(lda_scalings)}x{len(lda_scalings[0])}, output_dim: {lda_out}")
assert len(lda_xbar) == layout["total_dim"]
assert len(lda_scalings) == layout["total_dim"]
assert len(lda_scalings[0]) == lda_out

# Verify SVM dimensions
svm = params["svm"]
print(f"SVM gamma: {svm['gamma']}")
print(f"Number of binary models: {len(svm['binary_models'])}")
for bm in svm["binary_models"]:
    svs = bm["support_vectors"]
    dc = bm["dual_coefficients"]
    print(f"  {bm['label']}: {len(svs)} SVs, SV dim={len(svs[0])}, {len(dc)} coeffs, intercept={bm['intercept']:.4f}")
    assert len(svs) == len(dc)
    assert len(svs[0]) == lda_out

print("\n=== All dimension checks passed! ===\n")

# Cross-validate with sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd

CSV_PATH = "training/EfficientNetb0_HOG_pose_FM (1).csv"
df = pd.read_csv(CSV_PATH)
efn_cols = [str(i) for i in range(1000)]
hog_cols = [str(i) for i in range(1000, 2568)]
features = np.hstack([df[efn_cols].values, df[hog_cols].values]).astype(np.float64)
labels = np.array(df["Class"].tolist())

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
LABELS = ["Angry", "Disgust", "Happy", "Neutral", "Sad", "Surprise"]

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=5)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_test_lda = lda.transform(X_test_scaled)

# Verify exported scaler matches sklearn
print("=== Cross-validating exported params with sklearn ===")
sk_min = scaler.min_.tolist()
sk_scale = scaler.scale_.tolist()
min_diff = max(abs(a - b) for a, b in zip(sk_min, scaler_min))
scale_diff = max(abs(a - b) for a, b in zip(sk_scale, scaler_scale))
print(f"Scaler min max-diff: {min_diff:.2e}")
print(f"Scaler scale max-diff: {scale_diff:.2e}")

# Verify LDA
sk_xbar = lda.xbar_.tolist()
sk_scalings = lda.scalings_[:, :5].tolist()
xbar_diff = max(abs(a - b) for a, b in zip(sk_xbar, lda_xbar))
scaling_diff = max(abs(sk_scalings[i][j] - lda_scalings[i][j]) for i in range(len(sk_scalings)) for j in range(5))
print(f"LDA xbar max-diff: {xbar_diff:.2e}")
print(f"LDA scalings max-diff: {scaling_diff:.2e}")

# Test a real sample through both paths
sample = X_test[0:1]
sk_scaled = scaler.transform(sample)[0]
dart_scaled = [sample[0][i] * scaler_scale[i] + scaler_min[i] for i in range(2568)]
scaled_diff = max(abs(a - b) for a, b in zip(sk_scaled, dart_scaled))
print(f"Scaler output max-diff (sklearn vs dart formula): {scaled_diff:.2e}")

sk_lda = lda.transform(sample.reshape(1, -1))[0]  # Use unscaled for sklearn
sk_lda = lda.transform(scaler.transform(sample))[0]
dart_lda = []
for c in range(5):
    s = sum((dart_scaled[i] - lda_xbar[i]) * lda_scalings[i][c] for i in range(2568))
    dart_lda.append(s)
lda_diff = max(abs(a - b) for a, b in zip(sk_lda, dart_lda))
print(f"LDA output max-diff (sklearn vs dart formula): {lda_diff:.2e}")
print(f"  sklearn LDA: {[f'{v:.6f}' for v in sk_lda]}")
print(f"  dart LDA:    {[f'{v:.6f}' for v in dart_lda]}")

print("\n=== Pipeline cross-validation complete! ===")
