"""Check CSV feature statistics vs TFLite runtime output."""
import pandas as pd
import numpy as np
import os
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# ── Check CSV features ──
df = pd.read_csv("training/EfficientNetb0_HOG_pose_FM (1).csv")
efn = df[[str(i) for i in range(1000)]].values

print("=== CSV EfficientNet Feature Stats ===")
row_sums = efn.sum(axis=1)
print(f"  Row sums: min={row_sums.min():.2f} max={row_sums.max():.2f} mean={row_sums.mean():.2f}")
print(f"  Value range: min={efn.min():.6f} max={efn.max():.6f}")
print(f"  Mean per feature: min={efn.mean(axis=0).min():.6f} max={efn.mean(axis=0).max():.6f}")
print(f"  Sample row 0: sum={efn[0].sum():.4f}")
print(f"    first 10: {efn[0,:10]}")
print(f"    max value in row: {efn[0].max():.6f}")

# Check if features look like softmax probs (sum≈1) or raw features
if abs(row_sums.mean() - 1.0) < 0.1:
    print("\n  >>> Features look like softmax probabilities (sum≈1.0)")
    print("  >>> This matches include_top=True output")
else:
    print(f"\n  >>> Features do NOT sum to 1.0 (mean sum={row_sums.mean():.2f})")
    print("  >>> This might NOT be include_top=True softmax output")
    if efn.max() > 1.0:
        print("  >>> Values > 1.0 found — these are NOT softmax probabilities")
        print("  >>> May be include_top=False (raw features before softmax)")

# ── Check TFLite output for comparison ──
try:
    import tensorflow as tf
    print("\n=== TFLite Model Output Check ===")
    interpreter = tf.lite.Interpreter(model_path="assets/models/efficientnet_feature_extractor.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"  Input shape: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}, dtype: {output_details[0]['dtype']}")
    
    # Feed a random "face-like" image (skin-tone pixels)
    np.random.seed(42)
    test_input = np.random.randint(100, 200, (1, 224, 224, 3)).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    tflite_output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    print(f"  TFLite output sum: {tflite_output.sum():.4f}")
    print(f"  TFLite output range: [{tflite_output.min():.6f}, {tflite_output.max():.6f}]")
    print(f"  TFLite first 10: {tflite_output[:10]}")
    
    if abs(tflite_output.sum() - 1.0) < 0.01:
        print("\n  >>> TFLite output is softmax (sums to ~1.0)")
    else:
        print(f"\n  >>> TFLite output sum = {tflite_output.sum():.4f} (NOT softmax)")
except Exception as e:
    print(f"\nTFLite check failed: {e}")

print("\n=== DIAGNOSIS ===")
if row_sums.mean() > 10:
    print("The CSV features are NOT softmax probabilities.")
    print("They are likely raw features from include_top=False (before softmax).")
    print("But we exported include_top=True TFLite which outputs softmax probs.")
    print("SOLUTION: Export include_top=False TFLite model to match the CSV features.")
elif abs(row_sums.mean() - 1.0) < 0.1:
    print("Both CSV and TFLite should produce softmax probabilities.")
    print("If mismatch persists, it may be a preprocessing difference.")
