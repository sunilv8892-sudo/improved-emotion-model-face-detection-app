#!/usr/bin/env python3
import json

json_path = 'assets/models/emotion_runtime_params.json'

# Load JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Check if class_biases exist
if 'class_biases' in data.get('svm', {}):
    print("✓ class_biases ALREADY EXISTS in JSON!")
    print("Biases:", data['svm']['class_biases'])
else:
    print("✗ class_biases NOT found! Adding now...")
    
    # Add class_biases
    biases = {
        "Angry": -0.1,
        "Disgust": 0.0,
        "Happy": 0.0,
        "Neutral": 0.3,
        "Sad": 0.0,
        "Surprise": 0.0
    }
    
    if 'svm' not in data:
        data['svm'] = {}
    
    data['svm']['class_biases'] = biases
    
    # Write back
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print("✓ Added class_biases to JSON:")
    print("  - Neutral: +0.3 (boost detection)")
    print("  - Angry: -0.1 (reduce false positives)")
    print("  - Others: 0.0 (no change)")
    print(f"✓ Saved to {json_path}")
