# Multi-Model YOLO + Classifier Framework - Optimization Guide

## Current Status
✅ **YOLO Face Detection** - Enabled  
✅ **Softmax Classifier** - Enabled with normalized probabilities  
✅ **Real-time Confidence Adjustment** - Available via slider  

---

## Problem: YOLO Detecting 0 Faces

### Quick Diagnosis Flowchart

```
Are you seeing face detection boxes? 
  ├─ NO → Confidence threshold too high
  ├─ YES → Adjust classifier preprocessing
```

---

## Solution 1: Lower YOLO Confidence Threshold

**What it is:** YOLO skips boxes with confidence below this value  
**Current value:** See the slider in top-right of app (default 25%)  
**Action:** Drag slider LEFT to detect more faces (lower threshold)

### Try these values:
- **0.15 (15%)** - Very loose, lots of false positives
- **0.20 (20%)** - More lenient
- **0.25 (25%)** - Current default
- **0.35 (35%)** - Stricter

### How to test:
1. Run app: `flutter run`
2. Point camera at a face
3. Adjust slider in real-time
4. Watch console for: `Detection completed: X objects found`

---

## Solution 2: Fix Classifier Preprocessing (If faces detected but classification wrong)

The classifier has 4 main settings to try in Settings Menu:

### Setting 1: BGR vs RGB Color Order
- **Toggle `_useBGR`**
- Some models expect BGR order instead of RGB
- Try both, see which gives better results

### Setting 2: Normalization Range
- **Toggle `_useNormalization_1_1`**
- RGB mode: `[0, 1]` - divides by 255
- [-1, 1] mode: `(x / 127.5) - 1.0` - centered around 0
- Check your training code to match

### Setting 3: Crop Padding
- **Adjust `_cropPaddingPercent`** (currently 20%)
- Add more context around face: increase to 30-40%
- Show less context: decrease to 10%

### Setting 4: Confidence Threshold (Classifier)
- **Adjust `_minClassifierConfidence`** (currently 0.5)
- This is DIFFERENT from YOLO threshold
- Threshold below which we keep previous prediction
- Lower = stickier predictions

---

## Understanding the Console Output

### When YOLO works (prints like this):
```
D/FlutterVisionPlugin(17735): Detection completed: 2 objects found
```

### When Classifier runs (look for this):
```
========== CLASSIFIER DEBUG ==========
Input tensor type: TensorType.float32
Crop size: 245x240 -> 224x224
BGR mode: false | Normalization [-1,1]: false
Classifier logits: [-2.3, 5.1, -0.8]
Classifier probabilities: [0.0012, 0.9985, 0.0003]
Classifier result: idx=1, conf=0.999, label=me
=====================================
```

### Interpreting logits vs probabilities:
- **Logits:** Raw neural network output (-inf to +inf)
- **Probabilities:** After softmax (0.0 to 1.0)
- Softmax converts logits to valid probability distribution

---

## Troubleshooting Checklist

### Issue: YOLO finds 0 faces
- [ ] Lower confidence slider (top-right)
- [ ] Ensure good lighting
- [ ] Face should be 40-80% of screen
- [ ] Try values: 15%, 20%, 25%

### Issue: YOLO finds faces but classifier always same label
- [ ] Check BGR/RGB setting (toggle `_useBGR`)
- [ ] Check normalization (toggle `_useNormalization_1_1`)
- [ ] Increase crop padding (`_cropPaddingPercent` → 30-40%)
- [ ] Check logits output - all similar values = bad preprocessing

### Issue: Classifier confidence always low
- [ ] Increase `_minClassifierConfidence` slowly (try 0.3, 0.4)
- [ ] Check if logits ever exceed 2.0 - if not, model not learning
- [ ] Verify training data used same preprocessing

### Issue: Jerky/unstable predictions
- [ ] Increase `_smoothingWindow` (currently 5)
- [ ] Increase `_classifyEveryNFrames` to 5-10

---

## Model Training Checklist

If you trained a custom model, ensure you used:

```python
# Python training code MUST match app preprocessing:

# 1. Input size must be 224x224 (change in app if different)
input_shape = (224, 224, 3)

# 2. Normalization (check which you used in training):
# Option A: [0, 1] normalization
img = img.astype('float32') / 255.0

# Option B: [-1, 1] normalization  
img = (img.astype('float32') / 127.5) - 1.0

# 3. Color order (check if BGR or RGB):
# RGB (default): R, G, B channels as-is
# BGR: B, G, R channels swapped

# 4. Data augmentation used during training:
# - Random flips
# - Random rotations
# - Random brightness/contrast
```

---

## App Settings Reference

### Location: Settings Menu (top-left button)

| Setting | Current | Range | Effect |
|---------|---------|-------|--------|
| `confidenceThreshold` | 0.25 | 0.1-0.9 | YOLO detection threshold |
| `iouThreshold` | 0.4 | 0.1-0.9 | NMS (duplicate removal) |
| `_useBGR` | false | true/false | BGR vs RGB color order |
| `_useNormalization_1_1` | false | true/false | Normalization range |
| `_cropPaddingPercent` | 20 | 0-50 | Context padding % |
| `_minClassifierConfidence` | 0.5 | 0.0-1.0 | Min confidence threshold |
| `_smoothingWindow` | 5 | 1-20 | Prediction smoothing |
| `_classifyEveryNFrames` | 3 | 1-10 | Skip frames for speed |

---

## Performance Tips

1. **Skip frames** - Set `_classifyEveryNFrames = 5` to run classifier every 5 frames
2. **Lower resolution** - Process on smaller images
3. **GPU** - Enable GPU in settings if device supports
4. **Threading** - TFLite uses 1 thread by default, increase carefully

---

## Next Steps

1. Run: `flutter run`
2. Adjust YOLO slider until faces detected
3. Note down which threshold worked (e.g., 20%)
4. If classification bad:
   - Toggle BGR setting
   - Toggle Normalization setting
   - Check console logits output
5. If logits look wrong, retrain model with matching preprocessing

---

## Need More Help?

Check console output for:
```
========== CLASSIFIER DEBUG ==========
```

Share the logits and probabilities values - they tell us exactly what went wrong!

Example good output:
```
Classifier probabilities: [0.001, 0.997, 0.002]  // One class ~1.0, others ~0.0
```

Example bad output:
```
Classifier probabilities: [0.5, 0.5, 0.0]  // Multiple ~0.5 = confusion
```
