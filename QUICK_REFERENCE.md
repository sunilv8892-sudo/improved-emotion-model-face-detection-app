# 🎯 Multi-Model Framework - Quick Reference

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  Flutter Camera App                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────┐         ┌──────────────────────┐  │
│  │  Camera Input    │         │   YOLO Detection     │  │
│  │  (Live Frames)   │──────→  │  (Find Faces)        │  │
│  └──────────────────┘         └──────────────────────┘  │
│           ▲                             │                │
│           │                             ▼                │
│           │                    ┌──────────────────────┐  │
│           └────────────────────│  Secondary Model     │  │
│                                │  (Abstract)          │  │
│                                └──────────────────────┘  │
│                                         │                │
│                    ┌────────────────────┼────────────────┬──┐
│                    ▼                    ▼                │  │
│         ┌──────────────────┐  ┌──────────────────┐     │  │
│         │  Classifier      │  │  Embedding       │     │  │
│         │  (me/not_me)     │  │  (Face Verify)   │     │  │
│         └──────────────────┘  └──────────────────┘     │  │
│                    │                    │              │  │
│                    ▼                    ▼              ▼  │
│         ┌──────────────────┐  ┌──────────────────┐     │  │
│         │ Label +          │  │ Embedding        │ ... │  │
│         │ Confidence       │  │ Vector           │ ... │  │
│         └──────────────────┘  └──────────────────┘     │  │
│                                                        ▼  │
│                              ┌──────────────────────────┐ │
│                              │  UI Overlay              │ │
│                              │  (Show Results)          │ │
│                              └──────────────────────────┘ │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## File Locations in main.dart

```
Line 15-20    | softmax() function
Line 28       | enum SecondaryModelType
Line 27-34    | abstract class SecondaryModel
Line 36-48    | class SecondaryResult
Line 54-214   | class ClassifierModel implements SecondaryModel
Line 216-275  | class EmbeddingModel implements SecondaryModel
Line 514-530  | SecondaryModel _createSecondaryModel()
Line 580-699  | Future<void> _runSecondaryInference()
```

## How It Works

### 1. Initialization
```
App Start
  → _initEverything()
    → Load YOLO model
    → Create SecondaryModel via factory
    → Apply configuration
    → Ready!
```

### 2. Detection Loop
```
Camera Frame (30fps)
  → YOLO Detection (every frame)
    → Face found? YES → Crop region
    → Secondary Model inference (every 3rd frame)
      → ClassifierModel? → Extract label + confidence
      → EmbeddingModel? → Extract embedding vector
    → Update UI with results
```

### 3. Model Switching (Pluggable Design)
```
Old Way (Hardcoded):
  if (useClassifier) {
    load classifier interpreter
    run classifier logic
    handle classifier results
  }
  
  if (useEmbedding) {
    load embedding interpreter
    run embedding logic
    handle embedding results
  }
  // Duplicated logic, hard to maintain

New Way (Framework):
  model = factory.create(selectedType)
  await model.load()
  
  result = await model.infer(crop)
  // Handle result based on type
  
  // Same code for all model types!
  // Adding new model = 0 changes to inference logic
```

## Configuration Reference

### Runtime Settings (Change & Restart)

```dart
// In _YoloDetectionPageState class init

// Select model type
SecondaryModelType secondaryModelType = SecondaryModelType.classifier;
// Options: classifier, embedding

// Classifier-specific settings
bool _classifierUseBGR = true;                    // Color order
bool _classifierUseNormalization_1_1 = false;     // Normalization type
double _minSecondaryConfidence = 0.7;             // Confidence threshold

// Performance
int _classifyEveryNFrames = 3;                    // Run every Nth frame
int _smoothingWindow = 5;                         // Prediction smoothing

// Debug
bool _showDebugInfo = true;                       // Print to console
```

### Model Paths (Can customize)

```dart
// In factory function
case SecondaryModelType.classifier:
  return ClassifierModel(
    modelPath: 'assets/models/second_model.tflite',      // Edit here
    labelsPath: 'assets/models/second_labels.txt',        // Edit here
  );
```

## Common Tasks

### Task 1: Switch to Embedding Mode

```dart
// Change this line:
SecondaryModelType secondaryModelType = SecondaryModelType.classifier;

// To this:
SecondaryModelType secondaryModelType = SecondaryModelType.embedding;

// Restart app - framework handles the rest
```

### Task 2: Implement Face Verification

1. Update `EmbeddingModel.infer()` to return embedding
2. Add similarity comparison function:
```dart
double cosineSimilarity(List<double> a, List<double> b) {
  // See FRAMEWORK_GUIDE.md for implementation
}
```
3. Store reference embedding on first detection
4. Compare live embeddings with reference

### Task 3: Add OCR Support

1. Create class:
```dart
class OCRModel implements SecondaryModel { ... }
```

2. Add to enum:
```dart
enum SecondaryModelType { classifier, embedding, ocr }
```

3. Add factory case:
```dart
case SecondaryModelType.ocr:
  return OCRModel(...);
```

4. Handle in inference:
```dart
if (secondaryModelType == SecondaryModelType.ocr) {
  final text = result.rawOutput?['text'];
}
```

**That's it!** No changes to inference loop or UI logic needed.

## Debug Output

### Classifier Mode
```
========== CLASSIFIERMODEL DEBUG ==========
Crop size: 224x224 -> 224x224
Result: label=me, conf=0.891
Smoothed: me
=====================================
```

### Embedding Mode
```
========== EMBEDDINGMODEL DEBUG ==========
Embedding dim: 128
First 5 values: [0.2, -0.1, 0.5, 0.8, -0.3]
=====================================
```

## Testing Checklist

- [ ] App starts without crashes
- [ ] YOLO detects faces (red bounding boxes)
- [ ] Classifier runs every 3rd frame
- [ ] Labels appear on detections
- [ ] Prediction smoothing prevents flickering
- [ ] Can switch models via enum change
- [ ] Debug output shows correct model type
- [ ] No console errors

## Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Model not found | Wrong asset path | Check path in factory |
| Always low confidence | Wrong normalization | Toggle `_classifierUseNormalization_1_1` |
| Colors inverted | BGR mismatch | Toggle `_classifierUseBGR` |
| Crashes on infer | Tensor shape mismatch | Check model input size |
| No debug output | Debug disabled | Set `_showDebugInfo = true` |

## Performance Tips

- ✅ Already optimized: Run classifier every 3 frames (saves CPU)
- ✅ Already optimized: Process only first detection (saves time)
- ✅ Already optimized: Prediction smoothing (reduced flickering)
- 📝 Future: Consider frame skipping for embedding model
- 📝 Future: Cache model between mode switches

## Documentation Files

| File | Purpose |
|------|---------|
| `IMPLEMENTATION_STATUS.md` | Current state & verification |
| `REFACTORING_SUMMARY.md` | Technical architecture details |
| `FRAMEWORK_GUIDE.md` | Implementation guide for new features |
| This file | Quick reference & common tasks |

## Ready to Deploy?

✅ Framework is production-ready when:
- [x] Classifier mode tested
- [ ] Embedding mode implemented (if needed)
- [ ] Custom models integrated (if needed)
- [ ] UI shows appropriate metrics
- [ ] Performance is acceptable on target device

---

**Last Updated:** After refactoring completion  
**Framework Status:** ✅ READY FOR TESTING
