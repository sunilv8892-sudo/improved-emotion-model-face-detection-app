# Implementation Guide: Using the Multi-Model Framework

## Quick Start

### Current State
Your app now uses a **pluggable secondary model framework**. The classifier model is already implemented and integrated.

### Available Model Types

#### 1. **Classifier** (Currently Active)
For classification tasks (me/not_me, emotions, objects, etc.)

```dart
secondaryModelType = SecondaryModelType.classifier;
// Automatically loads: assets/models/second_model.tflite
// With labels from: assets/models/second_labels.txt
```

**Usage Example:**
```dart
final result = await secondaryModel!.infer(cropImage);
// result.label → "me" or "not_me"
// result.confidence → 0.0 to 1.0
```

#### 2. **Embedding** (Placeholder - Ready to Implement)
For similarity-based authentication (face verification, etc.)

```dart
secondaryModelType = SecondaryModelType.embedding;
// Load your embedding model:
// assets/models/embedding_model.tflite
```

**Usage Example:**
```dart
final result = await secondaryModel!.infer(cropImage);
// result.embedding → [0.2, -0.1, 0.5, ...] (embedding vector)
// Use cosine similarity to compare with reference
```

---

## How to Switch Models at Runtime

### Option 1: Simple Enum Change (Quick Test)
```dart
// In _YoloDetectionPageState class
SecondaryModelType secondaryModelType = SecondaryModelType.classifier; // Change here

// Then restart the app - it will load the appropriate model
```

### Option 2: Add UI Control (Production Ready)

Add a dropdown in your settings:

```dart
DropdownButton<SecondaryModelType>(
  value: secondaryModelType,
  items: [
    DropdownMenuItem(
      value: SecondaryModelType.classifier,
      child: Text('Classifier (me/not_me)'),
    ),
    DropdownMenuItem(
      value: SecondaryModelType.embedding,
      child: Text('Embedding (Face Verification)'),
    ),
  ],
  onChanged: (newType) {
    if (newType != null) {
      setState(() {
        secondaryModelType = newType;
      });
      // Reload model
      _initEverything();
    }
  },
)
```

---

## Implementing a New Model Type

### Example: Adding OCR Support

#### Step 1: Create OCR Model Class
```dart
class OCRModel implements SecondaryModel {
  final String modelPath;
  Interpreter? _interpreter;
  
  OCRModel({required this.modelPath});
  
  @override
  String get modelName => 'OCR';
  
  @override
  Future<void> load() async {
    _interpreter = await Interpreter.fromAsset(modelPath);
  }
  
  @override
  void dispose() {
    _interpreter?.close();
  }
  
  @override
  Future<SecondaryResult> infer(img.Image crop) async {
    // 1. Preprocess crop image
    final resized = img.copyResize(crop, width: 224, height: 224);
    
    // 2. Prepare input tensor
    final input = Float32List(1 * 224 * 224 * 3);
    // Fill with pixel values...
    
    // 3. Run inference
    final output = Float32List(someOutputSize);
    _interpreter!.run(input, output);
    
    // 4. Decode output to text
    final recognizedText = _decodeOutput(output);
    
    // 5. Return result
    return SecondaryResult(
      rawOutput: {'text': recognizedText}
    );
  }
  
  String _decodeOutput(Float32List output) {
    // Your OCR decoding logic
    return 'extracted text';
  }
}
```

#### Step 2: Add to Enum
```dart
enum SecondaryModelType { classifier, embedding, ocr }
```

#### Step 3: Update Factory Function
```dart
SecondaryModel _createSecondaryModel(SecondaryModelType type) {
  switch (type) {
    case SecondaryModelType.classifier:
      // ... existing classifier code
    
    case SecondaryModelType.embedding:
      // ... existing embedding code
    
    case SecondaryModelType.ocr:
      return OCRModel(modelPath: 'assets/models/ocr_model.tflite');
  }
}
```

#### Step 4: Handle in Inference
```dart
final result = await secondaryModel!.infer(resized);

if (secondaryModelType == SecondaryModelType.ocr) {
  final text = result.rawOutput?['text'] ?? 'no text';
  for (final det in dets) {
    det['ocr_text'] = text;
  }
}
```

**That's it!** No other code needs changes.

---

## Implementing Embedding-Based Face Verification

### Current Status
- `EmbeddingModel` is created but inference logic is not implemented
- Results are extracted but not compared

### Step 1: Implement EmbeddingModel.infer()

Update lines 165-190 in main.dart:

```dart
@override
Future<SecondaryResult> infer(img.Image crop) async {
  // Preprocess (similar to classifier)
  final resized = img.copyResize(crop, width: 224, height: 224);
  
  // Prepare input
  final floats = Float32List(1 * 224 * 224 * 3);
  int idx = 0;
  for (int py = 0; py < 224; py++) {
    for (int px = 0; px < 224; px++) {
      final pixel = resized.getPixel(px, py);
      // Normalize to [-1, 1]
      floats[idx++] = (pixel.r.toDouble() / 127.5) - 1.0;
      floats[idx++] = (pixel.g.toDouble() / 127.5) - 1.0;
      floats[idx++] = (pixel.b.toDouble() / 127.5) - 1.0;
    }
  }
  final input = floats.reshape([1, 224, 224, 3]);
  
  // Run inference
  final embeddingDim = 128; // Adjust based on your model
  final output = Float32List(embeddingDim).reshape([1, embeddingDim]);
  _interpreter!.run(input, output);
  
  // Extract embedding vector
  final embedding = List<double>.from(output[0]);
  
  return SecondaryResult(embedding: embedding);
}
```

### Step 2: Compute Cosine Similarity

Add helper function:

```dart
double cosineSimilarity(List<double> a, List<double> b) {
  double dotProduct = 0;
  double normA = 0;
  double normB = 0;
  
  for (int i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  normA = math.sqrt(normA);
  normB = math.sqrt(normB);
  
  return dotProduct / (normA * normB);
}
```

### Step 3: Store Reference Embedding

```dart
List<double>? _referenceEmbedding; // Store from first detection of user

// In inference:
if (secondaryModelType == SecondaryModelType.embedding) {
  final embedding = result.embedding ?? [];
  
  if (_referenceEmbedding == null && embedding.isNotEmpty) {
    _referenceEmbedding = embedding; // Store reference on first run
    debugPrint('✅ Reference embedding stored');
  }
  
  if (_referenceEmbedding != null && embedding.isNotEmpty) {
    final similarity = cosineSimilarity(_referenceEmbedding!, embedding);
    // similarity: 1.0 = identical, 0.0 = different, -1.0 = opposite
    // Use threshold: 0.6 = "me", < 0.6 = "not me"
    
    final isMe = similarity > 0.6;
    for (final det in dets) {
      det['is_me'] = isMe;
      det['similarity'] = similarity.toStringAsFixed(3);
    }
  }
}
```

---

## Configuration Reference

### Classifier Settings

```dart
// In class variables:
bool _classifierUseBGR = true;           // Try false if colors inverted
bool _classifierUseNormalization_1_1 = false; // [-1,1] vs [0,1]
double _minSecondaryConfidence = 0.7;    // Confidence threshold
int _classifyEveryNFrames = 3;           // Run every 3 frames (saves CPU)
```

### Debugging

```dart
bool _showDebugInfo = true; // Enable in _YoloDetectionPageState init

// Debug output shows:
// - Model type and name
// - Tensor shapes and types
// - Raw logits (before softmax)
// - Probabilities (after softmax)
// - Final prediction and confidence
// - Smoothing decisions
```

---

## Testing Checklist

### Before Deploying

- [ ] Classifier still detects me/not_me correctly
- [ ] Prediction smoothing prevents flickering
- [ ] YOLO detection still works (hasn't regressed)
- [ ] No console errors when switching models
- [ ] Debug output shows appropriate model info
- [ ] App doesn't crash when secondary model fails to load

### When Adding New Models

- [ ] New model class implements `SecondaryModel` interface
- [ ] `load()` properly initializes interpreter
- [ ] `infer()` returns correct `SecondaryResult` fields
- [ ] `dispose()` cleans up resources
- [ ] Added to enum and factory function
- [ ] Result handling in `_runSecondaryInference()` is correct
- [ ] Debug output shows model-specific information

---

## Troubleshooting

### Model not loading
```
Second model not loaded (optional): Exception...
```
**Fix:** Check file path and that model exists in assets

### Wrong color output
```
Colors look inverted or wrong
```
**Fix:** Toggle `_classifierUseBGR = !_classifierUseBGR`

### Poor accuracy
```
Confidence always low or always high
```
**Fix:** Toggle `_classifierUseNormalization_1_1` between true/false

### Crashes during inference
```
Secondary inference error: ...
```
**Check:** Model input size matches preprocessing size (currently 224x224)

---

## Files Modified

- `lib/main.dart` - Framework classes, refactored inference
- `REFACTORING_SUMMARY.md` - Technical details
- `FRAMEWORK_GUIDE.md` - This file

## Next Immediate Steps

1. Test classifier mode (should work as before)
2. Implement embedding model inference logic
3. Add UI for switching models
4. Deploy and verify on device
