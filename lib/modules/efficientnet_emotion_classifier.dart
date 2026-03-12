import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// EfficientNet-B0 based Emotion Classifier
///
/// Runs a TFLite EfficientNet-B0 model (fine-tuned on FER-2013 / AffectNet)
/// to classify a face ROI into one of 7 emotion categories.
///
/// Model input : 1 × 48 × 48 × 1  (grayscale) **or** 1 × 48 × 48 × 3 (RGB)
///               depending on the model variant.  The code auto-detects the
///               channel count from the loaded model's input tensor shape.
/// Model output: 1 × 7  (softmax probabilities)
///
/// Emotion labels (FER-2013 convention, index order):
///   0 = Angry, 1 = Disgust, 2 = Fear, 3 = Happy,
///   4 = Sad,   5 = Surprise, 6 = Neutral
class EfficientNetEmotionClassifier {
  static const String modelAssetPath = 'assets/models/emotion_model.tflite';

  /// Canonical FER-2013 emotion label order
  static const List<String> emotionLabels = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Sad',
    'Surprise',
    'Neutral',
  ];

  // ── Singleton ──
  static final EfficientNetEmotionClassifier _instance =
      EfficientNetEmotionClassifier._internal();
  factory EfficientNetEmotionClassifier() => _instance;
  EfficientNetEmotionClassifier._internal();

  Interpreter? _interpreter;
  bool _isInitialized = false;

  /// Model input dimensions inferred at load time
  int _inputHeight = 48;
  int _inputWidth = 48;
  int _inputChannels = 1; // grayscale default; auto-detected

  /// Initialize the TFLite interpreter from the bundled asset.
  Future<void> initialize() async {
    if (_isInitialized && _interpreter != null) return;
    try {
      final options = InterpreterOptions()..threads = 2;
      _interpreter = await Interpreter.fromAsset(
        modelAssetPath,
        options: options,
      );
      _interpreter?.allocateTensors();

      // Read input tensor shape: [1, H, W, C]
      final inputShape = _interpreter!.getInputTensor(0).shape;
      if (inputShape.length >= 4) {
        _inputHeight = inputShape[1];
        _inputWidth = inputShape[2];
        _inputChannels = inputShape[3];
      }

      // Validate output — should have 7 classes
      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final outLen = outputShape.fold<int>(1, (a, b) => a * b);
      if (outLen != emotionLabels.length) {
        throw Exception(
          'Emotion model output size $outLen ≠ expected ${emotionLabels.length}',
        );
      }

      _isInitialized = true;
    } catch (e) {
      _interpreter?.close();
      _interpreter = null;
      _isInitialized = false;
      rethrow;
    }
  }

  bool get isReady => _isInitialized && _interpreter != null;

  /// Classify the emotion of a cropped face image.
  ///
  /// Returns an [EmotionResult] containing the predicted label, its
  /// confidence, and the full probability distribution.
  Future<EmotionResult?> classify(Uint8List faceImageBytes) async {
    if (!isReady) await initialize();
    if (_interpreter == null) return null;

    try {
      final decoded = img.decodeImage(faceImageBytes);
      if (decoded == null) return null;

      final resized = img.copyResize(
        decoded,
        width: _inputWidth,
        height: _inputHeight,
      );

      // Build the input tensor
      final inputBuffer =
          Float32List(_inputHeight * _inputWidth * _inputChannels);
      int idx = 0;

      for (int y = 0; y < _inputHeight; y++) {
        for (int x = 0; x < _inputWidth; x++) {
          final pixel = resized.getPixel(x, y);
          if (_inputChannels == 1) {
            // Grayscale (luminance) normalised to [0, 1]
            final gray =
                (0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b) / 255.0;
            inputBuffer[idx++] = gray;
          } else {
            // RGB normalised to [0, 1]
            inputBuffer[idx++] = pixel.r / 255.0;
            inputBuffer[idx++] = pixel.g / 255.0;
            inputBuffer[idx++] = pixel.b / 255.0;
          }
        }
      }

      final inputData = inputBuffer.reshape(
        [1, _inputHeight, _inputWidth, _inputChannels],
      );
      final outputBuffer = List.generate(
        1,
        (_) => List.filled(emotionLabels.length, 0.0),
      );

      _interpreter!.run(inputData, outputBuffer);

      final probabilities = (outputBuffer.first as List)
          .map((v) => (v as num).toDouble())
          .toList();

      // Apply softmax if the model outputs logits instead of probabilities
      final probs = _ensureProbabilities(probabilities);

      int bestIdx = 0;
      double bestVal = probs[0];
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > bestVal) {
          bestVal = probs[i];
          bestIdx = i;
        }
      }

      return EmotionResult(
        label: emotionLabels[bestIdx],
        confidence: bestVal,
        probabilities: Map.fromIterables(emotionLabels, probs),
      );
    } catch (e) {
      // ignore: avoid_print
      print('EfficientNetEmotionClassifier.classify error: $e');
      return null;
    }
  }

  /// Cheap softmax (in case the model emits raw logits)
  List<double> _ensureProbabilities(List<double> values) {
    // Already sums to ≈1?  Then it's probably already softmax.
    double sum = 0;
    for (final v in values) {
      sum += v;
    }
    if ((sum - 1.0).abs() < 0.05 && values.every((v) => v >= 0)) {
      return values;
    }

    // Apply softmax
    double maxVal = values[0];
    for (final v in values) {
      if (v > maxVal) maxVal = v;
    }
    final exps = values.map((v) {
      final e = v - maxVal; // for numerical stability
      return e > -80 ? _exp(e) : 0.0;
    }).toList();
    double expSum = 0;
    for (final e in exps) {
      expSum += e;
    }
    if (expSum == 0) expSum = 1;
    return exps.map((e) => e / expSum).toList();
  }

  /// Simple exponential using Dart's built-in pow
  double _exp(double x) {
    // dart:math.exp equivalent hand-off to dart:math
    return x.isFinite ? _pow(2.718281828459045, x) : 0.0;
  }

  double _pow(double base, double exponent) {
    if (exponent == 0) return 1.0;
    if (exponent == 1) return base;
    // Use successive squaring for integer exponents, otherwise fallback
    return _dartExp(exponent * _ln(base));
  }

  /// Natural log using Dart arithmetic
  double _ln(double x) {
    if (x <= 0) return double.negativeInfinity;
    // Newton–Raphson: ln(x) via series
    // For speed just import dart:math
    return _dartLn(x);
  }

  // Fallback to dart:math functions for transcendentals
  static double _dartExp(double x) {
    // Taylor series up to sufficient terms
    double result = 1.0;
    double term = 1.0;
    for (int n = 1; n <= 30; n++) {
      term *= x / n;
      result += term;
    }
    return result;
  }

  static double _dartLn(double x) {
    if (x <= 0) return double.negativeInfinity;
    // Use identity: ln(x) = 2 * atanh((x-1)/(x+1))
    final z = (x - 1) / (x + 1);
    double sum = 0;
    double power = z;
    for (int n = 0; n < 40; n++) {
      sum += power / (2 * n + 1);
      power *= z * z;
    }
    return 2 * sum;
  }

  /// Dispose interpreter resources.
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }
}

/// Result of emotion classification.
class EmotionResult {
  /// Predicted emotion label (e.g. 'Happy', 'Sad', …)
  final String label;

  /// Confidence (probability) of the predicted label, in [0, 1]
  final double confidence;

  /// Full probability distribution over all 7 emotions
  final Map<String, double> probabilities;

  const EmotionResult({
    required this.label,
    required this.confidence,
    required this.probabilities,
  });

  @override
  String toString() =>
      'EmotionResult($label, ${(confidence * 100).toStringAsFixed(1)}%)';
}
