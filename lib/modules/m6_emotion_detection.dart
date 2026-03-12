import 'dart:typed_data';

import 'package:image/image.dart' as img;

import 'hog_feature_extractor.dart';
import 'emotion_engine.dart';

/// M6: Emotion Detection Module — EfficientNet + HOG + pose + scaler + LDA + SVM
///
/// This keeps the identity pipeline untouched and runs the emotion branch in
/// parallel using the research-paper workflow:
///
///   EfficientNet features + HOG features + pose angles
///   -> MinMaxScaler
///   -> LDA(5)
///   -> RBF SVM
class EmotionDetectionModule {
  // ── Singleton ──
  static final EmotionDetectionModule _instance =
      EmotionDetectionModule._internal();
  factory EmotionDetectionModule() => _instance;
  EmotionDetectionModule._internal();

  final HogFeatureExtractor _hogExtractor = HogFeatureExtractor();
  final EmotionEngine _engine = EmotionEngine();

  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      await _engine.initialize();
      _isInitialized = true;
    } catch (e) {
      _isInitialized = false;
      rethrow;
    }
  }

  bool get isReady => _isInitialized && _engine.isReady;

  Future<EmotionDetectionResult?> detectEmotion(
    Uint8List faceImageBytes, {
    double poseX = 0.0,
    double poseY = 0.0,
  }) async {
    if (!isReady) await initialize();

    try {
      final engineResult = await _engine.predictEmotion(
        faceImageBytes,
        poseX: poseX,
        poseY: poseY,
      );

      return EmotionDetectionResult(
        label: engineResult.label,
        confidence: engineResult.confidence,
        probabilities: engineResult.probabilities,
        hogDescriptor: engineResult.hogFeatures,
      );
    } catch (e) {
      print('EmotionDetectionModule.detectEmotion error: $e');
      return null;
    }
  }

  /// Convenience: detect emotion from an already-decoded image.
  Future<EmotionDetectionResult?> detectEmotionFromImage(
    img.Image faceImage,
    {
    double poseX = 0.0,
    double poseY = 0.0,
  }) async {
    final bytes = Uint8List.fromList(img.encodeJpg(faceImage, quality: 90));
    return detectEmotion(bytes, poseX: poseX, poseY: poseY);
  }

  /// Classify with a fallback when the TFLite model is not available.
  /// Uses a HOG-based heuristic (gradient energy ratios) as a rough
  /// fallback — much less accurate than EfficientNet but still better
  /// than the old smiling-probability approach.
  Future<EmotionDetectionResult> detectEmotionWithFallback(
    Uint8List faceImageBytes,
    {
    double poseX = 0.0,
    double poseY = 0.0,
  }
  ) async {
    final result = await detectEmotion(
      faceImageBytes,
      poseX: poseX,
      poseY: poseY,
    );
    if (result != null) return result;

    return _hogFallback(faceImageBytes);
  }

  /// Rough HOG-based fallback when no TFLite model is available.
  /// Analyses the gradient energy distribution to estimate emotion.
  EmotionDetectionResult _hogFallback(Uint8List faceImageBytes) {
    try {
      final hog = _hogExtractor.extractFromBytes(faceImageBytes);

      // Simple heuristic based on overall gradient energy.
      // This is NOT accurate — purely a graceful-degradation path.
      double totalEnergy = 0;
      double maxBinEnergy = 0;
      for (final v in hog) {
        totalEnergy += v;
        if (v > maxBinEnergy) maxBinEnergy = v;
      }

      final avgEnergy = hog.isNotEmpty ? totalEnergy / hog.length : 0.0;

      String label;
      double confidence;
      if (avgEnergy > 0.15) {
        label = 'Surprise';
        confidence = 0.30;
      } else if (avgEnergy > 0.10) {
        label = 'Happy';
        confidence = 0.30;
      } else if (avgEnergy < 0.04) {
        label = 'Sad';
        confidence = 0.25;
      } else {
        label = 'Neutral';
        confidence = 0.35;
      }

      final fallbackLabels = EmotionEngine.labels;
      final residual = fallbackLabels.length > 1
          ? (1.0 - confidence) / (fallbackLabels.length - 1)
          : 0.0;
      final probs = <String, double>{
        for (final currentLabel in fallbackLabels)
          currentLabel: currentLabel == label ? confidence : residual,
      };

      return EmotionDetectionResult(
        label: label,
        confidence: confidence,
        probabilities: probs,
        hogDescriptor: hog,
        isFallback: true,
      );
    } catch (_) {
      final fallbackLabels = EmotionEngine.labels;
      return EmotionDetectionResult(
        label: 'Neutral',
        confidence: 0.0,
        probabilities: {
          for (final currentLabel in fallbackLabels)
            currentLabel: 1.0 / fallbackLabels.length,
        },
        hogDescriptor: Float32List(0),
        isFallback: true,
      );
    }
  }

  void dispose() {
    _engine.dispose();
    _isInitialized = false;
  }
}

/// Result of the HOG + EfficientNet emotion detection.
class EmotionDetectionResult {
  /// Predicted emotion label
  final String label;

  /// Confidence of the top prediction [0, 1]
  final double confidence;

  /// Full probability map over the configured emotions
  final Map<String, double> probabilities;

  /// Raw HOG feature descriptor (can be logged / used elsewhere)
  final Float32List hogDescriptor;

  /// True when the result came from the HOG-only fallback path
  final bool isFallback;

  const EmotionDetectionResult({
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.hogDescriptor,
    this.isFallback = false,
  });

  @override
  String toString() =>
      'EmotionDetectionResult($label, ${(confidence * 100).toStringAsFixed(1)}%'
      '${isFallback ? ', fallback' : ''})';
}
