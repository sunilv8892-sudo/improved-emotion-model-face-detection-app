import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import 'emotion_feature_extractor.dart';
import 'emotion_model_parameters.dart';
import 'hog_feature_extractor.dart';
import 'svm_classifier.dart';

class EmotionEngineResult {
  final String label;
  final double confidence;
  final Map<String, double> probabilities;
  final List<double> efficientnetFeatures;
  final Float32List hogFeatures;
  final List<double> poseFeatures;
  final List<double> ldaFeatures;

  const EmotionEngineResult({
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.efficientnetFeatures,
    required this.hogFeatures,
    required this.poseFeatures,
    required this.ldaFeatures,
  });
}

class EmotionEngine {
  static const List<String> labels = [
    'Angry',
    'Disgust',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise',
  ];

  static final EmotionEngine _instance = EmotionEngine._internal();
  factory EmotionEngine() => _instance;
  EmotionEngine._internal();

  final EmotionFeatureExtractor _featureExtractor = EmotionFeatureExtractor();
  final HogFeatureExtractor _hogExtractor = HogFeatureExtractor();

  EmotionModelParameters? _parameters;
  EmotionSvmClassifier? _svmClassifier;
  bool _isInitialized = false;

  Future<void> initialize() async {
    if (_isInitialized && _parameters != null && _svmClassifier != null) return;

    await _featureExtractor.initialize();
    _parameters = await EmotionModelParameters.load();
    _svmClassifier = EmotionSvmClassifier(_parameters!.svm);
    _isInitialized = true;
  }

  bool get isReady =>
      _isInitialized && _parameters != null && _svmClassifier != null && _featureExtractor.isReady;

  Future<EmotionEngineResult> predictEmotion(
    Uint8List faceImageBytes, {
    double poseX = 0.0,
    double poseY = 0.0,
  }) async {
    if (!isReady) {
      await initialize();
    }

    final parameters = _parameters;
    final svmClassifier = _svmClassifier;
    if (parameters == null || svmClassifier == null) {
      throw Exception('Emotion engine is not initialized');
    }

    final efficientnetFeatures = await _featureExtractor.extractFeatures(faceImageBytes);
    final hogFeatures = parameters.hogFeatureCount > 0
        ? _hogExtractor.extractFromBytes(faceImageBytes)
        : Float32List(0);
    final poseFeatures = parameters.poseFeatureCount == 0
      ? const <double>[]
      : <double>[poseX, poseY];

    _validateFeatureDimensions(parameters, efficientnetFeatures, hogFeatures, poseFeatures);

    final combinedFeatures = <double>[
      ...efficientnetFeatures,
      ...hogFeatures,
      ...poseFeatures,
    ];

    // Diagnostic: log feature stats on first few calls
    debugPrint(
      '📊 Features: EFN[${efficientnetFeatures.length}] '
      'HOG[${hogFeatures.length}] '
      'Total[${combinedFeatures.length}]',
    );

    final scaled = parameters.scaler.transform(combinedFeatures);
    final ldaFeatures = parameters.lda.transform(scaled);
    final prediction = svmClassifier.predict(ldaFeatures);

    debugPrint(
      '🎭 Emotion: ${prediction.label} (${(prediction.confidence * 100).toStringAsFixed(1)}%) '
      'Raw: ${prediction.rawScores.entries.map((e) => '${e.key}=${e.value.toStringAsFixed(2)}').join(', ')}',
    );

    return EmotionEngineResult(
      label: prediction.label,
      confidence: prediction.confidence,
      probabilities: prediction.probabilities,
      efficientnetFeatures: efficientnetFeatures,
      hogFeatures: hogFeatures,
      poseFeatures: poseFeatures,
      ldaFeatures: ldaFeatures,
    );
  }

  void _validateFeatureDimensions(
    EmotionModelParameters parameters,
    List<double> efficientnetFeatures,
    List<double> hogFeatures,
    List<double> poseFeatures,
  ) {
    if (efficientnetFeatures.length != parameters.efficientnetFeatureCount) {
      throw Exception(
        'EfficientNet feature mismatch: model produced ${efficientnetFeatures.length}, '
        'but runtime expects ${parameters.efficientnetFeatureCount}. '
        'Regenerate the CSV features and retrain with the exported TFLite extractor.',
      );
    }

    if (hogFeatures.length != parameters.hogFeatureCount) {
      throw Exception(
        'HOG feature mismatch: extractor produced ${hogFeatures.length}, '
        'but runtime expects ${parameters.hogFeatureCount}.',
      );
    }

    if (poseFeatures.length != parameters.poseFeatureCount) {
      throw Exception(
        'Pose feature mismatch: got ${poseFeatures.length}, '
        'expected ${parameters.poseFeatureCount}.',
      );
    }
  }

  void dispose() {
    _featureExtractor.dispose();
    _parameters = null;
    _svmClassifier = null;
    _isInitialized = false;
  }
}