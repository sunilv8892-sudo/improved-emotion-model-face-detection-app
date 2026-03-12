import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

/// EfficientNet-B0 feature extractor used by the emotion pipeline.
///
/// The model is expected to be exported from the Python training pipeline as:
/// `assets/models/efficientnet_feature_extractor.tflite`
class EmotionFeatureExtractor {
  static const String modelAssetPath =
      'assets/models/efficientnet_feature_extractor.tflite';

  static final EmotionFeatureExtractor _instance =
      EmotionFeatureExtractor._internal();
  factory EmotionFeatureExtractor() => _instance;
  EmotionFeatureExtractor._internal();

  Interpreter? _interpreter;
  bool _isInitialized = false;
  int _inputHeight = 224;
  int _inputWidth = 224;
  int _inputChannels = 3;
  int _outputLength = 0;

  Future<void> initialize() async {
    if (_isInitialized && _interpreter != null) return;

    final options = InterpreterOptions()..threads = 4;
    _interpreter = await Interpreter.fromAsset(modelAssetPath, options: options);
    _interpreter!.allocateTensors();

    final inputShape = _interpreter!.getInputTensor(0).shape;
    if (inputShape.length >= 4) {
      _inputHeight = inputShape[1];
      _inputWidth = inputShape[2];
      _inputChannels = inputShape[3];
    }

    final outputShape = _interpreter!.getOutputTensor(0).shape;
    _outputLength = outputShape.fold<int>(1, (value, element) => value * element);
    _isInitialized = true;
  }

  bool get isReady => _isInitialized && _interpreter != null;
  int get outputLength => _outputLength;

  Future<List<double>> extractFeatures(Uint8List faceImageBytes) async {
    if (!isReady) {
      await initialize();
    }
    if (_interpreter == null) {
      throw Exception('Emotion feature extractor is not initialized');
    }

    final decoded = img.decodeImage(faceImageBytes);
    if (decoded == null) {
      throw Exception('Failed to decode face image for EfficientNet extraction');
    }

    final resized = img.copyResize(
      decoded,
      width: _inputWidth,
      height: _inputHeight,
    );

    final inputBuffer = Float32List(_inputWidth * _inputHeight * _inputChannels);
    int index = 0;
    for (int y = 0; y < _inputHeight; y++) {
      for (int x = 0; x < _inputWidth; x++) {
        final pixel = resized.getPixel(x, y);
        inputBuffer[index++] = pixel.r.toDouble();
        inputBuffer[index++] = pixel.g.toDouble();
        inputBuffer[index++] = pixel.b.toDouble();
      }
    }

    final input = inputBuffer.reshape([1, _inputHeight, _inputWidth, _inputChannels]);
    final output = List.generate(1, (_) => List.filled(_outputLength, 0.0));
    _interpreter!.run(input, output);

    // The training CSV used log-probabilities (ln of softmax output).
    // TFLite include_top=True outputs softmax probabilities in [0, 1].
    // Convert to log-probabilities to match the training features.
    return (output.first as List).map((value) {
      final prob = (value as num).toDouble();
      // Clamp to avoid log(0) = -infinity
      final clamped = prob < 1e-7 ? 1e-7 : prob;
      return math.log(clamped);
    }).toList();
  }

  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
    _outputLength = 0;
  }
}