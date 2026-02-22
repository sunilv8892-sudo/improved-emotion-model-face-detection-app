import 'dart:typed_data';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// M2: Face Embedding Module using FaceNet (128D)
/// Converts cropped face images into numerical vectors (embeddings)
/// Using a 128-dimensional FaceNet TFLite model
class FaceEmbeddingModule {
  static const String modelName = 'FaceNet-128';
  static const int embeddingDimension = 128; // FaceNet outputs 128D vectors
  static const String modelAssetPath = 'assets/models/embedding_model.tflite';

  Interpreter? _interpreter;
  bool _isInitialized = false;
  int _inputWidth = 160;
  int _inputHeight = 160;

  /// Initialize the FaceNet model
  Future<void> initialize() async {
    if (_isInitialized) return;

    try {
      final options = InterpreterOptions()..threads = 4;
      _interpreter = await Interpreter.fromAsset(
        modelAssetPath,
        options: options,
      );
      _interpreter?.allocateTensors();

      final inputShape = _interpreter!.getInputTensor(0).shape;
      if (inputShape.length >= 4) {
        _inputHeight = inputShape[1];
        _inputWidth = inputShape[2];
      }

      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final outputLength = outputShape.fold<int>(1, (a, b) => a * b);
      if (outputLength != embeddingDimension) {
        throw Exception(
          'Invalid embedding model output dimension: expected $embeddingDimension, got $outputLength',
        );
      }
      _isInitialized = true;
    } catch (e) {
      _interpreter?.close();
      _interpreter = null;
      _isInitialized = false;
      throw Exception('FaceNet init failed: $e');
    }
  }

  /// Generate face embedding from cropped face image
  /// Input: Cropped face image (uint8 bytes)
  /// Output: Embedding vector (list of doubles)
  Future<List<double>?> generateEmbedding(Uint8List faceImageBytes) async {
    if (!_isInitialized) await initialize();
    if (_interpreter == null) return null;

    try {
      // Decode and preprocess image
      final image = img.decodeImage(faceImageBytes);
      if (image == null) return null;

      // Resize to model input size from tensor metadata.
      final resized = img.copyResize(image, width: _inputWidth, height: _inputHeight);

      // Convert to RGB float array
      final inputBuffer = Float32List(_inputWidth * _inputHeight * 3);
      var index = 0;
      for (var y = 0; y < _inputHeight; y++) {
        for (var x = 0; x < _inputWidth; x++) {
          final pixel = resized.getPixel(x, y);
          inputBuffer[index++] = pixel.r / 255.0; // R
          inputBuffer[index++] = pixel.g / 255.0; // G
          inputBuffer[index++] = pixel.b / 255.0; // B
        }
      }

      // Prepare input tensor and output buffer
      final inputData = inputBuffer.reshape([1, _inputHeight, _inputWidth, 3]);
      final outputTensor = _interpreter!.getOutputTensor(0);
      final outputShape = outputTensor.shape;
      final outputLength = outputShape.fold<int>(1, (a, b) => a * b);
      final outputBuffer = List.generate(1, (_) => List.filled(outputLength, 0.0));

      // Run inference
      _interpreter!.run(inputData, outputBuffer);

      List<double> embedding = (outputBuffer.first as List)
          .map((v) => (v as num).toDouble())
          .toList();

      if (embedding.length != embeddingDimension) {
        throw Exception(
          'Embedding dimension mismatch: expected $embeddingDimension, got ${embedding.length}',
        );
      }

      return normalizeEmbedding(embedding);
    } catch (e) {
      print('Error generating embedding: $e');
      return null;
    }
  }

  /// Normalize embedding vector (L2 normalization)
  /// Keeping normalization is fine for downstream KNN/EUCLIDEAN use.
  List<double> normalizeEmbedding(List<double> embedding) {
    if (embedding.isEmpty) return embedding;
    final norm = _calculateNorm(embedding);
    if (norm == 0 || norm.isNaN) return embedding;
    return embedding.map((e) => e / norm).toList();
  }

  /// Calculate L2 norm of vector
  double _calculateNorm(List<double> vector) {
    double sum = 0;
    for (final v in vector) {
      sum += v * v;
    }
    return sum > 0 ? _sqrt(sum) : 0;
  }

  /// Simple square root approximation
  double _sqrt(double x) {
    if (x == 0) return 0;
    double z = x;
    double result = 0;
    while ((z - result).abs() > 1e-7) {
      result = z;
      z = 0.5 * (z + x / z);
    }
    return z;
  }

  /// Validate embedding
  bool isValidEmbedding(List<double> embedding) {
    return embedding.length == embeddingDimension && embedding.every((e) => e.isFinite);
  }

  /// Get embedding statistics
  Map<String, dynamic> getEmbeddingStats(List<double> embedding) {
    if (embedding.isEmpty) {
      return {'min': 0, 'max': 0, 'mean': 0, 'dimension': 0};
    }

    double min = embedding[0];
    double max = embedding[0];
    double sum = 0;

    for (final val in embedding) {
      if (val < min) min = val;
      if (val > max) max = val;
      sum += val;
    }

    return {
      'dimension': embedding.length,
      'min': min.toStringAsFixed(4),
      'max': max.toStringAsFixed(4),
      'mean': (sum / embedding.length).toStringAsFixed(4),
    };
  }

  /// Dispose resources
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isInitialized = false;
  }

  /// Indicates whether the interpreter is ready for inference
  bool get isReady => _isInitialized && _interpreter != null;
}
