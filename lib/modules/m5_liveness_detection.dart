import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:ui';
import 'm1_face_detection.dart';

/// M5: Liveness Detection Module
/// Detects if a face is live using blink detection
class LivenessDetectionModule {
  static const String modelName = 'Blink-based Liveness Detection';
  static const double blinkThreshold = 0.3; // EAR threshold for blink detection
  static const int requiredBlinks = 2; // Number of blinks required
  static const Duration blinkTimeout = Duration(
    seconds: 10,
  ); // Time to complete blinks

  /// Detect blink in a sequence of face images
  Future<bool> detectBlink(List<DetectedFace> faceSequence) async {
    if (faceSequence.length < 3) return false;

    final earValues = <double>[];
    for (final face in faceSequence) {
      final ear = _calculateEyeAspectRatio(face);
      if (ear != null) {
        earValues.add(ear);
      }
    }

    if (earValues.length < 3) return false;

    // Detect blink pattern: EAR drops below threshold then recovers
    return _detectBlinkPattern(earValues);
  }

  /// Calculate Eye Aspect Ratio (EAR) for blink detection
  double? _calculateEyeAspectRatio(DetectedFace face) {
    // Extract eye landmarks (assuming MediaPipe landmark order)
    // Left eye: landmarks 33, 160, 158, 133, 153, 144
    // Right eye: landmarks 362, 385, 387, 263, 373, 380

    if (face.landmarks.length < 380) return null; // Not enough landmarks

    final leftEye = _extractEyeLandmarks(face.landmarks, isLeftEye: true);
    final rightEye = _extractEyeLandmarks(face.landmarks, isLeftEye: false);

    if (leftEye == null || rightEye == null) return null;

    final leftEAR = _calculateEAR(leftEye);
    final rightEAR = _calculateEAR(rightEye);

    return (leftEAR + rightEAR) / 2.0; // Average EAR
  }

  /// Extract eye landmarks from face landmarks
  List<Offset>? _extractEyeLandmarks(
    List<Offset> landmarks, {
    required bool isLeftEye,
  }) {
    // MediaPipe face mesh landmark indices for eyes
    final eyeIndices = isLeftEye
        ? [33, 160, 158, 133, 153, 144] // Left eye
        : [362, 385, 387, 263, 373, 380]; // Right eye

    final eyePoints = <Offset>[];
    for (final index in eyeIndices) {
      if (index < landmarks.length) {
        eyePoints.add(landmarks[index]);
      }
    }

    return eyePoints.length == 6 ? eyePoints : null;
  }

  /// Calculate Eye Aspect Ratio for 6 eye landmarks
  double _calculateEAR(List<Offset> eyePoints) {
    // EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    final p1 = eyePoints[0];
    final p2 = eyePoints[1];
    final p3 = eyePoints[2];
    final p4 = eyePoints[3];
    final p5 = eyePoints[4];
    final p6 = eyePoints[5];

    final vertical1 = _euclideanDistance(p2, p6);
    final vertical2 = _euclideanDistance(p3, p5);
    final horizontal = _euclideanDistance(p1, p4);

    return horizontal > 0 ? (vertical1 + vertical2) / (2.0 * horizontal) : 0.0;
  }

  /// Calculate Euclidean distance between two points
  double _euclideanDistance(Offset p1, Offset p2) {
    final dx = p1.dx - p2.dx;
    final dy = p1.dy - p2.dy;
    return math.sqrt(dx * dx + dy * dy);
  }

  /// Detect blink pattern in EAR values
  bool _detectBlinkPattern(List<double> earValues) {
    // Find local minima (blink points)
    final blinkPoints = <int>[];

    for (var i = 1; i < earValues.length - 1; i++) {
      if (earValues[i] < blinkThreshold &&
          earValues[i] < earValues[i - 1] &&
          earValues[i] < earValues[i + 1]) {
        blinkPoints.add(i);
      }
    }

    // Check if we have enough distinct blinks
    return blinkPoints.length >= requiredBlinks;
  }

  /// Perform complete liveness check with blink detection
  Future<LivenessResult> checkLiveness(List<Uint8List> imageSequence) async {
    final faceDetector = FaceDetectionModule();
    await faceDetector.initialize();

    final faceSequence = <DetectedFace>[];

    try {
      for (final imageBytes in imageSequence) {
        final faces = await faceDetector.detectFaces(imageBytes);
        if (faces.isNotEmpty) {
          faceSequence.add(faces.first); // Use first detected face
        }
      }

      final isLive = await detectBlink(faceSequence);

      return LivenessResult(
        isLive: isLive,
        confidence: isLive ? 0.95 : 0.1,
        method: 'blink_detection',
        details: {
          'frames_processed': imageSequence.length,
          'faces_detected': faceSequence.length,
          'blink_detected': isLive,
        },
      );
    } finally {
      faceDetector.dispose();
    }
  }
}

/// Liveness detection result
class LivenessResult {
  final bool isLive;
  final double confidence;
  final String method;
  final Map<String, dynamic> details;

  LivenessResult({
    required this.isLive,
    required this.confidence,
    required this.method,
    required this.details,
  });

  @override
  String toString() =>
      'LivenessResult(isLive: $isLive, confidence: ${confidence.toStringAsFixed(2)}, method: $method)';
}
