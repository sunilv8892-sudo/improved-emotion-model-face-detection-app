import 'dart:ui';

/// Face detection result from M1 module
class DetectedFace {
  final int? id;
  final double x;
  final double y;
  final double width;
  final double height;
  final double confidence;
  final String expression;

  DetectedFace({
    this.id,
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.confidence,
    this.expression = '',
  });

  /// Get bounding box as [x, y, width, height]
  List<double> get boundingBox => [x, y, width, height];

  /// Calculate center point of the face
  Offset get center => Offset(x + width / 2, y + height / 2);

  /// Validate if face detection is valid
  bool get isValid =>
      x >= 0 &&
      y >= 0 &&
      width > 0 &&
      height > 0 &&
      confidence > 0 &&
      confidence <= 1.0;

  @override
  String toString() =>
      'DetectedFace(x: $x, y: $y, w: $width, h: $height, conf: $confidence)';
}
