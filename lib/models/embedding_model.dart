/// Face embedding model storing vector representation of a face
class FaceEmbedding {
  final int? id;
  final int studentId;
  final List<double> vector;
  final DateTime captureDate;

  FaceEmbedding({
    this.id,
    required this.studentId,
    required this.vector,
    DateTime? captureDate,
  }) : captureDate = captureDate ?? DateTime.now();

  /// Vector dimension (typically 128 or 192)
  int get dimension => vector.length;

  /// Convert to JSON for database storage
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'student_id': studentId,
      'vector': vector.join(','),
      'capture_date': captureDate.toIso8601String(),
    };
  }

  /// Create from database map
  factory FaceEmbedding.fromMap(Map<String, dynamic> map) {
    final vectorStr = map['vector'] as String;
    final vector = vectorStr.split(',').map((e) => double.parse(e)).toList();
    return FaceEmbedding(
      id: map['id'] as int?,
      studentId: map['student_id'] as int,
      vector: vector,
      captureDate: DateTime.parse(map['capture_date'] as String),
    );
  }

  @override
  String toString() =>
      'FaceEmbedding(id: $id, studentId: $studentId, dim: ${vector.length})';
}
