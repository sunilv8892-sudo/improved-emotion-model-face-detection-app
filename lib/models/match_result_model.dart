/// Result from face matching operation (M3)
class MatchResult {
  final String identityType; // "known" or "unknown"
  final int? studentId;
  final String? studentName;
  final double similarity;
  final DateTime timestamp;

  MatchResult({
    required this.identityType,
    this.studentId,
    this.studentName,
    required this.similarity,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// Is this a recognized known person?
  bool get isKnown => identityType == 'known' && studentId != null;

  /// Is this an unknown person?
  bool get isUnknown => identityType == 'unknown';

  @override
  String toString() =>
      'MatchResult(type: $identityType, student: $studentName, similarity: $similarity)';
}
