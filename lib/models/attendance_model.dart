/// Attendance record for a student on a specific date
class AttendanceRecord {
  final int? id;
  final int studentId;
  final DateTime date;
  final String? time;
  final AttendanceStatus status;
  final DateTime recordedAt;

  AttendanceRecord({
    this.id,
    required this.studentId,
    required this.date,
    this.time,
    required this.status,
    DateTime? recordedAt,
  }) : recordedAt = recordedAt ?? DateTime.now();

  /// Convert to JSON for database storage
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'student_id': studentId,
      'date': date.toIso8601String().split('T')[0], // YYYY-MM-DD
      'time':
          time ??
          DateTime.now()
              .toIso8601String()
              .split('T')[1]
              .split('.')[0], // HH:MM:SS
      'status': status.name,
      'recorded_at': recordedAt.toIso8601String(),
    };
  }

  /// Create from database map
  factory AttendanceRecord.fromMap(Map<String, dynamic> map) {
    return AttendanceRecord(
      id: map['id'] as int?,
      studentId: map['student_id'] as int,
      date: DateTime.parse(map['date'] as String),
      time: map['time'] as String?,
      status: AttendanceStatus.values.firstWhere(
        (e) => e.name == map['status'],
        orElse: () => AttendanceStatus.present,
      ),
      recordedAt: DateTime.parse(map['recorded_at'] as String),
    );
  }

  @override
  String toString() =>
      'AttendanceRecord(studentId: $studentId, date: $date, status: ${status.name})';
}

/// Attendance status enumeration
enum AttendanceStatus {
  present('Present'),
  absent('Absent'),
  late('Late');

  final String displayName;
  const AttendanceStatus(this.displayName);
}
