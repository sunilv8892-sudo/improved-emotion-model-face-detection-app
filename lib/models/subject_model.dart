/// Subject model for storing teacher subjects
class Subject {
  final int? id;
  final String name;
  final DateTime createdAt;

  Subject({
    this.id,
    required this.name,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'name': name,
      'created_at': createdAt.toIso8601String(),
    };
  }

  factory Subject.fromMap(Map<String, dynamic> map) {
    return Subject(
      id: map['id'] as int?,
      name: map['name'] as String,
      createdAt: DateTime.parse(map['created_at'] as String),
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Subject && runtimeType == other.runtimeType && id == other.id;

  @override
  int get hashCode => id.hashCode;

  @override
  String toString() => 'Subject(id: $id, name: $name)';
}

/// Teacher session for a specific subject and date
class TeacherSession {
  final int? id;
  final String teacherName;
  final int subjectId;
  final String subjectName;
  final DateTime date;
  final DateTime createdAt;

  TeacherSession({
    this.id,
    required this.teacherName,
    required this.subjectId,
    required this.subjectName,
    required this.date,
    DateTime? createdAt,
  }) : createdAt = createdAt ?? DateTime.now();

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'teacher_name': teacherName,
      'subject_id': subjectId,
      'subject_name': subjectName,
      'date': date.toIso8601String().split('T')[0],
      'created_at': createdAt.toIso8601String(),
    };
  }

  factory TeacherSession.fromMap(Map<String, dynamic> map) {
    return TeacherSession(
      id: map['id'] as int?,
      teacherName: map['teacher_name'] as String,
      subjectId: map['subject_id'] as int,
      subjectName: map['subject_name'] as String,
      date: DateTime.parse(map['date'] as String),
      createdAt: DateTime.parse(map['created_at'] as String),
    );
  }

  @override
  String toString() =>
      'TeacherSession(id: $id, teacher: $teacherName, subject: $subjectName, date: $date)';
}
