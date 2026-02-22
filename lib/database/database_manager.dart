import 'dart:convert';
import 'dart:math';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/student_model.dart' as model;
import '../models/embedding_model.dart' as model;
import '../models/attendance_model.dart' as model;
import '../models/subject_model.dart' as model;

/// SharedPreferences-backed database manager (safe, no codegen)
class DatabaseManager {
  static final DatabaseManager _instance = DatabaseManager._internal();

  factory DatabaseManager() {
    return _instance;
  }

  DatabaseManager._internal();

  int? _parseInt(dynamic value) {
    if (value is int) return value;
    if (value is String) return int.tryParse(value);
    return null;
  }

  DateTime _parseDate(String? s) =>
      DateTime.tryParse(s ?? '') ?? DateTime.now();

  // ==================== STUDENT OPERATIONS ====================

  Future<int> insertStudent(model.Student student) async {
    final prefs = await SharedPreferences.getInstance();
    final List<String> students = prefs.getStringList('students') ?? [];

    // Determine new id
    final ids = students
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .map((m) => _parseInt(m['id']))
        .whereType<int>()
        .toList();
    final newId = ids.isEmpty ? 1 : (ids.reduce((a, b) => a > b ? a : b) + 1);

    final record = student.toMap();
    record['id'] = newId; // Override id with the generated one
    students.add(jsonEncode(record));
    await prefs.setStringList('students', students);
    return newId;
  }

  Future<List<model.Student>> getAllStudents() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('students') ?? [];
    return jsonList.map((s) {
      final data = jsonDecode(s) as Map<String, dynamic>;
      return model.Student(
        id: _parseInt(data['id']),
        name: data['name'] as String? ?? '',
        rollNumber: data['roll_number'] as String? ?? '',
        className: data['class'] as String? ?? '',
        gender: data['gender'] as String? ?? '',
        age: data['age'] as int? ?? 0,
        phoneNumber: data['phone_number'] as String? ?? '',
        enrollmentDate: _parseDate(data['enrollment_date'] as String?),
      );
    }).toList();
  }

  Future<model.Student?> getStudentById(int id) async {
    final all = await getAllStudents();
    try {
      return all.firstWhere((s) => s.id == id);
    } catch (e) {
      return null;
    }
  }

  Map<String, dynamic> _createStudentMap(model.Student student, int id) {
    final map = Map<String, dynamic>.from(student.toMap());
    map['id'] = id;
    return map;
  }

  Future<int> updateStudent(int id, model.Student student) async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('students') ?? [];
    final updated = jsonList
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .map((m) => m['id'] == id ? _createStudentMap(student, id) : m)
        .map(jsonEncode)
        .toList();
    await prefs.setStringList('students', updated);
    return 1;
  }

  Future<int> deleteStudent(int id) async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('students') ?? [];
    final filtered = jsonList
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .where((m) => _parseInt(m['id']) != id)
        .map(jsonEncode)
        .toList();
    await prefs.setStringList('students', filtered);
    return 1;
  }

  // ==================== EMBEDDING OPERATIONS ====================

  Future<int> insertEmbedding(model.FaceEmbedding embedding) async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('embeddings') ?? [];
    final ids = jsonList
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .map((m) => _parseInt(m['id']))
        .whereType<int>()
        .toList();
    final newId = ids.isEmpty ? 1 : (ids.reduce((a, b) => a > b ? a : b) + 1);
    final record = {
      'id': newId,
      'studentId': embedding.studentId,
      'vector': embedding.vector,
      'captureDate': embedding.captureDate.toIso8601String(),
    };
    jsonList.add(jsonEncode(record));
    await prefs.setStringList('embeddings', jsonList);
    return newId;
  }

  Future<List<model.FaceEmbedding>> getEmbeddingsForStudent(
    int studentId,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('embeddings') ?? [];
    return jsonList
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .where((m) => _parseInt(m['studentId']) == studentId)
        .map(
          (m) => model.FaceEmbedding(
            id: _parseInt(m['id']),
            studentId: _parseInt(m['studentId'])!,
            vector: List<double>.from(
              (m['vector'] as List).map((e) => (e as num).toDouble()),
            ),
            captureDate: _parseDate(m['captureDate'] as String?),
          ),
        )
        .toList();
  }

  Future<List<model.FaceEmbedding>> getAllEmbeddings() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('embeddings') ?? [];
    return jsonList
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .map(
          (m) => model.FaceEmbedding(
            id: _parseInt(m['id']),
            studentId: _parseInt(m['studentId'])!,
            vector: List<double>.from(
              (m['vector'] as List).map((e) => (e as num).toDouble()),
            ),
            captureDate: _parseDate(m['captureDate'] as String?),
          ),
        )
        .toList();
  }

  /// Euclidean distance between two vectors
  double _euclideanDistance(List<double> a, List<double> b) {
    if (a.isEmpty || b.isEmpty || a.length != b.length) return double.infinity;
    final dim = a.length;
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sqrt(sum);
  }

  /// Convert distance to similarity-like score in (0,1]: sim = 1/(1+distance)
  double _distanceToSimilarity(double distance) {
    if (!distance.isFinite) return 0.0;
    return 1.0 / (1.0 + distance);
  }

  Future<List<model.FaceEmbedding>> findSimilarEmbeddings(
    List<double> queryVector,
    double threshold,
  ) async {
    final all = await getAllEmbeddings();
    return all.where((e) {
      final dist = _euclideanDistance(e.vector, queryVector);
      final sim = _distanceToSimilarity(dist);
      return sim >= threshold;
    }).toList();
  }

  // ==================== ATTENDANCE OPERATIONS ====================

  Future<int> insertAttendance(model.AttendanceRecord attendance) async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('attendance') ?? [];
    final ids = jsonList
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .map((m) => _parseInt(m['id']))
        .whereType<int>()
        .toList();
    final newId = ids.isEmpty ? 1 : (ids.reduce((a, b) => a > b ? a : b) + 1);
    final record = {
      'id': newId,
      'studentId': attendance.studentId,
      'date': attendance.date.toIso8601String(),
      'time': attendance.time,
      'status': attendance.status.name,
      'recordedAt': attendance.recordedAt.toIso8601String(),
    };
    jsonList.add(jsonEncode(record));
    await prefs.setStringList('attendance', jsonList);
    return newId;
  }

  Future<List<model.AttendanceRecord>> getAttendanceForStudent(
    int studentId,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    final records = prefs.getStringList('attendance') ?? [];
    return records
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .where((m) => _parseInt(m['studentId']) == studentId)
        .map(
          (m) => model.AttendanceRecord(
            id: _parseInt(m['id']),
            studentId: _parseInt(m['studentId'])!,
            date: _parseDate(m['date'] as String?),
            time: m['time'] as String?,
            status: model.AttendanceStatus.values.firstWhere(
              (e) => e.name == (m['status'] as String? ?? ''),
              orElse: () => model.AttendanceStatus.absent,
            ),
            recordedAt: _parseDate(m['recordedAt'] as String?),
          ),
        )
        .toList();
  }

  Future<List<model.AttendanceRecord>> getAttendanceForDate(
    DateTime date,
  ) async {
    final prefs = await SharedPreferences.getInstance();
    final records = prefs.getStringList('attendance') ?? [];
    final dateStr =
        '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
    final allForDate = records
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .where((m) {
          final d = _parseDate(m['date'] as String?);
          final recordDateStr =
              '${d.year}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';
          return recordDateStr == dateStr;
        })
        .map(
          (m) => model.AttendanceRecord(
            id: _parseInt(m['id']),
            studentId: _parseInt(m['studentId'])!,
            date: _parseDate(m['date'] as String?),
            time: m['time'] as String?,
            status: model.AttendanceStatus.values.firstWhere(
              (e) => e.name == (m['status'] as String? ?? ''),
              orElse: () => model.AttendanceStatus.absent,
            ),
            recordedAt: _parseDate(m['recordedAt'] as String?),
          ),
        )
        .toList();

    // Deduplicate: keep latest record per student
    final byStudent = <int, model.AttendanceRecord>{};
    for (final r in allForDate) {
      final existing = byStudent[r.studentId];
      if (existing == null || r.recordedAt.isAfter(existing.recordedAt)) {
        byStudent[r.studentId] = r;
      }
    }
    return byStudent.values.toList();
  }

  Future<model.AttendanceRecord?> getAttendanceForStudentOnDate(
    int studentId,
    DateTime date,
  ) async {
    final records = await getAttendanceForDate(date);
    try {
      return records.firstWhere((r) => r.studentId == studentId);
    } catch (e) {
      return null;
    }
  }

  // Backwards compatibility helpers for code that expects a Drift-style DB
  Future<dynamic> get database async => this;

  Future<void> close() async {}

  Future<int> recordAttendance(model.AttendanceRecord attendance) async {
    return await insertAttendance(attendance);
  }

  Future<Map<String, dynamic>> getAttendanceStats(int studentId) async {
    final records = await getAttendanceForStudent(studentId);

    // Deduplicate: keep only the latest record per date
    final byDate = <String, model.AttendanceRecord>{};
    for (final r in records) {
      final key = '${r.date.year}-${r.date.month}-${r.date.day}';
      final existing = byDate[key];
      if (existing == null || r.recordedAt.isAfter(existing.recordedAt)) {
        byDate[key] = r;
      }
    }
    final unique = byDate.values.toList();

    final total = unique.length;
    final present = unique
        .where((r) => r.status == model.AttendanceStatus.present)
        .length;
    final absent = unique
        .where((r) => r.status == model.AttendanceStatus.absent)
        .length;
    final late = unique
        .where((r) => r.status == model.AttendanceStatus.late)
        .length;
    return {
      'total': total,
      'present': present,
      'absent': absent,
      'late': late,
      'attendance_rate': total > 0 ? present / total : 0.0,
    };
  }

  Future<List<model.AttendanceRecord>> getAllAttendance() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('attendance') ?? [];
    return jsonList
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .map(
          (m) => model.AttendanceRecord(
            id: _parseInt(m['id']),
            studentId: _parseInt(m['studentId'])!,
            date: _parseDate(m['date'] as String?),
            time: m['time'] as String?,
            status: model.AttendanceStatus.values.firstWhere(
              (e) => e.name == (m['status'] as String? ?? ''),
              orElse: () => model.AttendanceStatus.absent,
            ),
            recordedAt: _parseDate(m['recordedAt'] as String?),
          ),
        )
        .toList();
  }

  Future<int> deleteEmbeddingsForStudent(int studentId) async {
    final prefs = await SharedPreferences.getInstance();
    final list = prefs.getStringList('embeddings') ?? [];
    final filtered = list
        .map((s) => jsonDecode(s) as Map<String, dynamic>)
        .where((m) => _parseInt(m['studentId']) != studentId)
        .map(jsonEncode)
        .toList();
    await prefs.setStringList('embeddings', filtered);
    return 1;
  }

  Future<List<model.Student>> getEnrolledStudents() async {
    final allEmbeddings = await getAllEmbeddings();
    final ids = <int>{};
    for (final e in allEmbeddings) {
      ids.add(e.studentId);
    }
    final students = <model.Student>[];
    for (final id in ids) {
      final s = await getStudentById(id);
      if (s != null) students.add(s);
    }
    return students;
  }

  // ==================== SUBJECTS & TEACHER SESSIONS ====================

  Future<void> insertSubject(model.Subject subject) async {
    final prefs = await SharedPreferences.getInstance();
    final subjects = prefs.getStringList('subjects') ?? [];
    subjects.add(
      jsonEncode({
        'id': subject.id,
        'name': subject.name,
        'createdAt': subject.createdAt.toIso8601String(),
      }),
    );
    await prefs.setStringList('subjects', subjects);
  }

  Future<List<model.Subject>> getAllSubjects() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('subjects') ?? [];

    return jsonList
        .map((json) {
          final data = jsonDecode(json) as Map<String, dynamic>;
          return model.Subject(
            id: _parseInt(data['id']),
            name: data['name'] as String? ?? '',
            createdAt: _parseDate(data['createdAt'] as String?),
          );
        })
        .whereType<model.Subject>()
        .toList();
  }

  Future<model.Subject> getOrCreateSubject(String subjectName) async {
    final subjects = await getAllSubjects();

    try {
      return subjects.firstWhere(
        (s) => s.name.toLowerCase() == subjectName.toLowerCase(),
      );
    } catch (e) {
      final newSubject = model.Subject(
        id: DateTime.now().millisecondsSinceEpoch,
        name: subjectName,
      );
      await insertSubject(newSubject);
      return newSubject;
    }
  }

  Future<void> insertTeacherSession(model.TeacherSession session) async {
    final prefs = await SharedPreferences.getInstance();
    final sessions = prefs.getStringList('teacherSessions') ?? [];
    sessions.add(
      jsonEncode({
        'id': session.id,
        'teacherName': session.teacherName,
        'subjectId': session.subjectId,
        'subjectName': session.subjectName,
        'date': session.date.toIso8601String(),
        'createdAt': session.createdAt.toIso8601String(),
      }),
    );
    await prefs.setStringList('teacherSessions', sessions);
  }

  Future<List<model.TeacherSession>> getAllTeacherSessions() async {
    final prefs = await SharedPreferences.getInstance();
    final jsonList = prefs.getStringList('teacherSessions') ?? [];

    return jsonList
        .map((json) {
          final data = jsonDecode(json) as Map<String, dynamic>;
          final id = _parseInt(data['id']);
          final subjectId = _parseInt(data['subjectId']);
          if (id == null || subjectId == null) return null;
          return model.TeacherSession(
            id: id,
            teacherName: data['teacherName'] as String? ?? '',
            subjectId: subjectId,
            subjectName: data['subjectName'] as String? ?? '',
            date:
                DateTime.tryParse(data['date'] as String? ?? '') ??
                DateTime.now(),
            createdAt:
                DateTime.tryParse(data['createdAt'] as String? ?? '') ??
                DateTime.now(),
          );
        })
        .whereType<model.TeacherSession>()
        .toList();
  }

  Future<List<model.TeacherSession>> getTeacherSessionsByDate(
    DateTime date,
  ) async {
    final sessions = await getAllTeacherSessions();
    final dateStr =
        '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
    return sessions.where((s) {
      final sessionDateStr =
          '${s.date.year}-${s.date.month.toString().padLeft(2, '0')}-${s.date.day.toString().padLeft(2, '0')}';
      return sessionDateStr == dateStr;
    }).toList();
  }

  Future<List<model.TeacherSession>> getTeacherSessionsBySubject(
    int subjectId,
  ) async {
    final sessions = await getAllTeacherSessions();
    return sessions.where((s) => s.subjectId == subjectId).toList();
  }
}
