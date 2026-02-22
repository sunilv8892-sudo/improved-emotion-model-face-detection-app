# Database Migration: Drift → SharedPreferences

## Summary
Successfully migrated from Drift ORM to SharedPreferences for data persistence to resolve code generation failures and simplify the architecture.

## Problem Statement
The original implementation used Drift ORM for SQLite database management. When attempting to add new tables (`Subjects` and `TeacherSessions` for the teacher+subject attendance feature), the schema migration (v1→v2) failed to generate required Dart code:
- Build runner compiled but didn't generate `FaceEmbeddingsCompanion`, `SubjectsCompanion`, `TeacherSessionsCompanion` classes
- 40+ compilation errors from missing generated types
- Complex code generation setup became blocker for feature development

## Solution
Migrated all data persistence to SharedPreferences with JSON serialization:
- **Simpler**: No code generation required
- **Effective**: SharedPreferences is ideal for this app's data volume and usage patterns
- **Testable**: Easy to implement custom CRUD operations in Dart without generated code
- **Portable**: JSON serialization makes data human-readable and backup-friendly

## Changed Files

### Core Database Layer
**`lib/database/database_manager.dart`** - Complete rewrite
- Removed dependency on Drift's `FaceRecognitionDatabase`
- Implemented SharedPreferences-based storage for all entities:
  - **Students**: List stored as `'students'` (with manual ID management)
  - **FaceEmbeddings**: List stored as `'embeddings'` (with 512-D vector serialization)
  - **Attendance**: List stored as `'attendance'` (with date/status tracking)
  - **Subjects**: List stored as `'subjects'` (NEW - for subject-based attendance)
  - **TeacherSessions**: List stored as `'teacherSessions'` (NEW - for teacher context)

Key methods preserved:
```dart
// Students
insertStudent(Student) → Future<int>
getAllStudents() → Future<List<Student>>
getStudentById(int) → Future<Student?>
updateStudent(int, Student) → Future<int>
deleteStudent(int) → Future<int>

// Embeddings
insertEmbedding(FaceEmbedding) → Future<int>
getEmbeddingsForStudent(int) → Future<List<FaceEmbedding>>
getAllEmbeddings() → Future<List<FaceEmbedding>>
findSimilarEmbeddings(List<double>, double) → Future<List<FaceEmbedding>>

// Attendance
insertAttendance(AttendanceRecord) → Future<int>
getAttendanceForStudent(int) → Future<List<AttendanceRecord>>
getAttendanceForDate(DateTime) → Future<List<AttendanceRecord>>
getAttendanceForStudentOnDate(int, DateTime) → Future<AttendanceRecord?>
getAllAttendance() → Future<List<AttendanceRecord>>

// Subjects (NEW)
insertSubject(Subject) → Future<void>
getAllSubjects() → Future<List<Subject>>
getOrCreateSubject(String) → Future<Subject>

// Teacher Sessions (NEW)
insertTeacherSession(TeacherSession) → Future<void>
getAllTeacherSessions() → Future<List<TeacherSession>>
getTeacherSessionsByDate(DateTime) → Future<List<TeacherSession>>
getTeacherSessionsBySubject(int) → Future<List<TeacherSession>>
```

**`lib/database/face_recognition_database.dart`** - Deprecated
- Converted to documentation file explaining migration
- Original Drift schema removed to prevent compilation errors
- Kept as reference for future schema documentation

### Updated Screens

**`lib/screens/settings_screen.dart`**
- Replaced Drift delete operations with SharedPreferences `remove()` calls
- Now clears: `'students'`, `'embeddings'`, `'attendance'`, `'subjects'`, `'teacherSessions'`
- Removed unused `DatabaseManager` import

**`lib/screens/teacher_setup_screen.dart`**
- Updated `_createNewSubject()` to use `getOrCreateSubject()` instead of `insertSubject()`
- Handles ID generation with auto-increment logic using current timestamp
- Replaced deprecated `WillPopScope` with `PopScope` (Flutter 3.12+)

### Modules

**`lib/modules/m4_attendance_management.dart`**
- Fixed `firstWhereOrNull()` → use `firstWhere()` with try-catch (Dart compatibility)
- Removed unused `subject_model` import

### Utility Files

**`lib/screens/export_screen.dart`**
- Removed unnecessary `dart:ui` import
- Already had `Subject` type support for subject-based CSV exports

## Data Structure Examples

### Student Storage
```json
{
  "id": 1,
  "name": "John Doe",
  "rollNumber": "2024001",
  "className": "10-A",
  "enrollmentDate": "2024-01-15T10:30:00.000Z"
}
```

### Face Embedding Storage
```json
{
  "id": 1,
  "studentId": 1,
  "vector": [0.123, -0.456, 0.789, ...],  // 512-element double array
  "captureDate": "2024-01-20T14:22:00.000Z"
}
```

### Subject Storage (NEW)
```json
{
  "id": 1705761600000,
  "name": "Mathematics",
  "createdAt": "2024-01-20T09:00:00.000Z"
}
```

### Teacher Session Storage (NEW)
```json
{
  "id": 1705761900000,
  "teacherName": "Mrs. Smith",
  "subjectId": 1,
  "subjectName": "Mathematics",
  "date": "2024-01-20T09:00:00.000Z",
  "createdAt": "2024-01-20T09:00:00.000Z"
}
```

## Implementation Details

### ID Generation
- Students, Embeddings, Attendance: Auto-increment based on max ID in list + 1
- Subjects, TeacherSessions: Use `DateTime.now().millisecondsSinceEpoch` for guaranteed uniqueness

### Vector Similarity
- Implemented cosine similarity using `dart:math.sqrt()`
- Matches original Drift formula: `dotProduct / (normA * normB)`
- Used by `findSimilarEmbeddings()` for face recognition matching

### Date Comparison
- `getAttendanceForDate()` and `getTeacherSessionsByDate()` use string comparison
- Format: `'YYYY-MM-DD'` to handle timezone-independent comparisons
- Extracts date portion ignoring time component

## Testing Recommendations

1. **Data Persistence Test**
   - Insert student → Retrieve by ID → Verify data intact

2. **Embedding Search Test**
   - Add embeddings for different students
   - Test `findSimilarEmbeddings()` with known vectors
   - Verify cosine similarity threshold correctly filters results

3. **Attendance Export Test**
   - Record attendance for multiple students
   - Export CSV for specific dates
   - Verify subject-based grouping works (M4 module)

4. **Subject Management Test**
   - Create new subject
   - Confirm `getOrCreateSubject()` prevents duplicates
   - Verify subject-to-session relationships maintained

## Migration Path for Future Database Upgrades

If upgrading to SQLite backend in future:
1. Keep `DatabaseManager` interface unchanged - all CRUD methods are identical
2. Create new `SqliteDatabase` implementation with same method signatures
3. Swap implementation by changing imports in dependent files
4. No changes needed to screens or modules due to abstraction

## Performance Considerations

- **Pros**: Instant writes, no async I/O blocking (SharedPreferences handles internally)
- **Cons**: All data loaded into memory for filtering/searching
- **Suitable for**: Current app scope (students in single class/section)
- **Future**: Switch to SQLite when handling 1000+ students with complex queries

## Compilation Status
✅ **All errors resolved** - `flutter analyze` reports 0 issues
✅ **Ready for testing** - App compiles and can be deployed
✅ **New features enabled** - Subject tracking and teacher session recording now possible
