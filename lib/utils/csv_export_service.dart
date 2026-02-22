import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import '../models/attendance_model.dart';
import '../models/student_model.dart';

class CsvExportService {
  static Future<String> generateAndSaveAttendanceReport({
    required String teacherName,
    required String subjectName,
    required DateTime attendanceDate,
    required Map<int, AttendanceStatus> attendanceStatus,
    required List<Student> enrolledStudents,
  }) async {
    try {
      // Build student lookup map (only students with valid IDs)
      final studentMap = <int, Student>{};
      for (final student in enrolledStudents) {
        if (student.id != null) {
          studentMap[student.id!] = student;
        }
      }

      // Separate attended and absent students by processing attendance map
      final attendedStudents = <Student>[];
      final absentStudents = <Student>[];

      for (final studentId in attendanceStatus.keys) {
        final student = studentMap[studentId];
        if (student == null) continue;
        
        final status = attendanceStatus[studentId];
        if (status == AttendanceStatus.present) {
          attendedStudents.add(student);
        } else {
          absentStudents.add(student);
        }
      }

      // Sort by name
      attendedStudents.sort((a, b) => a.name.compareTo(b.name));
      absentStudents.sort((a, b) => a.name.compareTo(b.name));

      // Create CSV content
      final dateStr = '${attendanceDate.year}-${attendanceDate.month.toString().padLeft(2, '0')}-${attendanceDate.day.toString().padLeft(2, '0')}';
      final csvContent = StringBuffer();

      // Header
      csvContent.writeln('Teacher Name,Subject');
      csvContent.writeln('"$teacherName","$subjectName"');
      csvContent.writeln('');
      csvContent.writeln('Date: $dateStr');
      csvContent.writeln('');
      
      // Single-summary cell (quoted) so spreadsheet shows the text in one column
      csvContent.writeln('"Attendees = ${attendedStudents.length}, Absentees = ${absentStudents.length}, Total = ${enrolledStudents.length}"');
      csvContent.writeln('');

      // Names listed side-by-side
      final maxLines = attendedStudents.length > absentStudents.length
          ? attendedStudents.length
          : absentStudents.length;

      for (int i = 0; i < maxLines; i++) {
        final presentName = i < attendedStudents.length ? attendedStudents[i].name : '';
        final absentName = i < absentStudents.length ? absentStudents[i].name : '';
        csvContent.writeln('"$presentName","$absentName",');
      }

      // Get export directory (same as export_screen)
      Directory? exportDir;
      try {
        if (Platform.isAndroid) {
          final storageStatus = await Permission.storage.request();
          if (storageStatus.isGranted) {
            final externals = await getExternalStorageDirectories(
              type: StorageDirectory.downloads,
            );
            if (externals != null && externals.isNotEmpty) {
              exportDir = Directory(externals.first.path);
            }
          }
        }
      } catch (_) {}

      exportDir ??= await getApplicationDocumentsDirectory();
      final dir = Directory('${exportDir.path}/FaceAttendanceExports');
      await dir.create(recursive: true);

      // Create filename
      final timeStamp = DateTime.now().toIso8601String().replaceAll(
        RegExp(r'[:\\.]'),
        '-',
      );
      final filename = '${teacherName}_${subjectName}_$timeStamp.csv'
          .replaceAll(' ', '_');
      
      // Save to FaceAttendanceExports directory
      final file = File('${dir.path}/$filename');
      await file.writeAsString(csvContent.toString(), flush: true);

      // Also save to Downloads folder
      try {
        final downloadsDir = Directory('/storage/emulated/0/Download');
        if (await downloadsDir.exists()) {
          final downloadFile = File('${downloadsDir.path}/$filename');
          await downloadFile.writeAsString(csvContent.toString(), flush: true);
        }
      } catch (e) {
        print('⚠️ Could not save to Downloads: $e');
      }

      return file.path;
    } catch (e) {
      throw Exception('Error generating CSV report: $e');
    }
  }
}
