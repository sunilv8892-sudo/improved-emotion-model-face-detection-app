import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:share_plus/share_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:file_picker/file_picker.dart';
import '../utils/constants.dart';
import '../utils/export_utils.dart';
import '../widgets/animated_background.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  // Real stats
  int _totalStudents = 0;
  int _totalEmbeddings = 0;
  int _totalAttendance = 0;
  int _totalSubjects = 0;
  int _totalSessions = 0;
  String _dataSize = '...';
  bool _ttsEnabled = true;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();

    // Load TTS preference
    _ttsEnabled = prefs.getBool('tts_enabled') ?? true;

    // Load real database stats
    final students = prefs.getStringList('students') ?? [];
    final embeddings = prefs.getStringList('embeddings') ?? [];
    final attendance = prefs.getStringList('attendance') ?? [];
    final subjects = prefs.getStringList('subjects') ?? [];
    final sessions = prefs.getStringList('teacherSessions') ?? [];

    // Calculate approximate data size
    int totalBytes = 0;
    for (final s in students) {
      totalBytes += s.length;
    }
    for (final s in embeddings) {
      totalBytes += s.length;
    }
    for (final s in attendance) {
      totalBytes += s.length;
    }
    for (final s in subjects) {
      totalBytes += s.length;
    }
    for (final s in sessions) {
      totalBytes += s.length;
    }

    String sizeStr;
    if (totalBytes < 1024) {
      sizeStr = '$totalBytes B';
    } else if (totalBytes < 1024 * 1024) {
      sizeStr = '${(totalBytes / 1024).toStringAsFixed(1)} KB';
    } else {
      sizeStr = '${(totalBytes / (1024 * 1024)).toStringAsFixed(2)} MB';
    }

    if (mounted) {
      setState(() {
        _totalStudents = students.length;
        _totalEmbeddings = embeddings.length;
        _totalAttendance = attendance.length;
        _totalSubjects = subjects.length;
        _totalSessions = sessions.length;
        _dataSize = sizeStr;
      });
    }
  }

  Future<void> _toggleTts(bool enabled) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('tts_enabled', enabled);
    setState(() => _ttsEnabled = enabled);
  }

  Future<void> _backupDatabase() async {
    try {
      final prefs = await SharedPreferences.getInstance();
      final backup = <String, dynamic>{
        'version': 1,
        'exportDate': DateTime.now().toIso8601String(),
        'students': prefs.getStringList('students') ?? [],
        'embeddings': prefs.getStringList('embeddings') ?? [],
        'attendance': prefs.getStringList('attendance') ?? [],
        'subjects': prefs.getStringList('subjects') ?? [],
        'teacherSessions': prefs.getStringList('teacherSessions') ?? [],
      };

      final jsonStr = jsonEncode(backup);

      // Save to file
      final dir = await getExportDirectory();

      final dateStr = DateTime.now()
          .toIso8601String()
          .replaceAll(RegExp(r'[:\\.]'), '-');
      final file = File('${dir.path}/backup_$dateStr.json');
      await file.writeAsString(jsonStr, flush: true);

      if (mounted) {
        // Offer to share
        final shouldShare = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Backup Created'),
            content: Text(
              'Backup saved successfully.\n\n'
              'Students: $_totalStudents\n'
              'Embeddings: $_totalEmbeddings\n'
              'Attendance Records: $_totalAttendance\n\n'
              'Share the backup file?',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('Done'),
              ),
              ElevatedButton.icon(
                onPressed: () => Navigator.pop(ctx, true),
                icon: const Icon(Icons.share, size: 18),
                label: const Text('Share'),
              ),
            ],
          ),
        );

        if (shouldShare == true) {
          await Share.shareXFiles(
            [XFile(file.path)],
            text: 'Face Attendance Database Backup',
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Backup failed: $e'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    }
  }

  Future<void> _restoreFromFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['json'],
        dialogTitle: 'Select backup JSON file',
      );
      if (result == null) return;
      final path = result.files.single.path;
      if (path == null) return;

      final file = File(path);
      final jsonStr = await file.readAsString();
      final data = jsonDecode(jsonStr) as Map<String, dynamic>;

      // Validate it looks like our backup format
      if (!data.containsKey('students') && !data.containsKey('embeddings')) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Invalid backup file — missing expected data keys.'),
              backgroundColor: AppConstants.errorColor,
            ),
          );
        }
        return;
      }

      // Show confirmation with stats before overwriting
      final studentCount = (data['students'] as List?)?.length ?? 0;
      final embeddingCount = (data['embeddings'] as List?)?.length ?? 0;
      final attendanceCount = (data['attendance'] as List?)?.length ?? 0;
      final subjectCount = (data['subjects'] as List?)?.length ?? 0;
      final sessionCount = (data['teacherSessions'] as List?)?.length ?? 0;
      final exportDate = data['exportDate'] ?? 'Unknown';

      if (!mounted) return;
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Restore From Backup?'),
          content: Text(
            'This will REPLACE all current data with the backup.\n\n'
            'Backup info:\n'
            'Export date: $exportDate\n'
            'Students: $studentCount\n'
            'Embeddings: $embeddingCount\n'
            'Attendance: $attendanceCount\n'
            'Subjects: $subjectCount\n'
            'Sessions: $sessionCount\n\n'
            'Current data will be lost. Continue?',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(ctx, true),
              icon: const Icon(Icons.restore, size: 18),
              label: const Text('Restore'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppConstants.primaryColor,
              ),
            ),
          ],
        ),
      );

      if (confirmed != true) return;

      final prefs = await SharedPreferences.getInstance();
      await prefs.setStringList('students', List<String>.from(data['students'] ?? []));
      await prefs.setStringList('embeddings', List<String>.from(data['embeddings'] ?? []));
      await prefs.setStringList('attendance', List<String>.from(data['attendance'] ?? []));
      await prefs.setStringList('subjects', List<String>.from(data['subjects'] ?? []));
      await prefs.setStringList('teacherSessions', List<String>.from(data['teacherSessions'] ?? []));

      await _loadSettings();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Restored $studentCount students, $attendanceCount attendance records from backup.'),
            backgroundColor: AppConstants.successColor,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Restore failed: $e'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    }
  }

  Future<void> _mergeFromFile() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['json'],
        dialogTitle: 'Select backup JSON file to merge',
      );
      if (result == null) return;
      final path = result.files.single.path;
      if (path == null) return;

      final file = File(path);
      final jsonStr = await file.readAsString();
      final data = jsonDecode(jsonStr) as Map<String, dynamic>;

      // Validate backup format
      if (!data.containsKey('students') && !data.containsKey('embeddings')) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Invalid backup file — missing expected data keys.'),
              backgroundColor: AppConstants.errorColor,
            ),
          );
        }
        return;
      }

      final importStudents = List<String>.from(data['students'] ?? []);
      final importEmbeddings = List<String>.from(data['embeddings'] ?? []);
      final importAttendance = List<String>.from(data['attendance'] ?? []);
      final importSubjects = List<String>.from(data['subjects'] ?? []);
      final importSessions = List<String>.from(data['teacherSessions'] ?? []);
      final exportDate = data['exportDate'] ?? 'Unknown';

      if (!mounted) return;
      final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Merge Backup Data?'),
          content: Text(
            'This will MERGE the backup data with your current data.\n'
            'Existing data will be preserved.\n\n'
            'Backup info:\n'
            'Export date: $exportDate\n'
            'Students: ${importStudents.length}\n'
            'Embeddings: ${importEmbeddings.length}\n'
            'Attendance: ${importAttendance.length}\n'
            'Subjects: ${importSubjects.length}\n'
            'Sessions: ${importSessions.length}\n\n'
            'New items will be added, duplicates will be skipped.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            ElevatedButton.icon(
              onPressed: () => Navigator.pop(ctx, true),
              icon: const Icon(Icons.merge_type, size: 18),
              label: const Text('Merge'),
              style: ElevatedButton.styleFrom(
                backgroundColor: AppConstants.primaryColor,
              ),
            ),
          ],
        ),
      );

      if (confirmed != true) return;

      final prefs = await SharedPreferences.getInstance();

      // Get current data
      final currentStudents = prefs.getStringList('students') ?? [];
      final currentEmbeddings = prefs.getStringList('embeddings') ?? [];
      final currentAttendance = prefs.getStringList('attendance') ?? [];
      final currentSubjects = prefs.getStringList('subjects') ?? [];
      final currentSessions = prefs.getStringList('teacherSessions') ?? [];

      // ── Merge Students ──
      final existingStudentNames = <String>{};
      int maxStudentId = 0;
      for (final s in currentStudents) {
        final map = jsonDecode(s) as Map<String, dynamic>;
        final name = (map['name'] as String? ?? '').toLowerCase().trim();
        final id = map['id'] as int? ?? 0;
        existingStudentNames.add(name);
        if (id > maxStudentId) maxStudentId = id;
      }

      // Map old imported IDs → new IDs (for embedding/attendance remapping)
      final idMapping = <int, int>{};
      int addedStudents = 0;

      for (final s in importStudents) {
        final map = jsonDecode(s) as Map<String, dynamic>;
        final name = (map['name'] as String? ?? '').toLowerCase().trim();
        final oldId = map['id'] as int? ?? 0;

        if (existingStudentNames.contains(name)) {
          // Student already exists — find their current ID for remapping
          for (final cs in currentStudents) {
            final cMap = jsonDecode(cs) as Map<String, dynamic>;
            if ((cMap['name'] as String? ?? '').toLowerCase().trim() == name) {
              idMapping[oldId] = cMap['id'] as int? ?? 0;
              break;
            }
          }
          continue;
        }

        // New student — assign fresh ID
        maxStudentId++;
        idMapping[oldId] = maxStudentId;
        map['id'] = maxStudentId;
        currentStudents.add(jsonEncode(map));
        existingStudentNames.add(name);
        addedStudents++;
      }

      // ── Merge Embeddings ──
      int addedEmbeddings = 0;
      final existingEmbeddingStudentIds = <int>{};
      for (final e in currentEmbeddings) {
        final map = jsonDecode(e) as Map<String, dynamic>;
        existingEmbeddingStudentIds.add(map['student_id'] as int? ?? 0);
      }

      for (final e in importEmbeddings) {
        final map = jsonDecode(e) as Map<String, dynamic>;
        final oldStudentId = map['student_id'] as int? ?? 0;
        final newStudentId = idMapping[oldStudentId] ?? oldStudentId;

        if (existingEmbeddingStudentIds.contains(newStudentId)) continue;

        map['student_id'] = newStudentId;
        currentEmbeddings.add(jsonEncode(map));
        existingEmbeddingStudentIds.add(newStudentId);
        addedEmbeddings++;
      }

      // ── Merge Attendance ──
      int addedAttendance = 0;
      final existingAttendanceKeys = <String>{};
      for (final a in currentAttendance) {
        final map = jsonDecode(a) as Map<String, dynamic>;
        existingAttendanceKeys.add('${map['student_id']}_${map['date']}');
      }

      for (final a in importAttendance) {
        final map = jsonDecode(a) as Map<String, dynamic>;
        final oldStudentId = map['student_id'] as int? ?? 0;
        final newStudentId = idMapping[oldStudentId] ?? oldStudentId;
        map['student_id'] = newStudentId;

        final key = '${newStudentId}_${map['date']}';
        if (existingAttendanceKeys.contains(key)) continue;

        currentAttendance.add(jsonEncode(map));
        existingAttendanceKeys.add(key);
        addedAttendance++;
      }

      // ── Merge Subjects ──
      int addedSubjects = 0;
      final existingSubjectNames = <String>{};
      for (final s in currentSubjects) {
        final map = jsonDecode(s) as Map<String, dynamic>;
        existingSubjectNames.add((map['name'] as String? ?? '').toLowerCase().trim());
      }

      for (final s in importSubjects) {
        final map = jsonDecode(s) as Map<String, dynamic>;
        final name = (map['name'] as String? ?? '').toLowerCase().trim();
        if (existingSubjectNames.contains(name)) continue;
        currentSubjects.add(s);
        existingSubjectNames.add(name);
        addedSubjects++;
      }

      // ── Merge Sessions ──
      int addedSessions = 0;
      final existingSessionKeys = currentSessions.toSet();
      for (final s in importSessions) {
        if (existingSessionKeys.contains(s)) continue;
        currentSessions.add(s);
        existingSessionKeys.add(s);
        addedSessions++;
      }

      // Save all merged data
      await prefs.setStringList('students', currentStudents);
      await prefs.setStringList('embeddings', currentEmbeddings);
      await prefs.setStringList('attendance', currentAttendance);
      await prefs.setStringList('subjects', currentSubjects);
      await prefs.setStringList('teacherSessions', currentSessions);

      await _loadSettings();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              'Merged: +$addedStudents students, +$addedEmbeddings embeddings, '
              '+$addedAttendance attendance, +$addedSubjects subjects, +$addedSessions sessions',
            ),
            backgroundColor: AppConstants.successColor,
            duration: const Duration(seconds: 4),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Merge failed: $e'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    }
  }

  Future<void> _clearAttendanceOnly() async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Clear Attendance Records?'),
        content: const Text(
          'This will delete all attendance records, sessions, and subjects.\n\n'
          'Students and their face embeddings will be kept.\n'
          'You can take fresh attendance afterward.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style:
                TextButton.styleFrom(foregroundColor: AppConstants.errorColor),
            child: const Text('Clear Attendance'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove('attendance');
      await prefs.remove('subjects');
      await prefs.remove('teacherSessions');

      // Also remove session_attendance_ keys
      final allKeys = prefs.getKeys();
      for (final key in allKeys) {
        if (key.startsWith('session_attendance_')) {
          await prefs.remove(key);
        }
      }

      await _loadSettings();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Attendance records cleared. Students preserved.'),
            backgroundColor: AppConstants.successColor,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text('Error: $e'),
              backgroundColor: AppConstants.errorColor),
        );
      }
    }
  }

  Future<void> _deleteExportedFiles() async {
    try {
      final dir = await getExportDirectory();

      if (!await dir.exists()) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No exported files found.')),
          );
        }
        return;
      }

      final files = dir.listSync().whereType<File>().toList();
      if (files.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No exported files found.')),
          );
        }
        return;
      }

      final confirmed = await showDialog<bool>(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('Delete All Exported Files?'),
          content: Text(
            'This will delete ${files.length} exported CSV/backup files.\nThis cannot be undone.',
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx, false),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(ctx, true),
              style:
                  TextButton.styleFrom(foregroundColor: AppConstants.errorColor),
              child: const Text('Delete All'),
            ),
          ],
        ),
      );

      if (confirmed != true) return;

      int deleted = 0;
      for (final file in files) {
        try {
          await file.delete();
          deleted++;
        } catch (_) {}
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Deleted $deleted exported files.'),
            backgroundColor: AppConstants.successColor,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content: Text('Error: $e'),
              backgroundColor: AppConstants.errorColor),
        );
      }
    }
  }

  void _confirmResetDatabase(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Reset Everything?'),
        content: const Text(
          'This will permanently delete:\n\n'
          '• All enrolled students\n'
          '• All face embeddings\n'
          '• All attendance records\n'
          '• All subjects & sessions\n\n'
          'This action CANNOT be undone.\n'
          'Consider creating a backup first.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              try {
                final prefs = await SharedPreferences.getInstance();
                final allKeys = prefs.getKeys().toList();
                for (final key in allKeys) {
                  if (key == 'tts_enabled' ||
                      key == 'required_samples') {
                    continue; // Keep settings
                  }
                  await prefs.remove(key);
                }
                await _loadSettings();
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text('All data cleared. Settings preserved.'),
                      backgroundColor: AppConstants.successColor,
                    ),
                  );
                }
              } catch (e) {
                if (mounted) {
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                        content: Text('Error: $e'),
                        backgroundColor: AppConstants.errorColor),
                  );
                }
              }
            },
            child: const Text(
              'Reset All Data',
              style: TextStyle(color: AppConstants.errorColor),
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        flexibleSpace: Container(
          decoration: BoxDecoration(gradient: AppConstants.blueGradient),
        ),
      ),
      body: AnimatedBackground(
        child: ListView(
          padding: const EdgeInsets.symmetric(vertical: 8),
          children: [
            // ── Voice Feedback ──
            _sectionHeader('Voice Feedback', Icons.volume_up),
            Card(
              margin: const EdgeInsets.symmetric(
                horizontal: AppConstants.paddingMedium,
                vertical: 4,
              ),
              child: SwitchListTile(
                title: const Text(
                  'TTS Attendance Confirmation',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600),
                ),
                subtitle: const Text(
                  'Speak student name when attendance is marked',
                  style: TextStyle(fontSize: 12, color: AppConstants.textTertiary),
                ),
                secondary: Icon(
                  _ttsEnabled ? Icons.record_voice_over : Icons.voice_over_off,
                  color: _ttsEnabled
                      ? AppConstants.primaryColor
                      : AppConstants.textTertiary,
                  size: 22,
                ),
                value: _ttsEnabled,
                activeThumbColor: AppConstants.primaryColor,
                onChanged: _toggleTts,
              ),
            ),

            const SizedBox(height: 8),

            // ── Data Management ──
            _sectionHeader('Data Management', Icons.storage),
            _buildDataCard(),

            const SizedBox(height: 8),

            // ── Model Information ──
            _sectionHeader('Models & Algorithms', Icons.memory),
            Card(
              margin: const EdgeInsets.symmetric(
                horizontal: AppConstants.paddingMedium,
                vertical: 4,
              ),
              child: Padding(
                padding: const EdgeInsets.all(AppConstants.paddingMedium),
                child: Column(
                  children: [
                    _infoRow('Face Detector', 'Google ML Kit (MediaPipe)'),
                    const Divider(height: 1, color: AppConstants.dividerColor),
                    _infoRow('Embedding Model', 'FaceNet-128 (TFLite)'),
                    const Divider(height: 1, color: AppConstants.dividerColor),
                    _infoRow('Embedding Dimension', '128D vectors'),
                    const Divider(height: 1, color: AppConstants.dividerColor),
                    _infoRow('Matching Algorithm', 'KNN (Euclidean)'),
                    const Divider(height: 1, color: AppConstants.dividerColor),
                    _infoRow('Inference', 'XNNPack CPU, 4 threads'),
                    const Divider(height: 1, color: AppConstants.dividerColor),
                    _infoRow('Enrollment Samples', '20 per student'),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 8),

            // ── About ──
            _sectionHeader('About', Icons.info_outline),
            Card(
              margin: const EdgeInsets.symmetric(
                horizontal: AppConstants.paddingMedium,
                vertical: 4,
              ),
              child: Padding(
                padding: const EdgeInsets.all(AppConstants.paddingMedium),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _infoRow('App', AppConstants.appName),
                    const Divider(height: 1, color: AppConstants.dividerColor),
                    _infoRow('Version', AppConstants.appVersion),
                    const Divider(height: 1, color: AppConstants.dividerColor),
                    _infoRow('Storage', 'SharedPreferences (Offline)'),
                    const SizedBox(height: 12),
                    Text(
                      AppConstants.subtitle,
                      style: const TextStyle(
                        fontSize: 12,
                        color: AppConstants.textTertiary,
                        height: 1.4,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),
            Center(
              child: Text(
                '© 2026 Face Recognition Attendance System',
                style: const TextStyle(
                  fontSize: 11,
                  color: AppConstants.textTertiary,
                ),
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  // ── Data Management Card ──
  Widget _buildDataCard() {
    return Card(
      margin: const EdgeInsets.symmetric(
        horizontal: AppConstants.paddingMedium,
        vertical: 4,
      ),
      child: Padding(
        padding: const EdgeInsets.all(AppConstants.paddingMedium),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Real data stats
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppConstants.inputFill,
                borderRadius: BorderRadius.circular(10),
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _miniStat('Students', _totalStudents.toString()),
                      _miniStat('Embeddings', _totalEmbeddings.toString()),
                      _miniStat('Records', _totalAttendance.toString()),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      _miniStat('Subjects', _totalSubjects.toString()),
                      _miniStat('Sessions', _totalSessions.toString()),
                      _miniStat('Size', _dataSize),
                    ],
                  ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // Backup
            _actionTile(
              icon: Icons.backup,
              title: 'Backup Database',
              subtitle: 'Export all data as JSON (shareable)',
              color: AppConstants.primaryColor,
              onTap: _backupDatabase,
            ),
            const Divider(height: 1, color: AppConstants.dividerColor),

            // Restore from backup file
            _actionTile(
              icon: Icons.restore,
              title: 'Restore From Backup',
              subtitle: 'Upload a JSON backup to restore app state',
              color: AppConstants.primaryColor,
              onTap: _restoreFromFile,
            ),
            const Divider(height: 1, color: AppConstants.dividerColor),

            // Merge from backup file
            _actionTile(
              icon: Icons.merge_type,
              title: 'Merge From Backup',
              subtitle: 'Add new data from JSON without erasing existing',
              color: AppConstants.primaryColor,
              onTap: _mergeFromFile,
            ),
            const Divider(height: 1, color: AppConstants.dividerColor),

            // Clear attendance only
            _actionTile(
              icon: Icons.event_busy,
              title: 'Clear Attendance Records',
              subtitle: 'Keep students, remove attendance & sessions',
              color: AppConstants.warningColor,
              onTap: _clearAttendanceOnly,
            ),
            const Divider(height: 1, color: AppConstants.dividerColor),

            // Delete exports
            _actionTile(
              icon: Icons.folder_delete,
              title: 'Delete Exported Files',
              subtitle: 'Remove all saved CSV and backup files',
              color: AppConstants.warningColor,
              onTap: _deleteExportedFiles,
            ),
            const Divider(height: 1, color: AppConstants.dividerColor),

            // Full reset
            _actionTile(
              icon: Icons.delete_forever,
              title: 'Reset All Data',
              subtitle: 'Delete everything — students, faces, attendance',
              color: AppConstants.errorColor,
              onTap: () => _confirmResetDatabase(context),
            ),
          ],
        ),
      ),
    );
  }

  // ── Helper Widgets ──

  Widget _sectionHeader(String title, IconData icon) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
      child: Row(
        children: [
          Icon(icon, color: AppConstants.primaryColor, size: 18),
          const SizedBox(width: 8),
          Text(
            title,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.bold,
              color: AppConstants.textSecondary,
              letterSpacing: 0.3,
            ),
          ),
        ],
      ),
    );
  }

  Widget _miniStat(String label, String value) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
            color: AppConstants.textPrimary,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: const TextStyle(
            fontSize: 10,
            color: AppConstants.textTertiary,
          ),
        ),
      ],
    );
  }

  Widget _actionTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required Color color,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 4),
        child: Row(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: color == AppConstants.errorColor
                          ? color
                          : AppConstants.textPrimary,
                    ),
                  ),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      fontSize: 11,
                      color: AppConstants.textTertiary,
                    ),
                  ),
                ],
              ),
            ),
            Icon(Icons.chevron_right,
                color: AppConstants.textTertiary, size: 18),
          ],
        ),
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontSize: 13,
              fontWeight: FontWeight.w500,
              color: AppConstants.textSecondary,
            ),
          ),
          Flexible(
            child: Text(
              value,
              style: const TextStyle(
                fontSize: 13,
                color: AppConstants.textPrimary,
                fontWeight: FontWeight.w600,
              ),
              textAlign: TextAlign.right,
            ),
          ),
        ],
      ),
    );
  }
}
