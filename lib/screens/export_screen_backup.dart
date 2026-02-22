import 'dart:convert';
import 'dart:io';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:permission_handler/permission_handler.dart';
import 'package:share_plus/share_plus.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../database/database_manager.dart';
import '../modules/m4_attendance_management.dart';
import '../models/attendance_model.dart';
import '../utils/constants.dart';
import '../widgets/animated_background.dart';

class ExportScreen extends StatefulWidget {
  const ExportScreen({super.key});

  @override
  State<ExportScreen> createState() => _ExportScreenState();
}

class _ExportScreenState extends State<ExportScreen> {
  late final DatabaseManager _dbManager;
  late final AttendanceManagementModule _attendanceModule;
  final List<ExportRecord> _history = [];
  Directory? _exportDirectory;
  bool _isInitializing = true;
  bool _isExporting = false;
  static const int _maxRecentExports = 5;

  static const MethodChannel _platform = MethodChannel(
    'com.coad.faceattendance/save',
  );

  @override
  void initState() {
    super.initState();
    _setupExportEnvironment();
  }

  Future<void> _exportEmbeddings() async {
    if (_isExporting || _exportDirectory == null) return;
    setState(() {
      _isExporting = true;
    });

    final timeStamp = DateTime.now().toIso8601String().replaceAll(
      RegExp(r'[:\\.]'),
      '-',
    );
    final file = File(
      '${_exportDirectory!.path}/embeddings_csv_$timeStamp.csv',
    );

    try {
      final csv = await _attendanceModule.exportEmbeddingsCSV();
      await file.writeAsString(csv, flush: true);

      // Save to Downloads folder
      try {
        final downloadsDir = Directory('/storage/emulated/0/Download');
        if (await downloadsDir.exists()) {
          final downloadFile = File(
            '${downloadsDir.path}/${p.basename(file.path)}',
          );
          await downloadFile.writeAsString(csv, flush: true);
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  '✅ Embeddings saved to Downloads/${p.basename(file.path)}',
                ),
                backgroundColor: AppConstants.successColor,
                duration: const Duration(seconds: 4),
              ),
            );
          }
        }
      } catch (e) {
        debugPrint('⚠️ Could not save to Downloads: $e');
      }

      final record = ExportRecord(
        format: 'EMBEDDINGS',
        path: file.path,
        timestamp: DateTime.now(),
      );
      if (mounted) {
        setState(() {
          _history.insert(0, record);
          if (_history.length > _maxRecentExports) {
            _history.removeRange(_maxRecentExports, _history.length);
          }
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Embeddings exported to ${file.path}'),
            backgroundColor: AppConstants.successColor,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Export failed: $e'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isExporting = false);
    }
  }

  Future<void> _exportSubjectAttendance() async {
    if (_isExporting || _exportDirectory == null) return;

    setState(() => _isExporting = true);

    try {
      final today = DateTime.now();
      var sessions = await _dbManager.getTeacherSessionsByDate(today);

      if (sessions.isEmpty) {
        final allSessions = await _dbManager.getAllTeacherSessions();
        if (allSessions.isEmpty) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('No attendance sessions found'),
                backgroundColor: AppConstants.warningColor,
              ),
            );
          }
          return;
        }

        allSessions.sort((a, b) => b.date.compareTo(a.date));
        final latest = allSessions.first.date;
        sessions = allSessions.where((s) {
          return s.date.year == latest.year &&
              s.date.month == latest.month &&
              s.date.day == latest.day;
        }).toList();

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'No session today. Exporting most recent (${latest.toString().split(' ')[0]}).',
              ),
              backgroundColor: AppConstants.warningColor,
              duration: const Duration(seconds: 3),
            ),
          );
        }
      }

      var exportedCount = 0;
      var savedToDownloads = false;

      final prefs = await SharedPreferences.getInstance();

      for (final session in sessions) {
        final sessionAttendance = _loadSessionAttendance(
          prefs: prefs,
          teacherName: session.teacherName,
          subjectId: session.subjectId,
          date: session.date,
        );
        final csv = await exportSubjectAttendanceAsCSV(
          _dbManager,
          session.teacherName,
          session.subjectName,
          session.date,
          sessionAttendance: sessionAttendance,
        );

        final timeStamp = DateTime.now().toIso8601String().replaceAll(
          RegExp(r'[:\\.]'),
          '-',
        );
        final rawName =
            '${session.teacherName}_${session.subjectName}_$timeStamp.csv'
                .replaceAll(' ', '_');
        final filename = _safeFilename(rawName);
        final file = File('${_exportDirectory!.path}/$filename');
        await file.writeAsString(csv, flush: true);

        if (!await file.exists()) {
          debugPrint('Subject attendance export failed to write: $filename');
          continue;
        }

        exportedCount++;

        final bytes = await file.readAsBytes();
        final saved = await _saveFileToDownloads(filename, bytes);
        if (saved == true) {
          savedToDownloads = true;
        }

        final record = ExportRecord(
          format: 'SUBJECT_ATTENDANCE',
          path: file.path,
          timestamp: DateTime.now(),
        );

        if (mounted) {
          setState(() {
            _history.insert(0, record);
            if (_history.length > _maxRecentExports) {
              _history.removeRange(_maxRecentExports, _history.length);
            }
          });
        }
      }

      if (mounted) {
        if (exportedCount == 0) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('No subject attendance files were exported'),
              backgroundColor: AppConstants.errorColor,
            ),
          );
          return;
        }

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              savedToDownloads
                  ? '✅ Subject attendance exported to Downloads/FaceAttendanceExports (${exportedCount} session${exportedCount > 1 ? 's' : ''})'
                  : '✅ Subject attendance exported (${exportedCount} session${exportedCount > 1 ? 's' : ''})',
            ),
            backgroundColor: AppConstants.successColor,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } catch (e) {
      debugPrint('Subject attendance export error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Export failed: $e'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isExporting = false);
    }
  }

  String _safeFilename(String value) {
    final cleaned = value.replaceAll(RegExp(r'[<>:"/\\|?*]'), '_');
    final trimmed = cleaned.replaceAll(RegExp(r'_+'), '_').trim();
    return trimmed.isEmpty ? 'subject_attendance.csv' : trimmed;
  }

  Map<int, AttendanceStatus>? _loadSessionAttendance({
    required SharedPreferences prefs,
    required String teacherName,
    required int subjectId,
    required DateTime date,
  }) {
    final key = _sessionAttendanceKey(
      teacherName: teacherName,
      subjectId: subjectId,
      date: date,
    );
    final raw = prefs.getString(key);
    if (raw == null || raw.isEmpty) return null;

    try {
      final decoded = jsonDecode(raw) as Map<String, dynamic>;
      final result = <int, AttendanceStatus>{};
      decoded.forEach((studentId, statusName) {
        final id = int.tryParse(studentId);
        if (id == null) return;
        final status = AttendanceStatus.values.firstWhere(
          (s) => s.name == statusName,
          orElse: () => AttendanceStatus.absent,
        );
        result[id] = status;
      });
      return result;
    } catch (e) {
      debugPrint('Failed to parse session attendance: $e');
      return null;
    }
  }

  String _sessionAttendanceKey({
    required String teacherName,
    required int subjectId,
    required DateTime date,
  }) {
    final dateStr =
        '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
    final safeTeacher = teacherName
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .trim();
    return 'session_attendance_${safeTeacher}_${subjectId}_$dateStr';
  }


  Future<void> _setupExportEnvironment() async {
    _dbManager = DatabaseManager();
    await _dbManager.database;
    _attendanceModule = AttendanceManagementModule(_dbManager);

    Directory? baseDir;
    try {
      if (Platform.isAndroid) {
        final storageStatus = await Permission.storage.request();
        if (storageStatus.isGranted) {
          final externals = await getExternalStorageDirectories(
            type: StorageDirectory.downloads,
          );
          if (externals != null && externals.isNotEmpty) {
            baseDir = Directory(externals.first.path);
          }
        }
      }
    } catch (_) {}

    baseDir ??= await getApplicationDocumentsDirectory();
    final dir = Directory('${baseDir.path}/FaceAttendanceExports');
    await dir.create(recursive: true);

    final existing =
        dir
            .listSync()
            .whereType<File>()
            .map(
              (file) => ExportRecord(
                format: file.path.split('.').last.toUpperCase(),
                path: file.path,
                timestamp: FileStat.statSync(file.path).modified,
              ),
            )
            .toList()
          ..sort((a, b) => b.timestamp.compareTo(a.timestamp));

    if (!mounted) return;
    setState(() {
      _exportDirectory = dir;
      _history.addAll(existing.take(_maxRecentExports));
      _isInitializing = false;
    });
  }

  Future<void> _shareLastExport() async {
    if (_history.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No exports available to share')),
      );
      return;
    }
    final last = _history.first; // Most recent
    final file = File(last.path);
    if (await file.exists()) {
      await Share.shareXFiles([
        XFile(file.path),
      ], text: 'Face Attendance Export: ${last.format}');
    } else {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('File not found')));
    }
  }

  /// Returns true if saved into public Downloads (Android MediaStore), false otherwise.
  Future<bool?> _saveFileToDownloads(String filename, List<int> bytes) async {
    try {
      if (!Platform.isAndroid) return null;
      debugPrint('Attempting to save $filename to Downloads via MediaStore...');
      final base64data = base64Encode(bytes);
      final res = await _platform.invokeMethod('saveToDownloads', {
        'filename': filename,
        'dataBase64': base64data,
        'subFolder': 'FaceAttendanceExports',
      });
      debugPrint('MediaStore save result: $res');
      return res == true;
    } catch (e) {
      debugPrint('MediaStore save failed: $e');
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Export Data'), flexibleSpace: Container(decoration: BoxDecoration(gradient: AppConstants.blueGradient))),
      body: AnimatedBackground(
        child: _isInitializing
            ? const Center(child: CircularProgressIndicator())
            : SingleChildScrollView(
                padding: const EdgeInsets.all(AppConstants.paddingMedium),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    // Export Options Card
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(AppConstants.paddingLarge),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(10),
                                  decoration: BoxDecoration(
                                    color: AppConstants.accentColor.withAlpha(26),
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: const Icon(
                                    Icons.download,
                                    color: AppConstants.accentColor,
                                    size: 24,
                                  ),
                                ),
                                const SizedBox(width: AppConstants.paddingMedium),
                                const Text(
                                  'Export Formats',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: AppConstants.paddingMedium),
                            const Text(
                              'Files are saved inside FaceAttendanceExports under your device Downloads (Android) or Documents.',
                              style: TextStyle(
                                fontSize: 13,
                                color: AppConstants.textSecondary,
                              ),
                            ),
                            const SizedBox(height: AppConstants.paddingLarge),
                            _exportButton(
                              icon: Icons.table_chart,
                              label: 'Attendance (CSV)',
                              onPressed: _isExporting
                                  ? null
                                  : () => _exportData('CSV'),
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            _exportButton(
                              icon: Icons.subject,
                              label: 'Subject Attendance (CSV)',
                              onPressed: _isExporting
                                  ? null
                                  : () => _exportSubjectAttendance(),
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            _exportButton(
                              icon: Icons.memory,
                              label: 'Embeddings (CSV)',
                              onPressed: _isExporting
                                  ? null
                                  : () => _exportEmbeddings(),
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            ElevatedButton.icon(
                              onPressed: _isExporting
                                  ? null
                                  : () => _shareLastExport(),
                              icon: const Icon(Icons.share),
                              label: const Text('Share Last Export'),
                              style: ElevatedButton.styleFrom(
                                backgroundColor: AppConstants.primaryColor,
                              ),
                            ),
                            const SizedBox(height: AppConstants.paddingMedium),
                            Container(
                              padding: const EdgeInsets.all(AppConstants.paddingSmall),
                              decoration: BoxDecoration(
                                color: AppConstants.inputFill,
                                borderRadius: BorderRadius.circular(
                                  AppConstants.borderRadius,
                                ),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  const Text(
                                    'Export Location:',
                                    style: TextStyle(
                                      fontSize: 12,
                                      color: AppConstants.textTertiary,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    _exportDirectory?.path ?? 'Preparing folder...',
                                    style: const TextStyle(
                                      fontSize: 11,
                                      color: AppConstants.textSecondary,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            Row(
                              children: [
                                if (_isExporting)
                                  const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  )
                                else
                                  Icon(
                                    Icons.check_circle,
                                    size: 16,
                                    color: AppConstants.successColor,
                                  ),
                                const SizedBox(width: 8),
                                Text(
                                  _isExporting ? 'Exporting...' : 'Ready',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: _isExporting
                                        ? AppConstants.warningColor
                                        : AppConstants.successColor,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: AppConstants.paddingLarge),
                    // Recent Exports
                    Card(
                      child: Padding(
                        padding: const EdgeInsets.all(AppConstants.paddingLarge),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.all(10),
                                  decoration: BoxDecoration(
                                    color: AppConstants.primaryColor.withAlpha(26),
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                  child: const Icon(
                                    Icons.history,
                                    color: AppConstants.primaryColor,
                                    size: 24,
                                  ),
                                ),
                                const SizedBox(width: AppConstants.paddingMedium),
                                const Text(
                                  'Recent Exports',
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: AppConstants.paddingMedium),
                            if (_history.isEmpty)
                              const Text(
                                'No exports yet. Tap a button above to create one.',
                              )
                            else
                              Column(
                                children: _history.take(_maxRecentExports).map(_historyRow).toList(),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
        );
  }

  Widget _glassCard({required Widget child}) {
    return Card(
      color: Colors.white.withAlpha((0.18 * 255).round()),
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
      ),
      clipBehavior: Clip.antiAlias,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(AppConstants.borderRadius),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 18, sigmaY: 18),
          child: child,
        ),
      ),
    );
  }

  Widget _exportButton({
    required IconData icon,
    required String label,
    required VoidCallback? onPressed,
  }) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon),
      label: Text(label),
      style: ElevatedButton.styleFrom(
        backgroundColor: AppConstants.primaryColor,
        disabledBackgroundColor: AppConstants.inputFill,
      ),
    );
  }

  Widget _historyRow(ExportRecord record) {
    return Padding(
      padding: const EdgeInsets.only(bottom: AppConstants.paddingSmall),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                record.format,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(
                _friendlyTimestamp(record.timestamp),
                style: const TextStyle(fontSize: 12, color: Colors.white70),
              ),
            ],
          ),
          const SizedBox(height: AppConstants.paddingSmall / 2),
          SelectableText(
            record.path,
            style: const TextStyle(fontSize: 12, color: Colors.white70),
          ),
          if (record != _history.last) Divider(color: Colors.white24),
        ],
      ),
    );
  }

  String _friendlyTimestamp(DateTime timestamp) {
    final year = timestamp.year;
    final month = timestamp.month.toString().padLeft(2, '0');
    final day = timestamp.day.toString().padLeft(2, '0');
    final hour = timestamp.hour.toString().padLeft(2, '0');
    final minute = timestamp.minute.toString().padLeft(2, '0');
    return '$year-$month-$day $hour:$minute';
  }

  Future<void> _exportData(String format) async {
    if (_isExporting || _exportDirectory == null) return;
    setState(() {
      _isExporting = true;
    });

    final extension = format == 'PDF'
        ? 'pdf'
        : format == 'Excel'
        ? 'xlsx'
        : 'csv';
    final timeStamp = DateTime.now().toIso8601String().replaceAll(
      RegExp(r'[:\\.]'),
      '-',
    );
    final file = File(
      '${_exportDirectory!.path}/attendance_${format.toLowerCase()}_$timeStamp.$extension',
    );

    try {
      if (format == 'PDF') {
        await _writePdf(file);
        try {
          final bytes = await file.readAsBytes();
          final saved = await _saveFileToDownloads(
            p.basename(file.path),
            bytes,
          );
          if (saved == true) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  'PDF exported to Downloads/${p.basename(file.path)}',
                ),
                backgroundColor: AppConstants.successColor,
                duration: const Duration(seconds: 3),
              ),
            );
          } else {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('PDF saved locally. Use Share button to export.'),
                backgroundColor: AppConstants.warningColor,
              ),
            );
          }
        } catch (_) {}
      } else {
        final csv = await _attendanceModule.exportAsCSV();
        await file.writeAsString(csv, flush: true);
        
        // Try to save to Downloads folder
        try {
          final downloadsDir = Directory('/storage/emulated/0/Download');
          if (await downloadsDir.exists()) {
            final downloadFile = File(
              '${downloadsDir.path}/${p.basename(file.path)}',
            );
            await downloadFile.writeAsString(csv, flush: true);
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(
                    '✅ CSV exported to Downloads/${p.basename(file.path)}',
                  ),
                  backgroundColor: AppConstants.successColor,
                  duration: const Duration(seconds: 3),
                ),
              );
            }
          } else {
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text('CSV saved to: ${file.path}'),
                  backgroundColor: AppConstants.warningColor,
                ),
              );
            }
          }
        } catch (e) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('CSV saved to: ${file.path}'),
                backgroundColor: AppConstants.warningColor,
              ),
            );
          }
        }
      }

      final record = ExportRecord(
        format: format,
        path: file.path,
        timestamp: DateTime.now(),
      );
      if (mounted) {
        setState(() {
          _history.insert(0, record);
          if (_history.length > _maxRecentExports) {
            _history.removeRange(_maxRecentExports, _history.length);
          }
        });
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Exported to ${file.path}'),
            backgroundColor: AppConstants.successColor,
            duration: const Duration(seconds: 3),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Export failed: $e'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isExporting = false;
        });
      }
    }
  }

  Future<void> _writePdf(File file) async {
    final csv = await _attendanceModule.exportAsCSV();
    final lines = LineSplitter()
        .convert(csv)
        .where((line) => line.trim().isNotEmpty)
        .toList();
    final headers = lines.isEmpty ? <String>[] : lines.first.split(',');
    final data = lines.length > 1
        ? lines.skip(1).map((line) => line.split(',')).toList()
        : <List<String>>[];

    final pdf = pw.Document();
    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: pw.EdgeInsets.all(24),
        build: (context) {
          final children = <pw.Widget>[
            pw.Text(
              'Attendance Export',
              style: pw.TextStyle(fontSize: 26, fontWeight: pw.FontWeight.bold),
            ),
            pw.SizedBox(height: 8),
            pw.Text(
              'Generated from ${AppConstants.appName} on ${_friendlyTimestamp(DateTime.now())}',
            ),
            pw.SizedBox(height: 16),
          ];
          if (headers.isNotEmpty) {
            children.add(
              pw.TableHelper.fromTextArray(
                headers: headers,
                data: data,
                headerStyle: pw.TextStyle(
                  fontWeight: pw.FontWeight.bold,
                  color: PdfColors.white,
                ),
                headerDecoration: pw.BoxDecoration(
                  color: PdfColors.blueGrey900,
                ),
                cellAlignment: pw.Alignment.centerLeft,
                border: pw.TableBorder.all(
                  color: PdfColors.grey300,
                  width: 0.5,
                ),
              ),
            );
          } else {
            children.add(pw.Text('No attendance data available.'));
          }
          return children;
        },
      ),
    );

    final bytes = await pdf.save();
    await file.writeAsBytes(bytes, flush: true);
  }
}

class ExportRecord {
  final String format;
  final String path;
  final DateTime timestamp;

  ExportRecord({
    required this.format,
    required this.path,
    required this.timestamp,
  });
}
