import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:share_plus/share_plus.dart';

import '../database/database_manager.dart';
import '../modules/m4_attendance_management.dart';
import '../utils/constants.dart';
import '../utils/export_utils.dart';
import '../widgets/animated_background.dart';

class ExportScreen extends StatefulWidget {
  const ExportScreen({super.key});

  @override
  State<ExportScreen> createState() => _ExportScreenState();
}

class _ExportScreenState extends State<ExportScreen> {
  late final DatabaseManager _dbManager;
  late final AttendanceManagementModule _attendanceModule;
  Directory? _exportDirectory;
  bool _isInitializing = true;
  bool _isExporting = false;

  // Saved attendance files
  List<SavedAttendanceFile> _savedFiles = [];

  static const MethodChannel _platform = MethodChannel(
    'com.coad.faceattendance/save',
  );

  @override
  void initState() {
    super.initState();
    _setupExportEnvironment();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _setupExportEnvironment() async {
    _dbManager = DatabaseManager();
    await _dbManager.database;
    _attendanceModule = AttendanceManagementModule(_dbManager);

    final dir = await getExportDirectory();

    if (!mounted) return;
    setState(() {
      _exportDirectory = dir;
      _isInitializing = false;
    });

    await _loadSavedFiles();
  }

  Future<void> _loadSavedFiles() async {
    if (_exportDirectory == null) return;

    final files =
        _exportDirectory!
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.csv') || f.path.endsWith('.json'))
            .map((file) {
              final name = p.basenameWithoutExtension(file.path);
              final stat = FileStat.statSync(file.path);
              return SavedAttendanceFile(
                fileName: p.basename(file.path),
                filePath: file.path,
                createdAt: stat.modified,
                displayName: name.replaceAll('_', ' '),
              );
            })
            .toList()
          ..sort((a, b) => b.createdAt.compareTo(a.createdAt));

    if (mounted) {
      setState(() {
        _savedFiles = files;
      });
    }
  }

  Future<void> _shareFile(SavedAttendanceFile savedFile) async {
    final file = File(savedFile.filePath);
    if (await file.exists()) {
      await Share.shareXFiles([
        XFile(file.path),
      ], text: 'Attendance Report: ${savedFile.displayName}');
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('File not found'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    }
  }

  Future<void> _deleteFile(SavedAttendanceFile savedFile) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete File'),
        content: Text(
          'Delete "${savedFile.fileName}"?\nThis cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(
              foregroundColor: AppConstants.errorColor,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    try {
      final file = File(savedFile.filePath);
      if (await file.exists()) {
        await file.delete();
      }
      await _loadSavedFiles();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Deleted: ${savedFile.fileName}'),
            backgroundColor: AppConstants.successColor,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to delete: $e'),
            backgroundColor: AppConstants.errorColor,
          ),
        );
      }
    }
  }

  Future<void> _exportEmbeddings() async {
    if (_isExporting || _exportDirectory == null) return;
    setState(() => _isExporting = true);

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

      try {
        final bytes = await file.readAsBytes();
        await _saveFileToDownloads(p.basename(file.path), bytes);
      } catch (_) {}

      await _loadSavedFiles();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('✅ Embeddings exported: ${p.basename(file.path)}'),
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

  Future<void> _exportGeneralAttendance() async {
    if (_isExporting || _exportDirectory == null) return;
    setState(() => _isExporting = true);

    // Always use a fixed filename so it stays up-to-date
    final file = File('${_exportDirectory!.path}/attendance_register.csv');

    try {
      final csv = await _attendanceModule.exportAsCSV();
      await file.writeAsString(csv, flush: true);

      try {
        final bytes = await file.readAsBytes();
        await _saveFileToDownloads(p.basename(file.path), bytes);
      } catch (_) {}

      await _loadSavedFiles();

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('✅ Attendance register exported: ${p.basename(file.path)}'),
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

  Future<bool?> _saveFileToDownloads(String filename, List<int> bytes) async {
    try {
      if (!Platform.isAndroid) return null;
      final base64data = base64Encode(bytes);
      final res = await _platform.invokeMethod('saveToDownloads', {
        'filename': filename,
        'dataBase64': base64data,
        'subFolder': 'FaceAttendanceExports',
      });
      return res == true;
    } catch (e) {
      debugPrint('MediaStore save failed: $e');
      return null;
    }
  }

  // ==================== BUILD UI ====================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Export & Attendance'),
        flexibleSpace: Container(
          decoration: BoxDecoration(gradient: AppConstants.blueGradient),
        ),
      ),
      body: AnimatedBackground(
        child: _isInitializing
            ? const Center(child: CircularProgressIndicator())
            : SingleChildScrollView(
                padding: const EdgeInsets.all(AppConstants.paddingMedium),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    _buildExportOptionsCard(),
                    const SizedBox(height: AppConstants.paddingMedium),
                    _buildSubjectAttendanceCard(),
                    const SizedBox(height: AppConstants.paddingMedium),
                    _buildSavedFilesCard(),
                  ],
                ),
              ),
      ),
    );
  }

  // ==================== EXPORT OPTIONS CARD ====================

  Widget _buildExportOptionsCard() {
    return Card(
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
                  'Quick Export',
                  style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: AppConstants.paddingMedium),
            const Text(
              'Export all attendance data or embeddings as CSV.',
              style: TextStyle(fontSize: 13, color: AppConstants.textSecondary),
            ),
            const SizedBox(height: AppConstants.paddingMedium),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isExporting ? null : _exportGeneralAttendance,
                    icon: const Icon(Icons.table_chart, size: 18),
                    label: const Text('Attendance CSV'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppConstants.primaryColor,
                      disabledBackgroundColor: AppConstants.inputFill,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isExporting ? null : _exportEmbeddings,
                    icon: const Icon(Icons.memory, size: 18),
                    label: const Text('Embeddings CSV'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppConstants.primaryColor,
                      disabledBackgroundColor: AppConstants.inputFill,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // ==================== SUBJECT ATTENDANCE CARD ====================

  List<SavedAttendanceFile> get _subjectFiles => _savedFiles.where((f) =>
      !f.fileName.startsWith('attendance_') &&
      !f.fileName.startsWith('embeddings_') &&
      !f.fileName.startsWith('backup_') &&
      f.fileName.endsWith('.csv')).toList();

  List<SavedAttendanceFile> get _registerFiles => _savedFiles.where((f) =>
      f.fileName == 'attendance_register.csv').toList();

  List<SavedAttendanceFile> get _otherFiles => _savedFiles.where((f) =>
      f.fileName.startsWith('attendance_') && f.fileName != 'attendance_register.csv' ||
      f.fileName.startsWith('embeddings_') ||
      f.fileName.startsWith('backup_') ||
      f.fileName.endsWith('.json')).toList();

  Widget _buildSubjectAttendanceCard() {
    final subjects = _subjectFiles;
    final registers = _registerFiles;
    final allFiles = [...registers, ...subjects];

    return Card(
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
                    color: AppConstants.successColor.withAlpha(26),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const Icon(
                    Icons.class_,
                    color: AppConstants.successColor,
                    size: 24,
                  ),
                ),
                const SizedBox(width: AppConstants.paddingMedium),
                const Expanded(
                  child: Text(
                    'Subject Attendance',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),
                if (allFiles.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                    decoration: BoxDecoration(
                      color: AppConstants.successColor.withAlpha(25),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: Text(
                      '${allFiles.length}',
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: AppConstants.successColor,
                      ),
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 4),
            const Text(
              'Auto-created when you submit attendance for a subject.',
              style: TextStyle(fontSize: 12, color: AppConstants.textTertiary),
            ),
            const SizedBox(height: AppConstants.paddingMedium),
            if (allFiles.isEmpty)
              Container(
                padding: const EdgeInsets.all(AppConstants.paddingLarge),
                alignment: Alignment.center,
                child: Column(
                  children: [
                    Icon(
                      Icons.class_outlined,
                      size: 48,
                      color: AppConstants.textTertiary.withAlpha(100),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'No subject attendance files yet.\nSubmit attendance with a subject to auto-generate.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: AppConstants.textTertiary,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              )
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: allFiles.length,
                separatorBuilder: (_, __) =>
                    const Divider(color: AppConstants.dividerColor, height: 1),
                itemBuilder: (context, index) {
                  return _buildFileRow(allFiles[index]);
                },
              ),
          ],
        ),
      ),
    );
  }

  // ==================== SAVED FILES CARD ====================

  Widget _buildSavedFilesCard() {
    final otherFiles = _otherFiles;

    return Card(
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
                    Icons.folder_open,
                    color: AppConstants.primaryColor,
                    size: 24,
                  ),
                ),
                const SizedBox(width: AppConstants.paddingMedium),
                const Expanded(
                  child: Text(
                    'Other Files',
                    style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                  ),
                ),
                IconButton(
                  onPressed: _loadSavedFiles,
                  icon: const Icon(Icons.refresh, size: 20),
                  color: AppConstants.primaryColor,
                  tooltip: 'Refresh',
                ),
              ],
            ),
            const SizedBox(height: 4),
            const Text(
              'Embeddings exports, backups and other files.',
              style: TextStyle(fontSize: 12, color: AppConstants.textTertiary),
            ),
            const SizedBox(height: AppConstants.paddingMedium),
            if (otherFiles.isEmpty)
              Container(
                padding: const EdgeInsets.all(AppConstants.paddingLarge),
                alignment: Alignment.center,
                child: Column(
                  children: [
                    Icon(
                      Icons.folder_off,
                      size: 48,
                      color: AppConstants.textTertiary.withAlpha(100),
                    ),
                    const SizedBox(height: 8),
                    const Text(
                      'No other files yet.',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: AppConstants.textTertiary,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              )
            else
              ListView.separated(
                shrinkWrap: true,
                physics: const NeverScrollableScrollPhysics(),
                itemCount: otherFiles.length,
                separatorBuilder: (_, __) =>
                    const Divider(color: AppConstants.dividerColor, height: 1),
                itemBuilder: (context, index) {
                  return _buildFileRow(otherFiles[index]);
                },
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildFileRow(SavedAttendanceFile file) {
    // Categorize files for better visual distinction
    final bool isSubjectFile =
        !file.fileName.startsWith('attendance_') &&
        !file.fileName.startsWith('embeddings_') &&
        !file.fileName.startsWith('backup_') &&
        file.fileName.endsWith('.csv');
    final bool isRegister = file.fileName == 'attendance_register.csv';
    final bool isBackup = file.fileName.endsWith('.json');

    final Color iconColor;
    final IconData icon;
    final String badge;

    if (isRegister) {
      iconColor = AppConstants.primaryColor;
      icon = Icons.table_chart;
      badge = 'REGISTER';
    } else if (isSubjectFile) {
      iconColor = AppConstants.successColor;
      icon = Icons.class_;
      badge = 'SUBJECT';
    } else if (isBackup) {
      iconColor = AppConstants.warningColor;
      icon = Icons.backup;
      badge = 'BACKUP';
    } else {
      iconColor = AppConstants.primaryColor;
      icon = Icons.insert_drive_file;
      badge = '';
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 10),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: iconColor.withAlpha(20),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, color: iconColor, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    if (badge.isNotEmpty) ...[
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 1),
                        decoration: BoxDecoration(
                          color: iconColor.withAlpha(25),
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          badge,
                          style: TextStyle(
                            fontSize: 9,
                            fontWeight: FontWeight.bold,
                            color: iconColor,
                          ),
                        ),
                      ),
                      const SizedBox(width: 6),
                    ],
                    Expanded(
                      child: Text(
                        file.fileName,
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: AppConstants.textPrimary,
                        ),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  _friendlyTimestamp(file.createdAt),
                  style: const TextStyle(
                    fontSize: 11,
                    color: AppConstants.textTertiary,
                  ),
                ),
              ],
            ),
          ),
          IconButton(
            onPressed: () => _shareFile(file),
            icon: const Icon(Icons.share, size: 20),
            color: AppConstants.primaryColor,
            tooltip: 'Share',
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
          ),
          IconButton(
            onPressed: () => _deleteFile(file),
            icon: const Icon(Icons.delete_outline, size: 20),
            color: AppConstants.errorColor,
            tooltip: 'Delete',
            padding: EdgeInsets.zero,
            constraints: const BoxConstraints(minWidth: 36, minHeight: 36),
          ),
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
}

class SavedAttendanceFile {
  final String fileName;
  final String filePath;
  final DateTime createdAt;
  final String displayName;

  SavedAttendanceFile({
    required this.fileName,
    required this.filePath,
    required this.createdAt,
    required this.displayName,
  });
}
