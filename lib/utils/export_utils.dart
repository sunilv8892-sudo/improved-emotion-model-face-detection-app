import 'dart:io';
import 'package:path_provider/path_provider.dart';

/// Single source of truth for the export directory.
/// Every screen that reads or writes CSV/backup files MUST use this
/// so they all point at the same folder.
Future<Directory> getExportDirectory() async {
  Directory? baseDir;

  if (Platform.isAndroid) {
    // getExternalStorageDirectories returns the app-scoped external dir
    // (/storage/emulated/0/Android/data/<pkg>/files/downloads/).
    // This does NOT require runtime storage permission on Android 11+.
    try {
      final externals = await getExternalStorageDirectories(
        type: StorageDirectory.downloads,
      );
      if (externals != null && externals.isNotEmpty) {
        baseDir = Directory(externals.first.path);
      }
    } catch (_) {}
  }

  // Fallback for iOS / desktop / any failure
  baseDir ??= await getApplicationDocumentsDirectory();

  final dir = Directory('${baseDir.path}/FaceAttendanceExports');
  await dir.create(recursive: true);
  return dir;
}
