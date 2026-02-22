import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Database setup with SQLite vector extension support
class DatabaseConnection {
  static DatabaseConnection? _instance;

  DatabaseConnection._();

  static DatabaseConnection get instance {
    _instance ??= DatabaseConnection._();
    return _instance!;
  }

  QueryExecutor openConnection(String dbName) {
    return LazyDatabase(() async {
      final documentsDir = await getApplicationDocumentsDirectory();
      final dbPath = p.join(documentsDir.path, dbName);

      // Use Drift's NativeDatabase with file path
      return NativeDatabase.createInBackground(File(dbPath));
    });
  }
}
