import 'package:flutter/material.dart';
import 'utils/constants.dart';
import 'database/database_manager.dart';
import 'screens/home_screen.dart';
import 'screens/enrollment_screen.dart';
import 'screens/attendance_screen_stub.dart';
import 'screens/database_screen.dart';
import 'screens/export_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/expression_detection_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize database
  final dbManager = DatabaseManager();
  await dbManager.database; // Initialize database on app startup

  runApp(FaceRecognitionApp(dbManager: dbManager));
}

/// Main Application Widget
class FaceRecognitionApp extends StatefulWidget {
  final DatabaseManager dbManager;

  const FaceRecognitionApp({super.key, required this.dbManager});

  @override
  State<FaceRecognitionApp> createState() => _FaceRecognitionAppState();
}

class _FaceRecognitionAppState extends State<FaceRecognitionApp> {
  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppConstants.appName,
      theme: AppTheme.lightTheme,
      debugShowCheckedModeBanner: false,
      home: const HomeScreen(),
      routes: {
        AppConstants.routeHome: (context) => const HomeScreen(),
        AppConstants.routeEnroll: (context) => const EnrollmentScreen(),
        AppConstants.routeAttendance: (context) => const AttendanceScreen(),
        AppConstants.routeDatabase: (context) => const DatabaseScreen(),
        AppConstants.routeExport: (context) => const ExportScreen(),
        AppConstants.routeSettings: (context) => const SettingsScreen(),
        AppConstants.routeExpressionDetection: (context) => const ExpressionDetectionScreen(),
      },
    );
  }

  @override
  void dispose() {
    widget.dbManager.close();
    super.dispose();
  }
}
