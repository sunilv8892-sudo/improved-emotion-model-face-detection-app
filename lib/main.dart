import 'package:flutter/material.dart';
import 'database/database_manager.dart';
import 'screens/home_screen.dart';
import 'screens/enrollment_screen.dart';
import 'screens/attendance_prep_screen.dart';
import 'screens/database_screen.dart';
import 'screens/export_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/expression_detection_screen.dart';
import 'utils/constants.dart';

late DatabaseManager _dbManager;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Initialize database
  _dbManager = DatabaseManager();
  await _dbManager.database;

  runApp(const FaceRecognitionApp());
}

class FaceRecognitionApp extends StatelessWidget {
  const FaceRecognitionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: AppConstants.appName,
      theme: AppTheme.lightTheme,
      builder: (context, child) {
        return child ?? const SizedBox.shrink();
      },
      initialRoute: AppConstants.routeHome,
      routes: {
        AppConstants.routeHome: (context) => const HomeScreen(),
        AppConstants.routeEnroll: (context) => const EnrollmentScreen(),
        AppConstants.routeAttendance: (context) => const AttendancePrepScreen(),
        AppConstants.routeDatabase: (context) => const DatabaseScreen(),
        AppConstants.routeExport: (context) => const ExportScreen(),
        AppConstants.routeSettings: (context) => const SettingsScreen(),
        AppConstants.routeExpressionDetection: (context) => const ExpressionDetectionScreen(),
      },
    );
  }
}
