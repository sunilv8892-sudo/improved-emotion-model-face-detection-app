import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppColors {
  AppColors._();

  // Primary palette (Green / Blue)
  static const Color primary = Color(0xFF2E7D32); // Green 700
  static const Color primaryVariant = Color(0xFF1B5E20); // Green 900
  static const Color secondary = Color(0xFF0288D1); // Light Blue 700
  static const Color accent = Color(0xFF26C6DA); // Cyan 400 (accent)

  // UI neutrals
  static const Color background = Color(0xFFF5F7FA);
  static const Color surface = Colors.white;
  static const Color success = Color(0xFF2E7D32);
  static const Color warning = Color(0xFFF57C00);
  static const Color error = Color(0xFFD32F2F);
}

class AppTheme {
  AppTheme._();

  static final ColorScheme _lightScheme = ColorScheme(
    brightness: Brightness.light,
    primary: AppColors.primary,
    onPrimary: Colors.white,
    secondary: AppColors.secondary,
    onSecondary: Colors.white,
    error: AppColors.error,
    onError: Colors.white,
    background: AppColors.background,
    onBackground: Colors.black87,
    surface: AppColors.surface,
    onSurface: Colors.black87,
  );

  static ThemeData get lightTheme {
    final base = ThemeData.from(colorScheme: _lightScheme);
    return base.copyWith(
      useMaterial3: false,
      textTheme: GoogleFonts.interTextTheme(base.textTheme),
      primaryColor: _lightScheme.primary,
      appBarTheme: AppBarTheme(
        backgroundColor: _lightScheme.primary.withValues(alpha: 0.95),
        foregroundColor: _lightScheme.onPrimary,
        elevation: 2,
        centerTitle: true,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: _lightScheme.primary,
          foregroundColor: _lightScheme.onPrimary,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          elevation: 3,
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: _lightScheme.primary,
          side: BorderSide(color: _lightScheme.primary.withValues(alpha: 0.85)),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        ),
      ),
      floatingActionButtonTheme: FloatingActionButtonThemeData(
        backgroundColor: AppColors.accent,
        foregroundColor: Colors.white,
      ),
      snackBarTheme: SnackBarThemeData(
        backgroundColor: _lightScheme.primary,
        contentTextStyle: GoogleFonts.inter(color: Colors.white),
      ),
    );
  }
}
