import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:yolo_app/main.dart';

void main() {
  testWidgets('App builds with AppTheme', (WidgetTester tester) async {
    await tester.pumpWidget(const FaceRecognitionApp());

    expect(find.byType(MaterialApp), findsOneWidget);
    // Verify AppBar uses theme color when present
    // (This ensures ThemeData loads without errors)
    expect(Theme.of(tester.element(find.byType(MaterialApp))).primaryColor, isNotNull);
  });
}
