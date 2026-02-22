// Quick test to understand YOLO model output shape
// This helps debug why boxes aren't appearing

import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';

void main() async {
  print('Loading YOLO model...');

  try {
    final interpreter = await Interpreter.fromAsset(
      'assets/models/model.tflite',
    );

    print('✅ Model loaded successfully');
    print('Input tensors: ${interpreter.getInputTensors()}');
    print('Output tensors: ${interpreter.getOutputTensors()}');

    // Check input shape
    final inputTensor = interpreter.getInputTensor(0);
    print('\n📥 INPUT:');
    print('  Shape: ${inputTensor.shape}');
    print('  Type: ${inputTensor.type}');

    // Check output shape
    final outputTensor = interpreter.getOutputTensor(0);
    print('\n📤 OUTPUT:');
    print('  Shape: ${outputTensor.shape}');
    print('  Type: ${outputTensor.type}');

    // Create dummy input
    final inputData = Float32List(1 * 320 * 320 * 3);
    print(
      '\nDummy input size: ${inputData.length} (should be ${1 * 320 * 320 * 3})',
    );

    // Run inference
    final input = inputData.reshape(inputTensor.shape);
    final outputSize = outputTensor.shape.fold<int>(
      1,
      (value, element) => value * element,
    );
    final output = List.filled(outputSize, 0.0);

    print('\nRunning inference...');
    interpreter.run(input, output);

    final outputReshaped = output.reshape(outputTensor.shape);
    print('✅ Inference complete');
    print('  Output shape: ${outputReshaped.shape}');

    interpreter.close();
  } catch (e) {
    print('❌ Error: $e');
  }
}
