import 'dart:convert';

import 'package:flutter/services.dart';

class EmotionModelParameters {
  static const String assetPath = 'assets/models/emotion_runtime_params.json';

  final List<String> labels;
  final int efficientnetFeatureCount;
  final int hogFeatureCount;
  final int poseFeatureCount;
  final EmotionMinMaxScaler scaler;
  final EmotionLdaTransform lda;
  final EmotionSvmParameters svm;

  const EmotionModelParameters({
    required this.labels,
    required this.efficientnetFeatureCount,
    required this.hogFeatureCount,
    required this.poseFeatureCount,
    required this.scaler,
    required this.lda,
    required this.svm,
  });

  int get totalFeatureCount =>
      efficientnetFeatureCount + hogFeatureCount + poseFeatureCount;

  static Future<EmotionModelParameters> load() async {
    final rawJson = await rootBundle.loadString(assetPath);
    final data = jsonDecode(rawJson) as Map<String, dynamic>;
    final featureLayout = data['feature_layout'] as Map<String, dynamic>;

    return EmotionModelParameters(
      labels: _toStringList(data['labels']),
      efficientnetFeatureCount: (featureLayout['efficientnet_dim'] as num).toInt(),
      hogFeatureCount: (featureLayout['hog_dim'] as num).toInt(),
      poseFeatureCount: (featureLayout['pose_dim'] as num).toInt(),
      scaler: EmotionMinMaxScaler.fromJson(data['scaler'] as Map<String, dynamic>),
      lda: EmotionLdaTransform.fromJson(data['lda'] as Map<String, dynamic>),
      svm: EmotionSvmParameters.fromJson(data['svm'] as Map<String, dynamic>),
    );
  }

  static List<String> _toStringList(dynamic value) =>
      (value as List).map((item) => item.toString()).toList();

  static List<double> toDoubleList(dynamic value) =>
      (value as List).map((item) => (item as num).toDouble()).toList();

  static List<List<double>> to2dDoubleList(dynamic value) =>
      (value as List)
          .map((row) => (row as List).map((item) => (item as num).toDouble()).toList())
          .toList();
}

class EmotionMinMaxScaler {
  final List<double> minValues;
  final List<double> scaleValues;

  const EmotionMinMaxScaler({
    required this.minValues,
    required this.scaleValues,
  });

  factory EmotionMinMaxScaler.fromJson(Map<String, dynamic> json) {
    return EmotionMinMaxScaler(
      minValues: EmotionModelParameters.toDoubleList(json['min']),
      scaleValues: EmotionModelParameters.toDoubleList(json['scale']),
    );
  }

  List<double> transform(List<double> features) {
    if (features.length != minValues.length || features.length != scaleValues.length) {
      throw Exception(
        'Scaler input mismatch: expected ${minValues.length}, got ${features.length}',
      );
    }

    return List<double>.generate(features.length, (index) {
      return features[index] * scaleValues[index] + minValues[index];
    }, growable: false);
  }
}

class EmotionLdaTransform {
  final List<double> xbar;
  final List<List<double>> scalings;
  final int outputDimension;

  const EmotionLdaTransform({
    required this.xbar,
    required this.scalings,
    required this.outputDimension,
  });

  factory EmotionLdaTransform.fromJson(Map<String, dynamic> json) {
    return EmotionLdaTransform(
      xbar: EmotionModelParameters.toDoubleList(json['xbar']),
      scalings: EmotionModelParameters.to2dDoubleList(json['scalings']),
      outputDimension: (json['output_dim'] as num).toInt(),
    );
  }

  List<double> transform(List<double> features) {
    if (features.length != xbar.length || scalings.length != xbar.length) {
      throw Exception(
        'LDA input mismatch: expected ${xbar.length}, got ${features.length}',
      );
    }

    final output = List<double>.filled(outputDimension, 0.0, growable: false);
    for (int component = 0; component < outputDimension; component++) {
      double sum = 0.0;
      for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
        final centered = features[featureIndex] - xbar[featureIndex];
        sum += centered * scalings[featureIndex][component];
      }
      output[component] = sum;
    }
    return output;
  }
}

class EmotionSvmParameters {
  final double gamma;
  final List<EmotionBinarySvmModel> binaryModels;
  final Map<String, double> classBiases;

  const EmotionSvmParameters({
    required this.gamma,
    required this.binaryModels,
    required this.classBiases,
  });

  factory EmotionSvmParameters.fromJson(Map<String, dynamic> json) {
    final biasesRaw = json['class_biases'] as Map<String, dynamic>?;
    final biases = biasesRaw != null
        ? biasesRaw.map((k, v) => MapEntry(k, (v as num).toDouble()))
        : <String, double>{};

    return EmotionSvmParameters(
      gamma: (json['gamma'] as num).toDouble(),
      binaryModels: (json['binary_models'] as List)
          .map(
            (model) => EmotionBinarySvmModel.fromJson(
              model as Map<String, dynamic>,
            ),
          )
          .toList(),
      classBiases: biases,
    );
  }
}

class EmotionBinarySvmModel {
  final String label;
  final List<List<double>> supportVectors;
  final List<double> dualCoefficients;
  final double intercept;

  const EmotionBinarySvmModel({
    required this.label,
    required this.supportVectors,
    required this.dualCoefficients,
    required this.intercept,
  });

  factory EmotionBinarySvmModel.fromJson(Map<String, dynamic> json) {
    return EmotionBinarySvmModel(
      label: json['label'].toString(),
      supportVectors: EmotionModelParameters.to2dDoubleList(json['support_vectors']),
      dualCoefficients: EmotionModelParameters.toDoubleList(json['dual_coefficients']),
      intercept: (json['intercept'] as num).toDouble(),
    );
  }
}