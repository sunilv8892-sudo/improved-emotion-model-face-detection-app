import 'dart:math' as math;

import 'emotion_model_parameters.dart';

class EmotionSvmPrediction {
  final String label;
  final double confidence;
  final Map<String, double> probabilities;
  final Map<String, double> rawScores;

  const EmotionSvmPrediction({
    required this.label,
    required this.confidence,
    required this.probabilities,
    required this.rawScores,
  });
}

class EmotionSvmClassifier {
  final EmotionSvmParameters parameters;

  const EmotionSvmClassifier(this.parameters);

  EmotionSvmPrediction predict(List<double> ldaFeatures) {
    final scores = <String, double>{};

    for (final model in parameters.binaryModels) {
      if (model.supportVectors.length != model.dualCoefficients.length) {
        throw Exception(
          'Binary SVM coefficient mismatch for ${model.label}: '
          '${model.supportVectors.length} support vectors but '
          '${model.dualCoefficients.length} coefficients',
        );
      }

      double score = model.intercept;
      for (int i = 0; i < model.supportVectors.length; i++) {
        final supportVector = model.supportVectors[i];
        double squaredDistance = 0.0;
        for (int j = 0; j < ldaFeatures.length; j++) {
          final delta = ldaFeatures[j] - supportVector[j];
          squaredDistance += delta * delta;
        }

        final kernel = math.exp(-parameters.gamma * squaredDistance);
        score += model.dualCoefficients[i] * kernel;
      }
      scores[model.label] = score;
    }

    // Check max raw score BEFORE biases — if no SVM is confident about any
    // specific emotion (all decision-function values negative), the face is
    // most likely neutral (absence of strong emotion = neutral).
    final maxRawScore = scores.values.reduce(math.max);

    // Apply class biases (e.g. +0.5 for Neutral, -0.2 for Angry)
    for (final entry in parameters.classBiases.entries) {
      if (scores.containsKey(entry.key)) {
        scores[entry.key] = scores[entry.key]! + entry.value;
      }
    }

    // "No confident emotion → Neutral" rule:
    // When every binary SVM outputs a negative decision value, none of them
    // claim "this IS my emotion."  No detected emotion = Neutral by definition.
    // This handles the domain gap between FER training images and live camera.
    const double neutralFallbackThreshold = -0.05;
    if (maxRawScore < neutralFallbackThreshold && scores.containsKey('Neutral')) {
      final biasedMax = scores.values.reduce(math.max);
      scores['Neutral'] = biasedMax + 0.5;
    }

    final probabilities = _softmax(scores);
    final bestEntry = probabilities.entries.reduce(
      (best, current) => current.value > best.value ? current : best,
    );

    return EmotionSvmPrediction(
      label: bestEntry.key,
      confidence: bestEntry.value,
      probabilities: probabilities,
      rawScores: scores,
    );
  }

  Map<String, double> _softmax(Map<String, double> scores) {
    // Use temperature < 1 to sharpen the distribution (more confident predictions)
    const double temperature = 0.5;
    final scaledScores = scores.map((label, score) => MapEntry(label, score / temperature));
    final maxValue = scaledScores.values.reduce(math.max);
    final exponentials = <String, double>{};
    double sum = 0.0;

    scaledScores.forEach((label, score) {
      final value = math.exp(score - maxValue);
      exponentials[label] = value;
      sum += value;
    });

    return exponentials.map((label, value) => MapEntry(label, value / sum));
  }
}