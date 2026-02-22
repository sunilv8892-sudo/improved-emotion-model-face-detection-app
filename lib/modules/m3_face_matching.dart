import '../models/embedding_model.dart';
import '../models/match_result_model.dart';

/// M3: Face Matching Module
/// Identifies who a face belongs to by comparing embeddings
/// Uses K-Nearest Neighbors with Euclidean distance (NOT a neural network)
class FaceMatchingModule {
  static const String modelName = 'Face Matcher (KNN - Euclidean)';
  // Threshold here is on converted similarity (1 / (1 + distance)).
  // Typical values will be in (0, 1], adjust in settings if needed.
  static const double defaultSimilarityThreshold = 0.60;
  static const int knnK = 1; // Use 1-NN for single match

  /// Match incoming embedding against database embeddings
  /// Returns the best match with highest similarity
  MatchResult matchFace(
    List<double> incomingEmbedding,
    List<FaceEmbedding> databaseEmbeddings, {
    double similarityThreshold = defaultSimilarityThreshold,
  }) {
    if (databaseEmbeddings.isEmpty) {
      return MatchResult(identityType: 'unknown', similarity: 0);
    }

    // Find nearest neighbor by Euclidean distance
    double bestDistance = double.infinity;
    FaceEmbedding? bestMatch;

    for (final dbEmbedding in databaseEmbeddings) {
      final dist = euclideanDistance(incomingEmbedding, dbEmbedding.vector);
      if (dist < bestDistance) {
        bestDistance = dist;
        bestMatch = dbEmbedding;
      }
    }

    // Convert distance to a similarity-like score in (0, 1]: sim = 1 / (1 + dist)
    final bestSimilarity = bestDistance.isFinite ? 1.0 / (1.0 + bestDistance) : 0.0;

    if (bestMatch != null && bestSimilarity >= similarityThreshold) {
      return MatchResult(
        identityType: 'known',
        studentId: bestMatch.studentId,
        similarity: bestSimilarity,
      );
    }

    return MatchResult(identityType: 'unknown', similarity: bestSimilarity);
  }

  // (cosineSimilarity removed — using Euclidean KNN instead)

  /// Calculate Euclidean distance between two vectors
  double euclideanDistance(List<double> vec1, List<double> vec2) {
    if (vec1.isEmpty || vec2.isEmpty || vec1.length != vec2.length) {
      return double.infinity;
    }
    final dim = vec1.length;

    double sum = 0;
    for (int i = 0; i < dim; i++) {
      final diff = vec1[i] - vec2[i];
      sum += diff * diff;
    }

    return _sqrt(sum);
  }

  /// K-Nearest Neighbors matching (for more robust identification)
  List<MatchResult> knnMatch(
    List<double> incomingEmbedding,
    List<FaceEmbedding> databaseEmbeddings, {
    int k = knnK,
    double similarityThreshold = defaultSimilarityThreshold,
  }) {
    if (databaseEmbeddings.isEmpty) {
      return [MatchResult(identityType: 'unknown', similarity: 0)];
    }

    // Calculate distances for all embeddings
    final distances = databaseEmbeddings.map((emb) {
      final d = euclideanDistance(incomingEmbedding, emb.vector);
      return MapEntry(emb, d);
    }).toList();

    // Sort by distance (ascending)
    distances.sort((a, b) => a.value.compareTo(b.value));

    // Take top K and convert distances to similarity scores
    final topMatches = distances.take(k).where((entry) => true).map((entry) {
      final sim = entry.value.isFinite ? 1.0 / (1.0 + entry.value) : 0.0;
      return MapEntry(entry.key, sim);
    }).where((me) => me.value >= similarityThreshold).map((me) {
      return MatchResult(
        identityType: 'known',
        studentId: me.key.studentId,
        similarity: me.value,
      );
    }).toList();

    if (topMatches.isNotEmpty) return topMatches;

    // If no top matches above threshold, return best candidate (converted similarity)
    final best = distances.first;
    final bestSim = best.value.isFinite ? 1.0 / (1.0 + best.value) : 0.0;
    return [MatchResult(identityType: 'unknown', similarity: bestSim)];
  }

  /// Get matching statistics
  Map<String, dynamic> getMatchingStats(
    List<double> incomingEmbedding,
    List<FaceEmbedding> databaseEmbeddings,
  ) {
    if (databaseEmbeddings.isEmpty) {
      return {
        'total_embeddings': 0,
        'best_similarity': 0,
        'threshold': defaultSimilarityThreshold,
      };
    }

    double bestSim = 0;
    double worstSim = 1;

    for (final emb in databaseEmbeddings) {
      final dist = euclideanDistance(incomingEmbedding, emb.vector);
      final sim = dist.isFinite ? 1.0 / (1.0 + dist) : 0.0;
      if (sim > bestSim) bestSim = sim;
      if (sim < worstSim) worstSim = sim;
    }

    return {
      'total_embeddings': databaseEmbeddings.length,
      'best_similarity': bestSim.toStringAsFixed(4),
      'worst_similarity': worstSim.toStringAsFixed(4),
      'threshold': defaultSimilarityThreshold,
    };
  }

  /// Simple square root helper
  double _sqrt(double x) {
    if (x == 0) return 0;
    double z = x;
    double result = 0;
    while ((z - result).abs() > 1e-7) {
      result = z;
      z = 0.5 * (z + x / z);
    }
    return z;
  }
}
