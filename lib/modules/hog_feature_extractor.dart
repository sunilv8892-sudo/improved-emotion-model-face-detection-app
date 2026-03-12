import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// HOG (Histogram of Oriented Gradients) Feature Extractor
///
/// Exact reimplementation of scikit-image's `skimage.feature.hog` with:
///   orientations=8, pixels_per_cell=(32,32), cells_per_block=(2,2),
///   block_norm='L2-Hys' on a 256×256 grayscale image.
///
/// These parameters match the original training notebook (Mood Prediction.ipynb):
///   ```python
///   hog(gray, orientations=8, pixels_per_cell=(32, 32),
///       cells_per_block=(2, 2), visualize=False)
///   ```
///
/// Output: 1568-dimensional descriptor (7×7 blocks × 2×2 cells × 8 bins).
class HogFeatureExtractor {
  /// Number of orientation bins (unsigned gradients, 0°–180°).
  static const int numBins = 8;

  /// Pixels per cell side (matches training: 32×32).
  static const int cellSize = 32;

  /// Cells per block side (2×2 standard).
  static const int blockSize = 2;

  /// Block stride in cells.
  static const int blockStride = 1;

  /// Image resized to 256×256 before HOG (matches training).
  static const int targetSize = 256;

  /// Degrees per orientation bin.
  static const double _degPerBin = 180.0 / numBins; // 22.5°

  /// skimage uses eps² in L2-Hys: eps=1e-5, eps²=1e-10.
  static const double _epsSquared = 1e-10;

  /// Inverse of cell area for averaging (1 / (32*32) = 1/1024).
  static const double _invCellArea = 1.0 / (cellSize * cellSize);

  /// Extract HOG features from JPEG/PNG face image bytes.
  Float32List extractFromBytes(Uint8List faceImageBytes) {
    final decoded = img.decodeImage(faceImageBytes);
    if (decoded == null) {
      throw Exception('HogFeatureExtractor: failed to decode image');
    }
    return extractFromImage(decoded);
  }

  /// Extract HOG features from a decoded [img.Image].
  Float32List extractFromImage(img.Image image) {
    final gray = _toGrayscaleResized(image);
    final gradMag = Float64List(targetSize * targetSize);
    final gradOri = Float64List(targetSize * targetSize);
    _computeGradients(gray, gradMag, gradOri);

    final numCells = targetSize ~/ cellSize; // 8
    final cellHist = List.generate(
      numCells,
      (_) => List.generate(numCells, (_) => Float64List(numBins)),
    );
    _buildCellHistograms(gradMag, gradOri, cellHist, numCells);
    return _blockNormalize(cellHist, numCells);
  }

  /// Convert to grayscale and resize to 256×256.
  /// Uses standard luminance: 0.299R + 0.587G + 0.114B (same as cv2.COLOR_RGB2GRAY).
  Float64List _toGrayscaleResized(img.Image image) {
    final resized =
        img.copyResize(image, width: targetSize, height: targetSize);
    final gray = Float64List(targetSize * targetSize);
    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final pixel = resized.getPixel(x, y);
        gray[y * targetSize + x] =
            0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
      }
    }
    return gray;
  }

  /// Compute gradients using central differences with 0 at borders.
  /// Matches skimage's `_hog_channel_gradient`:
  ///   g_row[1:-1,:] = image[2:,:] - image[:-2,:]  (0 at row 0 and last row)
  ///   g_col[:,1:-1] = image[:,2:] - image[:,:-2]   (0 at col 0 and last col)
  void _computeGradients(
    Float64List gray,
    Float64List magnitude,
    Float64List orientation,
  ) {
    for (int y = 0; y < targetSize; y++) {
      for (int x = 0; x < targetSize; x++) {
        final idx = y * targetSize + x;

        // Horizontal gradient: 0 at first/last column, central difference inside
        final double gx = (x > 0 && x < targetSize - 1)
            ? gray[y * targetSize + (x + 1)] - gray[y * targetSize + (x - 1)]
            : 0.0;

        // Vertical gradient: 0 at first/last row, central difference inside
        final double gy = (y > 0 && y < targetSize - 1)
            ? gray[(y + 1) * targetSize + x] - gray[(y - 1) * targetSize + x]
            : 0.0;

        magnitude[idx] = math.sqrt(gx * gx + gy * gy);

        // Unsigned orientation in [0, 180) degrees.
        // Matches: np.rad2deg(np.arctan2(g_row, g_col)) % 180
        double angle = math.atan2(gy, gx) * (180.0 / math.pi) % 180.0;
        if (angle < 0) angle += 180.0; // Dart % can return negative
        orientation[idx] = angle;
      }
    }
  }

  /// Build cell histograms using hard assignment (no interpolation).
  /// Each pixel votes into exactly one cell and one orientation bin.
  /// Cell histogram values are averaged by cell area (cellSize²).
  void _buildCellHistograms(
    Float64List magnitude,
    Float64List orientation,
    List<List<Float64List>> cellHist,
    int numCells,
  ) {
    for (int y = 0; y < targetSize; y++) {
      final int cy = y ~/ cellSize;
      final int cellRow = cy < numCells ? cy : numCells - 1;
      for (int x = 0; x < targetSize; x++) {
        final int cx = x ~/ cellSize;
        final int cellCol = cx < numCells ? cx : numCells - 1;
        final idx = y * targetSize + x;
        final mag = magnitude[idx];
        final ori = orientation[idx];

        int bin = (ori / _degPerBin).toInt();
        if (bin >= numBins) bin = numBins - 1;

        cellHist[cellRow][cellCol][bin] += mag;
      }
    }

    // Divide by cell area (matches skimage: total / (cell_rows * cell_columns))
    for (int cy = 0; cy < numCells; cy++) {
      for (int cx = 0; cx < numCells; cx++) {
        for (int b = 0; b < numBins; b++) {
          cellHist[cy][cx][b] *= _invCellArea;
        }
      }
    }
  }

  /// Block normalization using L2-Hys (L2 → clip(0.2) → renormalize).
  /// Uses eps²=1e-10 in the norm denominator, matching skimage exactly:
  ///   out = block / sqrt(sum(block²) + eps²)
  ///   out = min(out, 0.2)
  ///   out = out / sqrt(sum(out²) + eps²)
  Float32List _blockNormalize(
    List<List<Float64List>> cellHist,
    int numCells,
  ) {
    final numBlocks = numCells - blockSize + 1; // 7
    final featureLength = numBlocks * numBlocks * blockSize * blockSize * numBins;
    final descriptor = Float32List(featureLength);
    int offset = 0;

    for (int by = 0; by < numBlocks; by++) {
      for (int bx = 0; bx < numBlocks; bx++) {
        // Collect block vector
        final blockVec = Float64List(blockSize * blockSize * numBins);
        int vi = 0;
        for (int dy = 0; dy < blockSize; dy++) {
          for (int dx = 0; dx < blockSize; dx++) {
            final hist = cellHist[by + dy][bx + dx];
            for (int b = 0; b < numBins; b++) {
              blockVec[vi++] = hist[b];
            }
          }
        }

        // L2-Hys: first L2 normalize with eps²
        double sumSq = 0.0;
        for (int i = 0; i < blockVec.length; i++) {
          sumSq += blockVec[i] * blockVec[i];
        }
        double norm = math.sqrt(sumSq + _epsSquared);
        for (int i = 0; i < blockVec.length; i++) {
          blockVec[i] = blockVec[i] / norm;
          if (blockVec[i] > 0.2) blockVec[i] = 0.2;
        }

        // Re-normalize after clipping
        double sumSq2 = 0.0;
        for (int i = 0; i < blockVec.length; i++) {
          sumSq2 += blockVec[i] * blockVec[i];
        }
        double norm2 = math.sqrt(sumSq2 + _epsSquared);
        for (int i = 0; i < blockVec.length; i++) {
          descriptor[offset++] = blockVec[i] / norm2;
        }
      }
    }

    return descriptor;
  }

  /// Total length of the HOG descriptor: 1568 for 256×256 / 32px cells.
  static int get descriptorLength {
    final numCells = targetSize ~/ cellSize;
    final numBlocks = numCells - blockSize + 1;
    return numBlocks * numBlocks * blockSize * blockSize * numBins;
  }
}
