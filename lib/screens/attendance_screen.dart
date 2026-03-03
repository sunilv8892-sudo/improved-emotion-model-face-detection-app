import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
// removed: unused import 'package:path_provider/path_provider.dart'
import '../utils/export_utils.dart';
import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:screen_brightness/screen_brightness.dart';
import '../database/database_manager.dart';
import '../models/face_detection_model.dart';
import '../models/student_model.dart';
import '../models/attendance_model.dart';
import '../models/subject_model.dart';
import '../modules/m1_face_detection.dart' as face_detection_module;
import '../modules/m2_face_embedding.dart';
import '../utils/constants.dart';
import '../widgets/animated_background.dart';

class _KnnSample {
  final int studentId;
  final List<double> vector;

  const _KnnSample({required this.studentId, required this.vector});
}

class _KnnNeighbor {
  final int studentId;
  final double distance;
  final double similarity;

  const _KnnNeighbor({
    required this.studentId,
    required this.distance,
    required this.similarity,
  });
}

class AttendanceScreen extends StatefulWidget {
  final String teacherName;
  final Subject subject;

  const AttendanceScreen({
    super.key,
    required this.teacherName,
    required this.subject,
  });

  @override
  State<AttendanceScreen> createState() => _AttendanceScreenState();
}

class _AttendanceScreenState extends State<AttendanceScreen> {
  CameraController? _controller;
  late face_detection_module.FaceDetectionModule _faceDetector;
  late FaceEmbeddingModule _faceEmbedder;
  late FlutterTts _flutterTts;
  List<CameraDescription> _availableCameras = [];
  late CameraDescription _currentCamera;
  late DatabaseManager _dbManager;
  bool _isBackFlashOn = false;
  bool _isFrontLightOn = false;
  double? _previousAppBrightness;
  bool _isBrightnessBoostActive = false;

  List<Student> _enrolledStudents = [];
  final Map<int, List<List<double>>> _studentEmbeddings = {};
  bool _isProcessing = false;
  bool _isScanning = false;
  final Map<int, AttendanceStatus> _attendanceStatus = {};
  DateTime? _attendanceDate;
  static const double _similarityThreshold = 0.75;
  final Map<int, DateTime> _lastDetectionTime =
      {}; // Prevent duplicate detections
  static const Duration _detectionCooldown = Duration(seconds: 1);
  static const int _knnK = 5;
  final List<_KnnSample> _knnTrainingSet = [];

  // Per-student consecutive detection tracking (supports multiple faces)
  final Map<int, int> _consecutiveDetectionsMap = {};
    static const int _requiredConsecutiveDetections =
      2; // Require 2 consecutive matches
  int? _lastSingleFaceMatchId; // Prevents alternating identity false positives

  // Face overlay
  final List<DetectedFace> _overlayFaces = [];
  final List<String> _overlayNames = [];
  final List<Color> _overlayColors = [];
  final List<String> _overlayExpressions = [];
  Size? _imageSize;
  Timer? _overlayTimer;

  // Expression tracking per student
  final Map<int, String> _studentExpressions = {};

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  @override
  void dispose() {
    _isScanning = false;
    _isProcessing = false;
    _disableBrightnessBoost();
    _controller?.dispose();
    // NOTE: _faceDetector and _faceEmbedder are app-scoped singletons;
    // do NOT dispose them here — they are reused across screen instances.
    _overlayTimer?.cancel();
    super.dispose();
  }

  Future<void> _initialize() async {
    try {
      debugPrint('📊 Using fixed matching threshold: $_similarityThreshold');

      _dbManager = DatabaseManager();
      await _dbManager.database;
      // Initialize text-to-speech
      _flutterTts = FlutterTts();
      await _flutterTts.setLanguage('en-US');
      await _flutterTts.setSpeechRate(0.5);
      await _flutterTts.setVolume(1.0);
      await _flutterTts.setPitch(1.0);
      debugPrint('🔊 Text-to-Speech initialized');
      // Initialize modules
      _faceDetector = face_detection_module.FaceDetectionModule();
      await _faceDetector.initialize();

      _faceEmbedder = FaceEmbeddingModule();
      await _faceEmbedder.initialize();

      // Load enrolled students and their embeddings
      _enrolledStudents = await _dbManager.getAllStudents();
      debugPrint('📚 Loaded ${_enrolledStudents.length} enrolled students');

      for (final student in _enrolledStudents) {
        final embeddings = await _dbManager.getEmbeddingsForStudent(
          student.id!,
        );
        final validEmbeddings = embeddings
            .map((e) => e.vector)
            .where((v) => v.length == FaceEmbeddingModule.embeddingDimension)
            .toList();
        _studentEmbeddings[student.id!] = validEmbeddings;
        final dropped = embeddings.length - validEmbeddings.length;
        debugPrint('   ${student.name}: ${validEmbeddings.length} embeddings${dropped > 0 ? ' (dropped $dropped invalid)' : ''}');
      }
      _rebuildKnnTrainingSet();

      _attendanceDate = DateTime.now();
      // Normalize to midnight (no time component) for consistent date matching
      _attendanceDate = DateTime(
        _attendanceDate!.year,
        _attendanceDate!.month,
        _attendanceDate!.day,
      );
      await _initCamera();
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Init error: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameraStatus = await Permission.camera.request();
      if (!cameraStatus.isGranted) {
        debugPrint('❌ Camera permission denied');
        return;
      }
      _availableCameras = await availableCameras();
      if (_availableCameras.isEmpty) return;

      _currentCamera = _availableCameras.first;
      final preferredCamera = _availableCameras.firstWhere(
        (camera) => camera.lensDirection == CameraLensDirection.front,
        orElse: () => _availableCameras.first,
      );
      await _initCameraFor(preferredCamera);
    } catch (e) {
      debugPrint('Camera error: $e');
    }
  }

  Future<void> _initCameraFor(CameraDescription camera) async {
    try {
      await _controller?.dispose();
      _currentCamera = camera;
      _controller = CameraController(
        camera,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();
      await _controller!.setFlashMode(FlashMode.off);
      _isBackFlashOn = false;
      _isFrontLightOn = false;
      await _disableBrightnessBoost();
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Init camera error: $e');
    }
  }

  Future<void> _switchCamera() async {
    if (_availableCameras.length < 2) return;

    final nextCamera = _availableCameras.lastWhere(
      (camera) => camera.lensDirection != _currentCamera.lensDirection,
      orElse: () => _availableCameras.first,
    );

    _currentCamera = nextCamera;
    await _initCameraFor(nextCamera);
    debugPrint('Switched to ${nextCamera.lensDirection.toString()} camera');
  }

  bool get _isFrontCamera =>
      _currentCamera.lensDirection == CameraLensDirection.front;

  Future<void> _enableBrightnessBoost() async {
    if (_isBrightnessBoostActive) return;
    try {
      _previousAppBrightness = await ScreenBrightness.instance.application;
      await ScreenBrightness.instance.setApplicationScreenBrightness(1.0);
      _isBrightnessBoostActive = true;
    } catch (e) {
      debugPrint('Brightness enable error: $e');
    }
  }

  Future<void> _disableBrightnessBoost() async {
    if (!_isBrightnessBoostActive) return;
    try {
      if (_previousAppBrightness != null) {
        await ScreenBrightness.instance.setApplicationScreenBrightness(
          _previousAppBrightness!,
        );
      } else {
        await ScreenBrightness.instance.resetApplicationScreenBrightness();
      }
    } catch (e) {
      debugPrint('Brightness restore error: $e');
    } finally {
      _isBrightnessBoostActive = false;
      _previousAppBrightness = null;
    }
  }

  Future<void> _toggleCameraLight() async {
    if (_controller == null || !_controller!.value.isInitialized) return;

    if (_isFrontCamera) {
      final nextFrontLightState = !_isFrontLightOn;
      if (nextFrontLightState) {
        await _enableBrightnessBoost();
      } else {
        await _disableBrightnessBoost();
      }
      if (mounted) {
        setState(() {
          _isFrontLightOn = nextFrontLightState;
        });
      }
      return;
    }

    final nextFlashState = !_isBackFlashOn;
    try {
      await _controller!.setFlashMode(
        nextFlashState ? FlashMode.torch : FlashMode.off,
      );
      if (mounted) {
        setState(() {
          _isBackFlashOn = nextFlashState;
        });
      }
    } catch (e) {
      debugPrint('Flash mode error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Flash is not supported on this camera')),
        );
      }
    }
  }

  Future<void> _scanFace() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isProcessing) return;

    _isProcessing = true;

    try {
      final image = await _controller!.takePicture();
      // Detect faces directly from captured file path (avoids temp-file rewrite overhead)
      final detections = await _detectFaceWithMlKitPath(image.path);
      if (detections.isEmpty) {
        debugPrint('❌ No face detected');
        // Delete captured temp file before returning
        try { await File(image.path).delete(); } catch (_) {}
        return;
      }

      // Decode image only when detection succeeds
      final bytes = await image.readAsBytes();

      // Delete the temporary picture file to free disk/memory
      try { await File(image.path).delete(); } catch (_) {}

      final rawImage = img.decodeImage(bytes);
      if (rawImage != null) {
        debugPrint('📸 Scanning face: ${rawImage.width}x${rawImage.height}');

        _imageSize = Size(
          rawImage.width.toDouble(),
          rawImage.height.toDouble(),
        );

        // Filter valid faces (must be at least 60x60)
        final validFaces = detections
            .where((face) => face.width >= 60 && face.height >= 60)
            .toList();

        if (validFaces.isEmpty) {
          debugPrint('⚠️ No valid faces found');
          return;
        }

        // Clear previous overlays
        _overlayFaces.clear();
        _overlayNames.clear();
        _overlayColors.clear();
        _overlayExpressions.clear();
        final seenInCurrentScan = <int>{};
        final countedInCurrentScan = <int>{};

        // Process each valid face
        for (final face in validFaces) {
          // Crop and generate embedding
          final croppedFace = _cropFace(rawImage, face);
          final embedding = await _generateEmbedding(croppedFace);

          if (embedding.isEmpty) {
            debugPrint('❌ Failed to generate embedding for face');
            // No embedding, show unknown
            _overlayFaces.add(face);
            _overlayNames.add('Unknown');
            _overlayColors.add(Colors.red);
            _overlayExpressions.add('');
            continue;
          }

          // Find matching student with similarity check
          final match = _findMatchingStudent(embedding);
          if (match != null) {
            final studentId = match.id!;
            seenInCurrentScan.add(studentId);

            // For single-face: reset counters if identity changes between frames
            if (validFaces.length == 1) {
              if (_lastSingleFaceMatchId != null && _lastSingleFaceMatchId != studentId) {
                _consecutiveDetectionsMap.clear();
                debugPrint('🔄 Identity changed — consecutive counters reset');
              }
              _lastSingleFaceMatchId = studentId;
            }

            // Track consecutive detections per student (max 1 increment per scan)
            if (countedInCurrentScan.add(studentId)) {
              _consecutiveDetectionsMap[studentId] =
                  (_consecutiveDetectionsMap[studentId] ?? 0) + 1;
            }

            // If we have enough consecutive detections, check cooldown
            if (_consecutiveDetectionsMap[studentId]! >=
                _requiredConsecutiveDetections) {
              final now = DateTime.now();
              final lastTime = _lastDetectionTime[studentId] ?? DateTime(2000);

              if (now.difference(lastTime) >= _detectionCooldown) {
                debugPrint('✅ ${match.name} marked present (confirmed)');
                _lastDetectionTime[studentId] = now;
                _consecutiveDetectionsMap[studentId] = 0;

                if (mounted) {
                  setState(() {
                    _attendanceStatus[studentId] = AttendanceStatus.present;
                    _studentExpressions[studentId] = face.expression;
                  });
                  ScaffoldMessenger.of(context).showSnackBar(
                    SnackBar(
                      content: Text('✅ ${match.name} marked present'),
                      backgroundColor: Colors.green,
                      duration: const Duration(milliseconds: 800),
                    ),
                  );
                  // Speak the attendance confirmation
                  _speakAttendanceConfirmation(match.name);
                }

                // Add to overlay as marked present
                _overlayFaces.add(face);
                _overlayNames.add(match.name);
                _overlayColors.add(Colors.green);
                _overlayExpressions.add(face.expression);
              } else {
                // Cooldown not met, show name with pending
                _overlayFaces.add(face);
                _overlayNames.add(match.name);
                _overlayColors.add(Colors.orange);
                _overlayExpressions.add(face.expression);
              }
            } else {
              // Not enough consecutive yet, show name being detected
              _overlayFaces.add(face);
              _overlayNames.add(match.name);
              _overlayColors.add(Colors.orange);
              _overlayExpressions.add(face.expression);
            }
          } else {
            // No match, show unknown
            _overlayFaces.add(face);
            _overlayNames.add('Unknown');
            _overlayColors.add(Colors.red);
            _overlayExpressions.add(face.expression);
          }
        }

        // Strict consecutive rule: if a student is not seen in this scan, reset their counter
        _consecutiveDetectionsMap.removeWhere(
          (studentId, _) => !seenInCurrentScan.contains(studentId),
        );

        // Set overlay timer to clear after 2 seconds
        _overlayTimer?.cancel();
        _overlayTimer = Timer(const Duration(seconds: 2), () {
          if (mounted) {
            setState(() {
              _overlayFaces.clear();
              _overlayNames.clear();
              _overlayColors.clear();
              _overlayExpressions.clear();
            });
          }
        });

        if (mounted) setState(() {});
      }
    } catch (e) {
      debugPrint('Scan error: $e');
      if ('$e'.contains('Disposed CameraController')) {
        _isScanning = false;
      }
    } finally {
      _isProcessing = false;
    }
  }

  Future<void> _startContinuousScanning() async {
    if (_isScanning) return;
    _isScanning = true;
    if (mounted) setState(() {});

    try {
      while (_isScanning) {
        if (!_isProcessing) {
          await _scanFace();
        }
        await Future.delayed(const Duration(milliseconds: 300));
      }
    } finally {
      _isScanning = false;
      if (mounted) setState(() {});
    }
  }

  void _stopScanning() {
    _isScanning = false;
    _consecutiveDetectionsMap.clear();
    _lastSingleFaceMatchId = null;
    if (mounted) setState(() {});
  }

  // _detectFaceWithMlKit removed from this screen (unused here).

  Future<List<DetectedFace>> _detectFaceWithMlKitPath(String imagePath) async {
    try {
      final faces = await _faceDetector.detectFacesFromPath(imagePath);
      return faces
          .map(
            (face) => DetectedFace(
              x: face.boundingBox.left.toDouble(),
              y: face.boundingBox.top.toDouble(),
              width: face.boundingBox.width.toDouble(),
              height: face.boundingBox.height.toDouble(),
              confidence: 1.0,
              expression: face.expression,
            ),
          )
          .toList();
    } catch (e) {
      debugPrint('Face detection path error: $e');
      return [];
    }
  }

  img.Image _cropFace(img.Image fullImage, DetectedFace face) {
    final x = face.x.toInt().clamp(0, fullImage.width - 1);
    final y = face.y.toInt().clamp(0, fullImage.height - 1);
    final w = face.width.toInt().clamp(1, fullImage.width - x);
    final h = face.height.toInt().clamp(1, fullImage.height - y);
    return img.copyCrop(fullImage, x: x, y: y, width: w, height: h);
  }

  Future<List<double>> _generateEmbedding(img.Image faceImage) async {
    try {
      final faceBytes = Uint8List.fromList(
        img.encodeJpg(faceImage, quality: 100),
      );
      final embedding = await _faceEmbedder.generateEmbedding(faceBytes);
      return embedding ?? [];
    } catch (e) {
      debugPrint('Embedding generation error: $e');
      return [];
    }
  }

  Future<void> _speakAttendanceConfirmation(String studentName) async {
    try {
      // Check if TTS is enabled in settings
      final prefs = await SharedPreferences.getInstance();
      final ttsEnabled = prefs.getBool('tts_enabled') ?? true;
      if (!ttsEnabled) return;

      final message = "$studentName's attendance marked successfully";
      await _flutterTts.speak(message);
      debugPrint('🔊 Speaking: $message');
    } catch (e) {
      debugPrint('TTS error: $e');
    }
  }

  Student? _findMatchingStudent(List<double> embedding) {
    if (embedding.length != FaceEmbeddingModule.embeddingDimension) {
      debugPrint('❌ Query embedding dimension ${embedding.length} != ${FaceEmbeddingModule.embeddingDimension}');
      return null;
    }

    final neighbors = <_KnnNeighbor>[];
    for (final sample in _knnTrainingSet) {
      final dist = _euclideanDistance(embedding, sample.vector);
      if (!dist.isFinite) continue;
      final sim = 1.0 / (1.0 + dist);
      neighbors.add(_KnnNeighbor(studentId: sample.studentId, distance: dist, similarity: sim));
    }

    if (neighbors.isEmpty) return null;

    neighbors.sort((a, b) => a.distance.compareTo(b.distance));
    final k = neighbors.length < _knnK ? neighbors.length : _knnK;
    final topK = neighbors.take(k).toList();

    final voteWeights = <int, double>{};
    final voteCounts = <int, int>{};
    final bestPerStudentSim = <int, double>{};

    for (final n in topK) {
      final sid = n.studentId;
      final dist = n.distance;
      final sim = n.similarity;
      final weight = 1.0 / (dist + 1e-6);

      voteWeights[sid] = (voteWeights[sid] ?? 0.0) + weight;
      voteCounts[sid] = (voteCounts[sid] ?? 0) + 1;

      final currentBest = bestPerStudentSim[sid] ?? 0.0;
      if (sim > currentBest) {
        bestPerStudentSim[sid] = sim;
      }
    }

    if (voteWeights.isEmpty) return null;

    final ranked = voteWeights.entries.toList()
      ..sort((a, b) {
        final byWeight = b.value.compareTo(a.value);
        if (byWeight != 0) return byWeight;

        final c1 = voteCounts[a.key] ?? 0;
        final c2 = voteCounts[b.key] ?? 0;
        final byCount = c2.compareTo(c1);
        if (byCount != 0) return byCount;

        final s1 = bestPerStudentSim[a.key] ?? 0.0;
        final s2 = bestPerStudentSim[b.key] ?? 0.0;
        return s2.compareTo(s1);
      });

    final bestId = ranked.first.key;
    final secondId = ranked.length > 1 ? ranked[1].key : null;
    final bestStudent = _enrolledStudents.firstWhere((s) => s.id == bestId);
    final bestSim = bestPerStudentSim[bestId] ?? 0.0;
    final secondBestSim = secondId != null ? (bestPerStudentSim[secondId] ?? 0.0) : 0.0;
    final margin = bestSim - secondBestSim;
    final effectiveThreshold = _effectiveEuclideanSimilarityThreshold(_similarityThreshold);

    final minVotesRequired = k >= 3 ? 2 : 1;
    final bestVotes = voteCounts[bestId] ?? 0;
    if (bestVotes < minVotesRequired) {
      debugPrint('⚠️ KNN rejected: insufficient votes for ${bestStudent.name} ($bestVotes/$k)');
      return null;
    }

    if (bestSim < effectiveThreshold) {
      debugPrint(
        '🔍 KNN best match: ${bestStudent.name} (${bestSim.toStringAsFixed(3)}) [effective: ${effectiveThreshold.toStringAsFixed(3)} from slider:${_similarityThreshold.toStringAsFixed(2)}] — BELOW THRESHOLD',
      );
      return null;
    }

    // Stage-2 verification: candidate must also strongly match its own saved embeddings.
    final candidateEmbeddings = _studentEmbeddings[bestId] ?? const <List<double>>[];
    double verificationBest = 0.0;
    final verificationScores = <double>[];
    for (final saved in candidateEmbeddings) {
      final dist = _euclideanDistance(embedding, saved);
      if (!dist.isFinite) continue;
      final sim = 1.0 / (1.0 + dist);
      verificationScores.add(sim);
      if (sim > verificationBest) verificationBest = sim;
    }

    int verificationSupport = 0;
    for (final saved in candidateEmbeddings) {
      final dist = _euclideanDistance(embedding, saved);
      if (!dist.isFinite) continue;
      final sim = 1.0 / (1.0 + dist);
      if (sim >= effectiveThreshold) verificationSupport++;
    }

    if (verificationBest < effectiveThreshold) {
      debugPrint(
        '⚠️ Verification failed for ${bestStudent.name}: ${verificationBest.toStringAsFixed(3)} < ${effectiveThreshold.toStringAsFixed(3)}',
      );
      return null;
    }

    // Strong unknown rejection: require multiple template agreements when enough samples exist.
    final minSupport = candidateEmbeddings.length >= 8
        ? 3
        : candidateEmbeddings.length >= 5
            ? 2
            : 1;
    if (verificationSupport < minSupport) {
      debugPrint(
        '⚠️ Verification support failed for ${bestStudent.name}: support $verificationSupport < $minSupport',
      );
      return null;
    }

    // Additional hard guard: average of top-3 verification scores must be strong.
    verificationScores.sort((a, b) => b.compareTo(a));
    final topCount = verificationScores.length >= 3 ? 3 : verificationScores.length;
    double topAvg = 0.0;
    for (int i = 0; i < topCount; i++) {
      topAvg += verificationScores[i];
    }
    topAvg = topCount > 0 ? topAvg / topCount : 0.0;

    if (topCount >= 2 && topAvg < (effectiveThreshold + 0.02)) {
      debugPrint(
        '⚠️ Verification top-$topCount average failed for ${bestStudent.name}: ${topAvg.toStringAsFixed(3)} < ${(effectiveThreshold + 0.02).toStringAsFixed(3)}',
      );
      return null;
    }

    final double requiredMargin = _similarityThreshold <= 0.75
        ? 0.005
        : _similarityThreshold <= 0.80
            ? 0.010
            : _similarityThreshold <= 0.85
                ? 0.020
                : 0.030;

    final bool isStrongMatch = bestSim >= (_similarityThreshold + 0.05);

    if (!isStrongMatch && ranked.length > 1 && margin < requiredMargin) {
      final secondStudent = _enrolledStudents.firstWhere((s) => s.id == secondId);
      debugPrint(
        '⚠️ KNN AMBIGUOUS: ${bestStudent.name} (${bestSim.toStringAsFixed(3)}) vs ${secondStudent.name} (${secondBestSim.toStringAsFixed(3)}) — margin ${margin.toStringAsFixed(3)} < $requiredMargin',
      );
      return null;
    }

    debugPrint(
      '🔍 KNN best match: ${bestStudent.name} (${bestSim.toStringAsFixed(3)}) verify:${verificationBest.toStringAsFixed(3)} topAvg:${topAvg.toStringAsFixed(3)} support:$verificationSupport [effective: ${effectiveThreshold.toStringAsFixed(3)} from slider:${_similarityThreshold.toStringAsFixed(2)}] margin: ${margin.toStringAsFixed(3)} k:$k votes:$bestVotes',
    );
    return bestStudent;
  }

  // Slider values were originally tuned for cosine similarity. Convert them to
  // equivalent Euclidean-similarity scale (sim = 1 / (1 + distance)).
  double _effectiveEuclideanSimilarityThreshold(double sliderThreshold) {
    final cos = sliderThreshold.clamp(0.0, 0.999).toDouble();
    final d = sqrt((2.0 - (2.0 * cos)).clamp(0.0, double.infinity));
    return 1.0 / (1.0 + d);
  }

  double _euclideanDistance(List<double> a, List<double> b) {
    if (a.isEmpty || b.isEmpty || a.length != b.length) return double.infinity;
    final dim = a.length;
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
      final diff = a[i] - b[i];
      sum += diff * diff;
    }
    return sqrt(sum);
  }

  void _rebuildKnnTrainingSet() {
    _knnTrainingSet.clear();
    _studentEmbeddings.forEach((studentId, vectors) {
      for (final vector in vectors) {
        if (vector.length == FaceEmbeddingModule.embeddingDimension) {
          _knnTrainingSet.add(_KnnSample(studentId: studentId, vector: vector));
        }
      }
    });
    debugPrint('🧠 KNN trained with ${_knnTrainingSet.length} samples (${FaceEmbeddingModule.embeddingDimension}D)');
  }

  Future<void> _submitAttendance() async {
    if (_attendanceStatus.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('No attendance marked')));
      return;
    }

    try {
      int submitted = 0;
      for (final entry in _attendanceStatus.entries) {
        await _dbManager.insertAttendance(
          AttendanceRecord(
            studentId: entry.key,
            date: _attendanceDate!,
            time: '${DateTime.now().hour}:${DateTime.now().minute}',
            status: entry.value,
            recordedAt: DateTime.now(),
          ),
        );
        submitted++;
      }

      final subjectId =
          widget.subject.id ?? DateTime.now().millisecondsSinceEpoch;
      final prefs = await SharedPreferences.getInstance();
      final sessionKey = _sessionAttendanceKey(
        teacherName: widget.teacherName,
        subjectId: subjectId,
        date: _attendanceDate!,
      );
      final sessionPayload = <String, String>{};
      for (final entry in _attendanceStatus.entries) {
        sessionPayload[entry.key.toString()] = entry.value.name;
      }
      await prefs.setString(sessionKey, jsonEncode(sessionPayload));

      final existingSessions = await _dbManager.getTeacherSessionsByDate(
        _attendanceDate!,
      );
      final alreadySaved = existingSessions.any(
        (session) =>
            session.subjectId == subjectId &&
            session.teacherName.toLowerCase() ==
                widget.teacherName.toLowerCase(),
      );
      if (!alreadySaved) {
        await _dbManager.insertTeacherSession(
          TeacherSession(
            id: DateTime.now().millisecondsSinceEpoch,
            teacherName: widget.teacherName,
            subjectId: subjectId,
            subjectName: widget.subject.name,
            date: _attendanceDate!,
            createdAt: DateTime.now(),
          ),
        );
      }
      debugPrint('✅ Attendance submitted for $submitted students');

      // Auto-generate CSV file (MUST complete before showing dialog)
      String? csvError;
      try {
        await _generateAttendanceCSV();
      } catch (e) {
        csvError = e.toString();
        debugPrint('⚠️ CSV generation failed in submit: $csvError');
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
              csvError != null
                  ? '✅ Attendance saved ($submitted) — CSV failed: $csvError'
                  : '✅ Attendance submitted for $submitted students + CSV saved',
            ),
            backgroundColor: csvError != null
                ? AppConstants.warningColor
                : AppConstants.successColor,
            duration: const Duration(seconds: 2),
          ),
        );

        // Show animated success, then close screen
        showDialog(
          context: context,
          barrierDismissible: false,
          builder: (ctx) => Center(
            child: Material(
              color: Colors.transparent,
              child: TweenAnimationBuilder<double>(
                tween: Tween(begin: 0.0, end: 1.0),
                duration: const Duration(milliseconds: 700),
                builder: (context, val, child) =>
                    Transform.scale(scale: val, child: child),
                child: Container(
                  padding: const EdgeInsets.all(28),
                  decoration: BoxDecoration(
                    color: Theme.of(context).colorScheme.surface,
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [AppConstants.cardShadow],
                  ),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.check_circle,
                        size: 72,
                        color: ColorSchemes.presentColor,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        'Attendance Saved',
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );

        Future.delayed(const Duration(milliseconds: 900), () {
          if (mounted) {
            Navigator.of(context).pop(); // close dialog
            Navigator.of(context).pop(); // go back
          }
        });
      }
    } catch (e) {
      debugPrint('Submit error: $e');
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }

  String _sessionAttendanceKey({
    required String teacherName,
    required int subjectId,
    required DateTime date,
  }) {
    final dateStr =
        '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
    final safeTeacher = teacherName
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '_')
        .replaceAll(RegExp(r'_+'), '_')
        .trim();
    return 'session_attendance_${safeTeacher}_${subjectId}_$dateStr';
  }

  static const MethodChannel _csvPlatform = MethodChannel(
    'com.coad.faceattendance/save',
  );

  Future<void> _generateAttendanceCSV() async {
    final date = _attendanceDate ?? DateTime.now();
    final dateStr =
        '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';

    // ── Resolve export directory (shared utility) ──
    final dir = await getExportDirectory();
    debugPrint('📁 Export dir: ${dir.path}');

    // ── 1) Subject Attendance CSV ──
    // Build the CSV string inline (no external function dependency)
    final allStudents = await _dbManager.getAllStudents();
    final studentMap = <int, Student>{};
    for (final s in allStudents) {
      if (s.id != null) studentMap[s.id!] = s;
    }

    final presentNames = <String>[];
    final presentExpressions = <String>[];
    final absentNames = <String>[];
    for (final entry in studentMap.entries) {
      final status = _attendanceStatus[entry.key] ?? AttendanceStatus.absent;
      if (status == AttendanceStatus.present) {
        presentNames.add(entry.value.name);
        presentExpressions.add(_studentExpressions[entry.key] ?? '');
      } else {
        absentNames.add(entry.value.name);
      }
    }

    final buf = StringBuffer();
    buf.writeln('Teacher Name,Subject');
    buf.writeln('"${widget.teacherName}","${widget.subject.name}"');
    buf.writeln('');
    // Date in its own row (date in second column)
    buf.writeln('Date:,$dateStr');
    buf.writeln('');
    // Single-summary cell (quoted) so it's visible in one column as text
    buf.writeln('"Attendees = ${presentNames.length}, Absentees = ${absentNames.length}, Total = ${studentMap.length}"');
    buf.writeln('');

    // Header row: Absentees, Attendees, Expression (absentees first)
    buf.writeln('Absentees,Attendees,Expression');
    // Now list names side-by-side under Absentees and Attendees columns,
    // with the Expression column aligned to the Attendees row.
    final maxLen = presentNames.length > absentNames.length ? presentNames.length : absentNames.length;
    for (int i = 0; i < maxLen; i++) {
      final a = i < absentNames.length ? absentNames[i] : '';
      final p = i < presentNames.length ? presentNames[i] : '';
      final e = i < presentExpressions.length ? presentExpressions[i] : '';
      buf.writeln('"$a","$p","$e"');
    }

    final safeSubject = widget.subject.name
        .replaceAll(RegExp(r'[<>:"/\\|?*\s]'), '_')
        .replaceAll(RegExp(r'_+'), '_');
    final subjectFileName = '${safeSubject}_$dateStr.csv';
    final subjectFile = File('${dir.path}/$subjectFileName');
    await subjectFile.writeAsString(buf.toString(), flush: true);

    // Verify file was actually written
    if (await subjectFile.exists()) {
      debugPrint('✅ Subject CSV written (${await subjectFile.length()} bytes): ${subjectFile.path}');
    } else {
      debugPrint('❌ Subject CSV NOT found after write!');
    }

    // ── 2) Cumulative Attendance CSV (master register) ──
    // Format: names on top, dates as rows, totals at bottom
    try {
      final allRecords = await _dbManager.getAllAttendance();
      if (allStudents.isNotEmpty && allRecords.isNotEmpty) {
        // Collect all unique dates
        final dates = <DateTime>{};
        for (final r in allRecords) {
          dates.add(DateTime(r.date.year, r.date.month, r.date.day));
        }
        final sortedDates = dates.toList()..sort();

        // Build attendance lookup: studentId → date → status
        final lookup = <int, Map<String, String>>{};
        for (final r in allRecords) {
          final key =
              '${r.date.year}-${r.date.month.toString().padLeft(2, '0')}-${r.date.day.toString().padLeft(2, '0')}';
          lookup.putIfAbsent(r.studentId, () => {});
          lookup[r.studentId]![key] =
              r.status == AttendanceStatus.present ? '1' : '0';
        }

        final csv = StringBuffer();

        // Header row: blank + student names
        csv.write('Date');
        for (final s in allStudents) {
          csv.write(',"${s.name}"');
        }
        csv.writeln();

        // Data rows: one per date
        final totalPresent = <int, int>{};
        final totalAbsent = <int, int>{};
        for (final s in allStudents) {
          totalPresent[s.id!] = 0;
          totalAbsent[s.id!] = 0;
        }
        final totalClasses = sortedDates.length;

        for (final date in sortedDates) {
          final dStr =
              '${date.year}-${date.month.toString().padLeft(2, '0')}-${date.day.toString().padLeft(2, '0')}';
          csv.write(dStr);
          for (final s in allStudents) {
            final val = lookup[s.id]?[dStr] ?? '0';
            csv.write(',$val');
            if (val == '1') {
              totalPresent[s.id!] = (totalPresent[s.id!] ?? 0) + 1;
            } else {
              totalAbsent[s.id!] = (totalAbsent[s.id!] ?? 0) + 1;
            }
          }
          csv.writeln();
        }

        // Summary rows
        csv.write('Attendees');
        for (final s in allStudents) {
          csv.write(',${totalPresent[s.id!] ?? 0}');
        }
        csv.writeln();

        csv.write('Absentees');
        for (final s in allStudents) {
          csv.write(',${totalAbsent[s.id!] ?? 0}');
        }
        csv.writeln();

        csv.write('Attendance %');
        for (final s in allStudents) {
          final p = totalPresent[s.id!] ?? 0;
          final pct = totalClasses > 0 ? (p / totalClasses * 100).round() : 0;
          csv.write(',$pct%');
        }
        csv.writeln();

        csv.write('Total Classes Taken');
        for (final s in allStudents) {
          csv.write(',$totalClasses');
        }
        csv.writeln();

        final cumulativeFile = File('${dir.path}/attendance_register.csv');
        await cumulativeFile.writeAsString(csv.toString(), flush: true);
        debugPrint('✅ Cumulative CSV updated: ${cumulativeFile.path}');
      }
    } catch (e) {
      debugPrint('⚠️ Cumulative CSV failed (non-fatal): $e');
    }

    // ── 3) MediaStore copy for Android visibility ──
    try {
      if (Platform.isAndroid) {
        final bytes = await subjectFile.readAsBytes();
        final base64data = base64Encode(bytes);
        await _csvPlatform.invokeMethod('saveToDownloads', {
          'filename': subjectFileName,
          'dataBase64': base64data,
          'subFolder': 'FaceAttendanceExports',
        });
      }
    } catch (e) {
      debugPrint('⚠️ MediaStore save failed (non-fatal): $e');
    }
  }

  Widget _buildFaceOverlay(Size displaySize) {
    if (_overlayFaces.isEmpty) return const SizedBox.shrink();

    return Stack(
      children: List.generate(_overlayFaces.length, (index) {
        final face = _overlayFaces[index];
        final name = _overlayNames[index];
        final color = _overlayColors[index];
        final expr = index < _overlayExpressions.length ? _overlayExpressions[index] : '';

        // ── Coordinate mapping overview ──
        // takePicture() returns a JPEG that is already rotation-corrected (EXIF
        // applied by the camera plugin). img.decodeImage gives us the upright
        // image (e.g. 1440×1920 portrait). ML Kit InputImage.fromFilePath also
        // reads the EXIF, so the bounding box is in upright image coordinates.
        // Therefore: NO sensorOrientation rotation is needed.  We just need to
        // scale face-box coords from image-space → display-space, accounting
        // for how CameraPreview fills the widget.

        final double imgW = _imageSize!.width;
        final double imgH = _imageSize!.height;

        // Face bounding box in image coordinates (upright)
        double faceX = face.x.toDouble();
        double faceY = face.y.toDouble();
        double faceW = face.width.toDouble();
        double faceH = face.height.toDouble();

        // The CameraPreview fills its parent via SizedBox.expand, so it uses
        // "cover" behaviour: the preview is scaled so the shorter axis matches
        // the widget, and the longer axis overflows (clipped by ClipRRect).
        // The captured JPEG and preview share the same aspect ratio from the
        // same camera, so we can treat the mapping as a simple "cover" fit
        // from imgW×imgH → displaySize.

        final double dispW = displaySize.width;
        final double dispH = displaySize.height;

        // Cover: scale to fill, crop overflow
        final double scale = max(dispW / imgW, dispH / imgH);
        final double scaledImgW = imgW * scale;
        final double scaledImgH = imgH * scale;
        // Offset to center the scaled image inside the display area
        final double offsetX = (dispW - scaledImgW) / 2;
        final double offsetY = (dispH - scaledImgH) / 2;

        double mappedX = faceX * scale + offsetX;
        double mappedY = faceY * scale + offsetY;
        double mappedW = faceW * scale;
        double mappedH = faceH * scale;

        // Mirror horizontally for front camera (preview is mirrored)
        if (_currentCamera.lensDirection == CameraLensDirection.front) {
          mappedX = dispW - (mappedX + mappedW);
        }

        // Circle centred on the mapped face box
        final double centerX = mappedX + mappedW / 2;
        final double centerY = mappedY + mappedH / 2;
        final double radius = max(mappedW, mappedH) / 2;

        final double circleLeft = centerX - radius;
        final double circleTop = centerY - radius;

        // Clamp so the circle stays within the display area
        final double maxLeft = max(0.0, dispW - radius * 2);
        final double maxTop = max(0.0, dispH - radius * 2);

        return Positioned(
          left: circleLeft.clamp(0.0, maxLeft),
          top: circleTop.clamp(0.0, maxTop),
          width: radius * 2,
          height: radius * 2,
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              // Circle border
              Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: color, width: 3),
                ),
              ),
              // Name + expression label above the circle
              Positioned(
                top: -30,
                left: -20,
                right: -20,
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: color.withAlpha(179),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          name,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                          textAlign: TextAlign.center,
                          overflow: TextOverflow.ellipsis,
                        ),
                        if (expr.isNotEmpty)
                          Text(
                            expr,
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 10,
                              fontWeight: FontWeight.w500,
                            ),
                            textAlign: TextAlign.center,
                          ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        );
      }),
    );
  }

  @override
  Widget build(BuildContext context) {
    final isReady = _controller != null && _controller!.value.isInitialized;
    final markedCount = _attendanceStatus.values
        .where((s) => s == AttendanceStatus.present)
        .length;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Mark Attendance'),
        elevation: 0,
        flexibleSpace: Container(
          decoration: BoxDecoration(gradient: AppConstants.blueGradient),
        ),
        actions: [
          if (markedCount > 0)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: ColorSchemes.presentColor.withAlpha(26),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Text(
                    '✓ $markedCount Marked',
                    style: const TextStyle(
                      color: ColorSchemes.presentColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 12,
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: _submitAttendance,
        icon: const Icon(Icons.send),
        label: const Text('Submit Attendance'),
      ),
      body: AnimatedBackground(
        child: Column(
          children: [
            // Camera Preview Section
            if (isReady)
              Expanded(
                flex: 3,
                child: Center(
                  child: Container(
                    margin: const EdgeInsets.all(AppConstants.paddingMedium),
                    constraints: const BoxConstraints(maxWidth: 500),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(
                        AppConstants.borderRadiusLarge,
                      ),
                      border: Border.all(
                        color:
                            _attendanceStatus.containsValue(
                              AttendanceStatus.present,
                            )
                            ? Colors.green
                            : AppConstants.cardBorder,
                        width:
                            _attendanceStatus.containsValue(
                              AttendanceStatus.present,
                            )
                            ? 3
                            : 2,
                      ),
                      boxShadow: [AppConstants.cardShadow],
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(
                        AppConstants.borderRadiusLarge,
                      ),
                      child: ExcludeSemantics(
                        child: LayoutBuilder(
                        builder: (context, constraints) {
                          final displaySize = constraints.biggest;
                          return Stack(
                            fit: StackFit.expand,
                            children: [
                              // Camera preview fills the container (cover behaviour)
                              RepaintBoundary(
                                child: FittedBox(
                                    fit: BoxFit.cover,
                                    clipBehavior: Clip.hardEdge,
                                    child: SizedBox(
                                      width:
                                          _controller!
                                              .value
                                              .previewSize
                                              ?.height ??
                                          1,
                                      height:
                                          _controller!.value.previewSize?.width ??
                                          1,
                                      child: CameraPreview(_controller!),
                                    ),
                                ),
                              ),
                              if (_isFrontCamera && _isFrontLightOn)
                                IgnorePointer(
                                  child: CustomPaint(
                                    painter: _CenterOvalFlashMaskPainter(),
                                    child: const SizedBox.expand(),
                                  ),
                                ),
                              // Face Overlay
                              if (_overlayFaces.isNotEmpty &&
                                  _imageSize != null)
                                _buildFaceOverlay(displaySize),
                              // Processing Overlay
                              if (_isProcessing)
                                Container(
                                  decoration: BoxDecoration(
                                    color: Colors.black.withAlpha(102),
                                  ),
                                  child: const Center(
                                    child: Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
                                      children: [
                                        CircularProgressIndicator(
                                          valueColor:
                                              AlwaysStoppedAnimation<Color>(
                                                AppConstants.primaryColor,
                                              ),
                                        ),
                                        SizedBox(
                                          height: AppConstants.paddingMedium,
                                        ),
                                        Text(
                                          'Scanning face...',
                                          style: TextStyle(
                                            color: Colors.white,
                                            fontSize: 14,
                                            fontWeight: FontWeight.w600,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ),
                              // Scan Status Badge
                              Positioned(
                                top: 12,
                                right: 12,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 12,
                                    vertical: 6,
                                  ),
                                  decoration: BoxDecoration(
                                    color: _isScanning
                                        ? ColorSchemes.presentColor
                                        : Colors.grey,
                                    borderRadius: BorderRadius.circular(20),
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.black.withAlpha(77),
                                        blurRadius: 8,
                                      ),
                                    ],
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Container(
                                        width: 8,
                                        height: 8,
                                        decoration: BoxDecoration(
                                          color: Colors.white,
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                      const SizedBox(width: 6),
                                      Text(
                                        _isScanning ? 'Scanning' : 'Ready',
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontWeight: FontWeight.bold,
                                          fontSize: 12,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              // Camera Switch Button
                              if (_availableCameras.length > 1)
                                Positioned(
                                  top: 12,
                                  left: 12,
                                  child: Container(
                                    decoration: BoxDecoration(
                                      color: Colors.black.withAlpha(153),
                                      shape: BoxShape.circle,
                                    ),
                                    child: IconButton(
                                      onPressed: _switchCamera,
                                      icon: const Icon(
                                        Icons.cameraswitch,
                                        color: Colors.white,
                                        size: 24,
                                      ),
                                    ),
                                  ),
                                ),
                              Positioned(
                                top: 72,
                                left: 12,
                                child: Container(
                                  decoration: BoxDecoration(
                                    color: Colors.black.withAlpha(153),
                                    shape: BoxShape.circle,
                                  ),
                                  child: IconButton(
                                    onPressed: _toggleCameraLight,
                                    icon: Icon(
                                      _isFrontCamera
                                          ? (_isFrontLightOn
                                              ? Icons.wb_sunny
                                              : Icons.wb_sunny_outlined)
                                          : (_isBackFlashOn
                                              ? Icons.flash_on
                                              : Icons.flash_off),
                                      color: Colors.white,
                                      size: 24,
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          );
                        },
                      ),
                      ),
                    ),
                  ),
                ),
              )
            else
              Expanded(
                flex: 3,
                child: Center(
                  child: Container(
                    margin: const EdgeInsets.all(AppConstants.paddingMedium),
                    constraints: const BoxConstraints(maxWidth: 500),
                    decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(
                        AppConstants.borderRadiusLarge,
                      ),
                      border: Border.all(
                        color: AppConstants.cardBorder,
                        width: 2,
                      ),
                      color: AppConstants.cardColor,
                    ),
                    child: const Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(height: AppConstants.paddingMedium),
                          Text(
                            'Initializing Camera...',
                            style: TextStyle(
                              color: AppConstants.textSecondary,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),

            // Student List Section
            Expanded(
              flex: 2,
              child: Container(
                margin: const EdgeInsets.symmetric(
                  horizontal: AppConstants.paddingMedium,
                  vertical: AppConstants.paddingSmall,
                ),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(
                    AppConstants.borderRadiusLarge,
                  ),
                  border: Border.all(color: AppConstants.cardBorder),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(
                    AppConstants.borderRadiusLarge,
                  ),
                  child: _enrolledStudents.isEmpty
                      ? Center(
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.person_off,
                                size: 48,
                                color: AppConstants.textTertiary,
                              ),
                              const SizedBox(
                                height: AppConstants.paddingMedium,
                              ),
                              const Text(
                                'No enrolled students',
                                style: TextStyle(
                                  color: AppConstants.textSecondary,
                                  fontSize: 14,
                                ),
                              ),
                            ],
                          ),
                        )
                      : ListView.separated(
                          itemCount: _enrolledStudents.length,
                          separatorBuilder: (context, index) => Container(
                            height: 1,
                            color: AppConstants.cardBorder,
                          ),
                          itemBuilder: (context, index) {
                            final student = _enrolledStudents[index];
                            final status = _attendanceStatus[student.id];
                            final isPresent =
                                status == AttendanceStatus.present;
                            final studentExpr = _studentExpressions[student.id] ?? '';
                            final initials = student.name
                                .split(' ')
                                .where((s) => s.isNotEmpty)
                                .map((s) => s[0])
                                .take(2)
                                .join();

                            return GestureDetector(
                              onTap: () {
                                setState(() {
                                  if (isPresent) {
                                    _attendanceStatus.remove(student.id);
                                  } else {
                                    _attendanceStatus[student.id!] =
                                        AttendanceStatus.present;
                                  }
                                });
                              },
                              child: AnimatedContainer(
                                duration: const Duration(milliseconds: 280),
                                margin: const EdgeInsets.symmetric(
                                  vertical: 6,
                                  horizontal: 8,
                                ),
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 12,
                                  vertical: 10,
                                ),
                                decoration: BoxDecoration(
                                  color: isPresent
                                      ? ColorSchemes.presentColor.withValues(alpha: 0.08)
                                      : AppConstants.cardColor,
                                  borderRadius: BorderRadius.circular(12),
                                  boxShadow: isPresent
                                      ? [
                                          BoxShadow(
                                            color: ColorSchemes.presentColor.withValues(alpha: 0.12),
                                            blurRadius: 8,
                                            offset: const Offset(0, 2),
                                          ),
                                        ]
                                      : [],
                                ),
                                child: Row(
                                  children: [
                                    CircleAvatar(
                                      radius: 22,
                                      backgroundColor: isPresent
                                          ? ColorSchemes.presentColor
                                          : AppConstants.inputFill,
                                      child: Text(
                                        initials.toUpperCase(),
                                        style: TextStyle(
                                          color: isPresent
                                              ? Colors.white
                                              : AppConstants.textTertiary,
                                          fontWeight: FontWeight.w700,
                                        ),
                                      ),
                                    ),
                                    const SizedBox(
                                      width: AppConstants.paddingMedium,
                                    ),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          Row(
                                            children: [
                                              Flexible(
                                                child: Text(
                                                  student.name,
                                                  style: const TextStyle(
                                                    fontWeight: FontWeight.w600,
                                                    fontSize: 14,
                                                  ),
                                                ),
                                              ),
                                              if (isPresent && studentExpr.isNotEmpty) ...[
                                                const SizedBox(width: 6),
                                                Container(
                                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                                  decoration: BoxDecoration(
                                                    color: AppConstants.primaryColor.withAlpha(20),
                                                    borderRadius: BorderRadius.circular(8),
                                                    border: Border.all(color: AppConstants.primaryColor.withAlpha(60)),
                                                  ),
                                                  child: Text(
                                                    studentExpr,
                                                    style: TextStyle(
                                                      fontSize: 10,
                                                      fontWeight: FontWeight.w600,
                                                      color: AppConstants.primaryColor,
                                                    ),
                                                  ),
                                                ),
                                              ],
                                            ],
                                          ),
                                          const SizedBox(height: 4),
                                          Text(
                                            '${student.rollNumber} • ${student.className}',
                                            style: const TextStyle(
                                              color: AppConstants.textTertiary,
                                              fontSize: 12,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                    AnimatedSwitcher(
                                      duration: const Duration(
                                        milliseconds: 220,
                                      ),
                                      transitionBuilder: (child, anim) =>
                                          ScaleTransition(
                                            scale: anim,
                                            child: child,
                                          ),
                                      child: isPresent
                                          ? Icon(
                                              Icons.check_circle,
                                              key: const ValueKey('present'),
                                              color: ColorSchemes.presentColor,
                                              size: 28,
                                            )
                                          : Icon(
                                              Icons.radio_button_unchecked,
                                              key: const ValueKey('absent'),
                                              color: AppConstants.textTertiary,
                                              size: 20,
                                            ),
                                    ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                ),
              ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: AppConstants.secondaryColor,
          border: Border(top: BorderSide(color: AppConstants.cardBorder)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withAlpha(77),
              blurRadius: 8,
              offset: const Offset(0, -2),
            ),
          ],
        ),
        padding: EdgeInsets.fromLTRB(
          AppConstants.paddingMedium,
          AppConstants.paddingMedium,
          AppConstants.paddingMedium,
          AppConstants.paddingMedium + MediaQuery.of(context).padding.bottom,
        ),
        child: Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: _isScanning
                    ? _stopScanning
                    : _startContinuousScanning,
                icon: Icon(_isScanning ? Icons.stop_circle : Icons.videocam),
                label: Text(
                  _isScanning ? 'Stop' : 'Scan',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: _isScanning
                      ? AppConstants.warningColor
                      : AppConstants.primaryColor,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _CenterOvalFlashMaskPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;

    final layerRect = Offset.zero & size;
    canvas.saveLayer(layerRect, Paint());

    final overlayPaint = Paint()..color = Colors.white.withAlpha(245);
    canvas.drawRect(layerRect, overlayPaint);

    final ovalRect = Rect.fromCenter(
      center: Offset(size.width / 2, size.height / 2),
      width: size.width * 0.62,
      height: size.height * 0.72,
    );

    final clearPaint = Paint()..blendMode = BlendMode.clear;
    canvas.drawOval(ovalRect, clearPaint);
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

/// Custom Painter for Face Detection Bounding Box (Spider-Man Mask Style)
