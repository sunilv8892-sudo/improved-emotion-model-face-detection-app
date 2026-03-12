import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

import '../models/face_detection_model.dart';
import '../modules/m1_face_detection.dart' as face_detection_module;
import '../modules/m6_emotion_detection.dart';
import '../utils/constants.dart';

// ──────────────────────────────────────────────────────────────
//  Expression Detection Screen
//  Camera → ML Kit face detection → expression overlay + log
//  Radically simplified layout: no bottomNavigationBar, no
//  FittedBox, no LayoutBuilder — just CameraPreview in a
//  constrained container (same approach as EnrollmentScreen).
// ──────────────────────────────────────────────────────────────

class ExpressionDetectionScreen extends StatefulWidget {
  const ExpressionDetectionScreen({super.key});

  @override
  State<ExpressionDetectionScreen> createState() =>
      _ExpressionDetectionScreenState();
}

class _ExpressionDetectionScreenState extends State<ExpressionDetectionScreen> {
  static const int _stabilityWindowSize = 3;
  static const double _minStableConfidence = 0.22;
  static const double _minStableMargin = 0.01;
  static const double _holdStableConfidence = 0.18;
  static const Duration _scanInterval = Duration(milliseconds: 700);
  static const Duration _minLogInterval = Duration(seconds: 2);
  static const Duration _stableEmotionHold = Duration(seconds: 2);

  CameraController? _controller;
  late face_detection_module.FaceDetectionModule _faceDetector;
  late EmotionDetectionModule _emotionDetector;
  List<CameraDescription> _availableCameras = [];
  late CameraDescription _currentCamera;

  bool _isProcessing = false;
  bool _isScanning = false;
  bool _emotionModelReady = false;
  String _pipelineStatusText = 'Loading emotion pipeline...';

  // Overlay
  final List<DetectedFace> _overlayFaces = [];
  final List<String> _overlayExpressions = [];
  final List<Color> _overlayColors = [];
  Size? _imageSize;
  Timer? _overlayTimer;

  // Log
  final List<_ExpressionLogEntry> _expressionLog = [];
  static const int _maxLogEntries = 50;
  final List<_TemporalEmotionSample> _recentEmotionSamples = [];
  String? _lastLoggedEmotion;
  DateTime? _lastLoggedAt;
  String? _lastStableEmotion;
  double _lastStableConfidence = 0.0;
  DateTime? _lastStableAt;

  // Key used to read camera container size for overlay mapping
  final GlobalKey _cameraKey = GlobalKey();

  @override
  void initState() {
    super.initState();
    _initialize();
  }

  @override
  void dispose() {
    _isScanning = false;
    _overlayTimer?.cancel();
    _controller?.dispose();
    _faceDetector.dispose();
    _emotionDetector.dispose();
    super.dispose();
  }

  // ─────────────────── Init ─────────────────────────────────

  Future<void> _initialize() async {
    try {
      _faceDetector = face_detection_module.FaceDetectionModule();
      await _faceDetector.initialize();

      // Initialize the paper-style emotion pipeline.
      _emotionDetector = EmotionDetectionModule();
      try {
        await _emotionDetector.initialize();
        _emotionModelReady = true;
        _pipelineStatusText = 'Paper emotion pipeline ready';
        debugPrint('✅ Emotion pipeline loaded');
      } catch (e) {
        _emotionModelReady = false;
        _pipelineStatusText = 'Model unavailable, fallback heuristics active';
        debugPrint('⚠️ Emotion pipeline not available, using fallback: $e');
      }

      await _initCamera();
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Expression init error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Error: $e')));
      }
    }
  }

  Future<void> _initCamera() async {
    try {
      final status = await Permission.camera.request();
      if (!status.isGranted) return;
      _availableCameras = await availableCameras();
      if (_availableCameras.isEmpty) return;
      _currentCamera = _availableCameras.first;
      final preferred = _availableCameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => _availableCameras.first,
      );
      await _startCamera(preferred);
    } catch (e) {
      debugPrint('Camera error: $e');
    }
  }

  Future<void> _startCamera(CameraDescription camera) async {
    try {
      await _controller?.dispose();
      _resetEmotionHistory();
      _currentCamera = camera;
      _controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );
      await _controller!.initialize();
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Start camera error: $e');
    }
  }

  Future<void> _switchCamera() async {
    if (_availableCameras.length < 2) return;
    final next = _availableCameras.lastWhere(
      (c) => c.lensDirection != _currentCamera.lensDirection,
      orElse: () => _availableCameras.first,
    );
    await _startCamera(next);
  }

  // ─────────────────── Scanning ─────────────────────────────

  Future<void> _scanExpression() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isProcessing) return;
    _isProcessing = true;

    try {
      final xFile = await _controller!.takePicture();
      final bytes = await xFile.readAsBytes();
      final rawImage = img.decodeImage(bytes);
      if (rawImage == null) return;

      debugPrint('📸 Expression scan: ${rawImage.width}x${rawImage.height}');

      final detections = await _detectFaces(bytes);
      if (detections.isEmpty) {
        _resetEmotionHistory();
        debugPrint('❌ No face detected');
        return;
      }

      _imageSize = Size(
        rawImage.width.toDouble(),
        rawImage.height.toDouble(),
      );

      final validFaces =
          detections.where((f) => f.width >= 60 && f.height >= 60).toList();
      if (validFaces.isEmpty) {
        _resetEmotionHistory();
        return;
      }

      if (validFaces.length != 1) {
        _resetEmotionHistory();
      }

      _overlayFaces.clear();
      _overlayExpressions.clear();
      _overlayColors.clear();

      for (final face in validFaces) {
        String expr = 'Neutral';
        double confidence = 0.0;

        try {
          // Crop face ROI from the full image for emotion detection
          final int fx = face.x.round().clamp(0, rawImage.width - 1);
          final int fy = face.y.round().clamp(0, rawImage.height - 1);
          final int fw =
              face.width.round().clamp(1, rawImage.width - fx);
          final int fh =
              face.height.round().clamp(1, rawImage.height - fy);
          final faceROI =
              img.copyCrop(rawImage, x: fx, y: fy, width: fw, height: fh);
          final faceBytes = Uint8List.fromList(img.encodeJpg(faceROI));

          final emotionResult =
              await _emotionDetector.detectEmotionWithFallback(
            faceBytes,
            poseX: face.poseX ?? 0.0,
            poseY: face.poseY ?? 0.0,
            );
          final decision = validFaces.length == 1
              ? _stabilizeEmotion(emotionResult)
              : _rawDecision(emotionResult);
          expr = decision.label;
          confidence = decision.confidence;
          _pipelineStatusText = decision.statusText;
          // Debug: print top-3 raw probabilities
          final sortedProbs = emotionResult.probabilities.entries.toList()
            ..sort((a, b) => b.value.compareTo(a.value));
          final top3 = sortedProbs.take(3).map((e) => '${e.key}:${(e.value * 100).toStringAsFixed(1)}%').join(' ');
          debugPrint(
              '🎭 Emotion: $expr (${(confidence * 100).toStringAsFixed(1)}%) '
              'top3=[$top3] '
              '${emotionResult.isFallback ? "[fallback]" : "[SVM pipeline]"}');

          if (decision.accepted) {
            _appendStableEmotionLog(decision.label);
          }
        } catch (e) {
          expr = 'Analyzing';
          confidence = 0.0;
          _pipelineStatusText = 'Emotion inference failed, using neutral fallback';
          debugPrint('⚠️ Emotion detection failed, using Neutral fallback: $e');
        }

        _overlayFaces.add(face);
        _overlayExpressions
            .add(confidence > 0 ? '$expr ${(confidence * 100).toStringAsFixed(0)}%' : expr);
        _overlayColors.add(_colorForExpression(expr));
      }

      _overlayTimer?.cancel();
      _overlayTimer = Timer(const Duration(seconds: 2), () {
        if (mounted) {
          setState(() {
            _overlayFaces.clear();
            _overlayExpressions.clear();
            _overlayColors.clear();
          });
        }
      });

      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Scan error: $e');
    } finally {
      _isProcessing = false;
      if (mounted) setState(() {});
    }
  }

  Future<void> _startContinuousScanning() async {
    if (_isScanning) return;
    _isScanning = true;
    if (mounted) setState(() {});
    try {
      while (_isScanning) {
        if (!_isProcessing) await _scanExpression();
        await Future.delayed(_scanInterval);
      }
    } finally {
      _isScanning = false;
      if (mounted) setState(() {});
    }
  }

  void _stopScanning() {
    _isScanning = false;
    _resetEmotionHistory();
    if (mounted) setState(() {});
  }

  // ─────────────────── ML Kit ───────────────────────────────

  Future<List<DetectedFace>> _detectFaces(Uint8List imageBytes) async {
    try {
      final faces = await _faceDetector.detectFaces(imageBytes);
      return faces
          .map((f) => DetectedFace(
                x: f.boundingBox.left.toDouble(),
                y: f.boundingBox.top.toDouble(),
                width: f.boundingBox.width.toDouble(),
                height: f.boundingBox.height.toDouble(),
                confidence: 1.0,
                expression: f.expression,
                poseX: f.headEulerAngleY,
                poseY: f.headEulerAngleZ,
              ))
          .toList();
    } catch (e) {
      debugPrint('Face detection error: $e');
      return [];
    }
  }

  // ─────────────────── Helpers ──────────────────────────────

  Color _colorForExpression(String expr) {
    // FER-2013 seven-class emotion labels
    switch (expr) {
      case 'Analyzing':
        return const Color(0xFF9E9E9E);
      case 'Angry':
        return const Color(0xFFF44336); // Red
      case 'Disgust':
        return const Color(0xFF795548); // Brown
      case 'Fear':
        return const Color(0xFF9C27B0); // Purple
      case 'Happy':
        return const Color(0xFF4CAF50); // Green
      case 'Sad':
        return const Color(0xFF2196F3); // Blue
      case 'Surprise':
        return const Color(0xFFFF9800); // Orange
      case 'Neutral':
        return const Color(0xFF607D8B); // Blue-grey
      // Legacy ML Kit labels (fallback)
      case 'Smiling':
        return const Color(0xFF8BC34A);
      case 'Serious':
        return const Color(0xFF455A64);
      case 'Winking':
        return const Color(0xFFFFEB3B);
      case 'Eyes Closed':
        return const Color(0xFF9E9E9E);
      default:
        return const Color(0xFF9E9E9E);
    }
  }

  IconData _iconForExpression(String expr) {
    // FER-2013 seven-class emotion labels
    switch (expr) {
      case 'Analyzing':
        return Icons.hourglass_top_rounded;
      case 'Angry':
        return Icons.sentiment_very_dissatisfied;
      case 'Disgust':
        return Icons.sick;
      case 'Fear':
        return Icons.psychology_alt;
      case 'Happy':
        return Icons.sentiment_very_satisfied;
      case 'Sad':
        return Icons.sentiment_dissatisfied;
      case 'Surprise':
        return Icons.emoji_emotions;
      case 'Neutral':
        return Icons.sentiment_neutral;
      // Legacy ML Kit labels (fallback)
      case 'Smiling':
        return Icons.sentiment_satisfied;
      case 'Serious':
        return Icons.face;
      case 'Winking':
        return Icons.face_retouching_natural;
      case 'Eyes Closed':
        return Icons.visibility_off;
      default:
        return Icons.face;
    }
  }

  // ─────────────────── Overlay ──────────────────────────────
  // Reads the actual rendered size of the camera container via
  // GlobalKey instead of LayoutBuilder.

  Size _getCameraDisplaySize() {
    final rb = _cameraKey.currentContext?.findRenderObject() as RenderBox?;
    return rb?.size ?? Size.zero;
  }

  _EmotionDisplayDecision _rawDecision(EmotionDetectionResult result) {
    return _EmotionDisplayDecision(
      label: result.label,
      confidence: result.confidence,
      accepted: result.confidence >= _minStableConfidence,
      statusText: result.isFallback
          ? 'Fallback heuristics active'
          : 'Paper emotion pipeline active',
    );
  }

  _EmotionDisplayDecision _stabilizeEmotion(EmotionDetectionResult result) {
    final now = DateTime.now();
    _recentEmotionSamples.add(
      _TemporalEmotionSample(
        probabilities: Map<String, double>.from(result.probabilities),
        rawLabel: result.label,
        confidence: result.confidence,
        isFallback: result.isFallback,
      ),
    );

    if (_recentEmotionSamples.length > _stabilityWindowSize) {
      _recentEmotionSamples.removeAt(0);
    }

    final aggregate = <String, double>{};
    double totalWeight = 0.0;
    for (int index = 0; index < _recentEmotionSamples.length; index++) {
      final sample = _recentEmotionSamples[index];
      final weight = (index + 1) * max(sample.confidence, 0.35);
      totalWeight += weight;
      sample.probabilities.forEach((label, probability) {
        aggregate[label] = (aggregate[label] ?? 0.0) + probability * weight;
      });
    }

    if (totalWeight <= 0.0 || aggregate.isEmpty) {
      return const _EmotionDisplayDecision(
        label: 'Analyzing',
        confidence: 0.0,
        accepted: false,
        statusText: 'Collecting frames for a stable emotion',
      );
    }

    final normalized = aggregate.map(
      (label, value) => MapEntry(label, value / totalWeight),
    );
    final ranked = normalized.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    final best = ranked.first;
    final second = ranked.length > 1 ? ranked[1] : const MapEntry('', 0.0);
    final margin = best.value - second.value;
    final consensusCount = _recentEmotionSamples
        .where((sample) => sample.rawLabel == best.key)
        .length;
    final minConsensus = 1;
    final accepted = _recentEmotionSamples.length >= 1 &&
        best.value >= _minStableConfidence &&
        margin >= _minStableMargin &&
        consensusCount >= minConsensus;

    if (!accepted) {
      if (_lastStableEmotion != null &&
          _lastStableAt != null &&
          now.difference(_lastStableAt!) <= _stableEmotionHold &&
          best.value >= _holdStableConfidence) {
        return _EmotionDisplayDecision(
          label: _lastStableEmotion!,
          confidence: _lastStableConfidence,
          accepted: false,
          statusText: result.isFallback
              ? 'Holding last stable fallback emotion'
              : 'Holding last stable emotion',
        );
      }

      return _EmotionDisplayDecision(
        label: 'Analyzing',
        confidence: best.value,
        accepted: false,
        statusText: result.isFallback
            ? 'Fallback active, confidence too low'
            : 'Model active, waiting for stable consensus',
      );
    }

    _lastStableEmotion = best.key;
    _lastStableConfidence = best.value;
    _lastStableAt = now;

    return _EmotionDisplayDecision(
      label: best.key,
      confidence: best.value,
      accepted: true,
      statusText: result.isFallback
          ? 'Fallback heuristics active'
          : 'Paper emotion pipeline stabilized',
    );
  }

  void _appendStableEmotionLog(String emotion) {
    final now = DateTime.now();
    _lastStableEmotion = emotion;
    _lastStableAt = now;
    if (_lastLoggedEmotion == emotion &&
        _lastLoggedAt != null &&
        now.difference(_lastLoggedAt!) < _minLogInterval) {
      return;
    }

    _lastLoggedEmotion = emotion;
    _lastLoggedAt = now;
    _expressionLog.insert(
      0,
      _ExpressionLogEntry(expression: emotion, timestamp: now),
    );
    if (_expressionLog.length > _maxLogEntries) {
      _expressionLog.removeLast();
    }
  }

  void _resetEmotionHistory() {
    _recentEmotionSamples.clear();
    _lastLoggedEmotion = null;
    _lastLoggedAt = null;
    _lastStableEmotion = null;
    _lastStableConfidence = 0.0;
    _lastStableAt = null;
  }

  Widget _buildFaceOverlay() {
    if (_overlayFaces.isEmpty || _imageSize == null) {
      return const SizedBox.shrink();
    }

    final displaySize = _getCameraDisplaySize();
    if (displaySize == Size.zero) return const SizedBox.shrink();

    return Stack(
      children: List.generate(_overlayFaces.length, (index) {
        final face = _overlayFaces[index];
        final expr = _overlayExpressions[index];
        final color = _overlayColors[index];

        final double imgW = _imageSize!.width;
        final double imgH = _imageSize!.height;

        // The CameraPreview stretches to fill (no FittedBox) so mapping
        // is a direct linear scale from image dims → display dims.
        final double scaleX = displaySize.width / imgW;
        final double scaleY = displaySize.height / imgH;

        double mappedX = face.x * scaleX;
        double mappedY = face.y * scaleY;
        double mappedW = face.width * scaleX;
        double mappedH = face.height * scaleY;

        // Mirror for front camera
        if (_currentCamera.lensDirection == CameraLensDirection.front) {
          mappedX = displaySize.width - (mappedX + mappedW);
        }

        final double centerX = mappedX + mappedW / 2;
        final double centerY = mappedY + mappedH / 2;
        final double radius = max(mappedW, mappedH) / 2;
        final double circleLeft = centerX - radius;
        final double circleTop = centerY - radius;
        final double maxLeft = max(0.0, displaySize.width - radius * 2);
        final double maxTop = max(0.0, displaySize.height - radius * 2);

        return Positioned(
          left: circleLeft.clamp(0.0, maxLeft),
          top: circleTop.clamp(0.0, maxTop),
          width: radius * 2,
          height: radius * 2,
          child: Stack(
            clipBehavior: Clip.none,
            children: [
              Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: color, width: 3),
                ),
              ),
              Positioned(
                top: -30,
                left: -20,
                right: -20,
                child: Center(
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                    decoration: BoxDecoration(
                      color: color.withAlpha(179),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(_iconForExpression(expr.split(' ').first),
                            color: Colors.white, size: 14),
                        const SizedBox(width: 4),
                        Text(
                          expr,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                          overflow: TextOverflow.ellipsis,
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

  // ─────────────────── BUILD ────────────────────────────────
  // Radically simplified: buttons inside body Column (no
  // bottomNavigationBar), camera via direct CameraPreview in
  // Stack (no FittedBox / LayoutBuilder / ExcludeSemantics).

  @override
  Widget build(BuildContext context) {
    final bool isReady =
        _controller != null && _controller!.value.isInitialized;
    final bottomPad = MediaQuery.of(context).padding.bottom;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Emotion AI'),
        elevation: 0,
        flexibleSpace: Container(
          decoration: BoxDecoration(gradient: AppConstants.blueGradient),
        ),
      ),
      body: SafeArea(
        child: Column(
          children: [
            // ── Camera ──────────────────────────────────────
            Expanded(
              flex: 3,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 12, 12, 4),
                child: Container(
                  key: _cameraKey,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: AppConstants.cardBorder, width: 2),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: !isReady
                        ? Container(
                            color: AppConstants.cardColor,
                            child: const Center(
                              child: Column(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  CircularProgressIndicator(),
                                  SizedBox(height: 12),
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
                          )
                        : Stack(
                            children: [
                              // Camera preview — fills container
                              SizedBox.expand(
                                child: CameraPreview(_controller!),
                              ),
                              // Face overlay
                              if (_overlayFaces.isNotEmpty &&
                                  _imageSize != null)
                                SizedBox.expand(child: _buildFaceOverlay()),
                              // Processing spinner
                              if (_isProcessing)
                                Positioned.fill(
                                  child: Container(
                                    color: Colors.black54,
                                    child: const Center(
                                      child: Column(
                                        mainAxisSize: MainAxisSize.min,
                                        children: [
                                          CircularProgressIndicator(
                                            valueColor:
                                                AlwaysStoppedAnimation<Color>(
                                              Colors.white,
                                            ),
                                          ),
                                          SizedBox(height: 12),
                                          Text(
                                            'Detecting...',
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
                                ),
                              // Status badge
                              Positioned(
                                top: 8,
                                right: 8,
                                child: Container(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 10, vertical: 5),
                                  decoration: BoxDecoration(
                                    color: _isScanning
                                        ? Colors.green
                                        : Colors.grey,
                                    borderRadius: BorderRadius.circular(20),
                                  ),
                                  child: Row(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      Container(
                                        width: 8,
                                        height: 8,
                                        decoration: const BoxDecoration(
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
                                          fontSize: 11,
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                              // Camera switch
                              if (_availableCameras.length > 1)
                                Positioned(
                                  top: 8,
                                  left: 8,
                                  child: Container(
                                    decoration: BoxDecoration(
                                      color: Colors.black54,
                                      shape: BoxShape.circle,
                                    ),
                                    child: IconButton(
                                      onPressed: _switchCamera,
                                      icon: const Icon(
                                        Icons.cameraswitch,
                                        color: Colors.white,
                                        size: 22,
                                      ),
                                    ),
                                  ),
                                ),
                            ],
                          ),
                  ),
                ),
              ),
            ),

            // ── Expression Log ──────────────────────────────
            Expanded(
              flex: 2,
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 4, 12, 4),
                child: Container(
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: AppConstants.cardBorder),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: Column(
                      children: [
                        // Header
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.symmetric(
                              horizontal: 16, vertical: 10),
                          decoration: BoxDecoration(
                            color: AppConstants.primaryColor.withAlpha(15),
                            border: Border(
                              bottom:
                                  BorderSide(color: AppConstants.cardBorder),
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(Icons.history,
                                      size: 16,
                                      color: AppConstants.primaryColor),
                                  const SizedBox(width: 8),
                                  const Text(
                                    'Emotion Log',
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: 13,
                                      color: AppConstants.textPrimary,
                                    ),
                                  ),
                                  const Spacer(),
                                  Text(
                                    '${_expressionLog.length} detected',
                                    style: const TextStyle(
                                      fontSize: 11,
                                      color: AppConstants.textTertiary,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 8),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 10,
                                  vertical: 6,
                                ),
                                decoration: BoxDecoration(
                                  color: (_emotionModelReady
                                          ? AppConstants.successColor
                                          : AppConstants.warningColor)
                                      .withAlpha(25),
                                  borderRadius: BorderRadius.circular(999),
                                  border: Border.all(
                                    color: _emotionModelReady
                                        ? AppConstants.successColor.withAlpha(90)
                                        : AppConstants.warningColor.withAlpha(90),
                                  ),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    Icon(
                                      _emotionModelReady
                                          ? Icons.model_training
                                          : Icons.warning_amber_rounded,
                                      size: 14,
                                      color: _emotionModelReady
                                          ? AppConstants.successColor
                                          : AppConstants.warningColor,
                                    ),
                                    const SizedBox(width: 6),
                                    Flexible(
                                      child: Text(
                                        _pipelineStatusText,
                                        style: TextStyle(
                                          fontSize: 11,
                                          fontWeight: FontWeight.w600,
                                          color: _emotionModelReady
                                              ? AppConstants.successColor
                                              : AppConstants.warningColor,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),
                        // List
                        Expanded(
                          child: _expressionLog.isEmpty
                              ? const Center(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(Icons.face,
                                          size: 48,
                                          color: AppConstants.textTertiary),
                                      SizedBox(height: 8),
                                      Text(
                                        'Start scanning to detect emotions',
                                        style: TextStyle(
                                          color: AppConstants.textSecondary,
                                          fontSize: 13,
                                        ),
                                      ),
                                    ],
                                  ),
                                )
                              : ListView.separated(
                                  itemCount: _expressionLog.length,
                                  separatorBuilder: (_, __) => Container(
                                    height: 1,
                                    color: AppConstants.cardBorder,
                                  ),
                                  itemBuilder: (context, index) {
                                    final entry = _expressionLog[index];
                                    final color =
                                        _colorForExpression(entry.expression);
                                    return ListTile(
                                      dense: true,
                                      leading: CircleAvatar(
                                        radius: 18,
                                        backgroundColor: color.withAlpha(30),
                                        child: Icon(
                                          _iconForExpression(
                                              entry.expression),
                                          color: color,
                                          size: 20,
                                        ),
                                      ),
                                      title: Text(
                                        entry.expression,
                                        style: TextStyle(
                                          fontWeight: FontWeight.w600,
                                          fontSize: 14,
                                          color: color,
                                        ),
                                      ),
                                      trailing: Text(
                                        '${entry.timestamp.hour.toString().padLeft(2, '0')}:'
                                        '${entry.timestamp.minute.toString().padLeft(2, '0')}:'
                                        '${entry.timestamp.second.toString().padLeft(2, '0')}',
                                        style: const TextStyle(
                                          fontSize: 11,
                                          color: AppConstants.textTertiary,
                                        ),
                                      ),
                                    );
                                  },
                                ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),

            // ── Buttons (inside body, NOT bottomNavigationBar) ──
            Padding(
              padding: EdgeInsets.fromLTRB(12, 4, 12, 8 + bottomPad),
              child: SizedBox(
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed:
                      _isScanning ? _stopScanning : _startContinuousScanning,
                  icon:
                      Icon(_isScanning ? Icons.stop_circle : Icons.videocam),
                  label: Text(
                    _isScanning ? 'Stop Scanning' : 'Start Scanning',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: _isScanning
                        ? AppConstants.warningColor
                        : AppConstants.primaryColor,
                    padding: const EdgeInsets.symmetric(vertical: 14),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────
class _ExpressionLogEntry {
  final String expression;
  final DateTime timestamp;
  _ExpressionLogEntry({required this.expression, required this.timestamp});
}

class _TemporalEmotionSample {
  final Map<String, double> probabilities;
  final String rawLabel;
  final double confidence;
  final bool isFallback;

  const _TemporalEmotionSample({
    required this.probabilities,
    required this.rawLabel,
    required this.confidence,
    required this.isFallback,
  });
}

class _EmotionDisplayDecision {
  final String label;
  final double confidence;
  final bool accepted;
  final String statusText;

  const _EmotionDisplayDecision({
    required this.label,
    required this.confidence,
    required this.accepted,
    required this.statusText,
  });
}
