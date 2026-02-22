import 'dart:async';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';

import '../models/face_detection_model.dart';
import '../modules/m1_face_detection.dart' as face_detection_module;
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
  CameraController? _controller;
  late face_detection_module.FaceDetectionModule _faceDetector;
  List<CameraDescription> _availableCameras = [];
  late CameraDescription _currentCamera;

  bool _isProcessing = false;
  bool _isScanning = false;

  // Overlay
  final List<DetectedFace> _overlayFaces = [];
  final List<String> _overlayExpressions = [];
  final List<Color> _overlayColors = [];
  Size? _imageSize;
  Timer? _overlayTimer;

  // Log
  final List<_ExpressionLogEntry> _expressionLog = [];
  static const int _maxLogEntries = 50;

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
    super.dispose();
  }

  // ─────────────────── Init ─────────────────────────────────

  Future<void> _initialize() async {
    try {
      _faceDetector = face_detection_module.FaceDetectionModule();
      await _faceDetector.initialize();
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
        debugPrint('❌ No face detected');
        return;
      }

      _imageSize = Size(
        rawImage.width.toDouble(),
        rawImage.height.toDouble(),
      );

      final validFaces =
          detections.where((f) => f.width >= 60 && f.height >= 60).toList();
      if (validFaces.isEmpty) return;

      _overlayFaces.clear();
      _overlayExpressions.clear();
      _overlayColors.clear();

      for (final face in validFaces) {
        final expr = face.expression.isNotEmpty ? face.expression : 'Neutral';
        _overlayFaces.add(face);
        _overlayExpressions.add(expr);
        _overlayColors.add(_colorForExpression(expr));

        _expressionLog.insert(
          0,
          _ExpressionLogEntry(expression: expr, timestamp: DateTime.now()),
        );
        if (_expressionLog.length > _maxLogEntries) {
          _expressionLog.removeLast();
        }
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
        await Future.delayed(const Duration(milliseconds: 500));
      }
    } finally {
      _isScanning = false;
      if (mounted) setState(() {});
    }
  }

  void _stopScanning() {
    _isScanning = false;
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
              ))
          .toList();
    } catch (e) {
      debugPrint('Face detection error: $e');
      return [];
    }
  }

  // ─────────────────── Helpers ──────────────────────────────

  Color _colorForExpression(String expr) {
    switch (expr) {
      case 'Happy':
        return const Color(0xFF4CAF50);
      case 'Smiling':
        return const Color(0xFF8BC34A);
      case 'Neutral':
        return const Color(0xFF2196F3);
      case 'Serious':
        return const Color(0xFF607D8B);
      case 'Sad':
        return const Color(0xFF9C27B0);
      case 'Winking':
        return const Color(0xFFFF9800);
      case 'Eyes Closed':
        return const Color(0xFF795548);
      default:
        return const Color(0xFF9E9E9E);
    }
  }

  IconData _iconForExpression(String expr) {
    switch (expr) {
      case 'Happy':
        return Icons.sentiment_very_satisfied;
      case 'Smiling':
        return Icons.sentiment_satisfied;
      case 'Neutral':
        return Icons.sentiment_neutral;
      case 'Serious':
        return Icons.sentiment_dissatisfied;
      case 'Sad':
        return Icons.sentiment_very_dissatisfied;
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
                        Icon(_iconForExpression(expr),
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
        title: const Text('Expression Detection'),
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
                          child: Row(
                            children: [
                              Icon(Icons.history,
                                  size: 16,
                                  color: AppConstants.primaryColor),
                              const SizedBox(width: 8),
                              const Text(
                                'Expression Log',
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
                                        'Start scanning to detect expressions',
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
