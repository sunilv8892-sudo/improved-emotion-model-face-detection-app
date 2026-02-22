import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
// 'dart:typed_data' is provided via 'package:flutter/services.dart'
import 'package:image/image.dart' as img;
import 'package:permission_handler/permission_handler.dart';
import 'package:screen_brightness/screen_brightness.dart';
import '../database/database_manager.dart';
import '../models/face_detection_model.dart';
import '../models/student_model.dart';
import '../models/embedding_model.dart';
import '../modules/m1_face_detection.dart' as face_detection_module;
import '../modules/m2_face_embedding.dart';
import '../utils/constants.dart';
import '../widgets/animated_background.dart';

class EnrollmentScreen extends StatefulWidget {
  const EnrollmentScreen({super.key});

  @override
  State<EnrollmentScreen> createState() => _EnrollmentScreenState();
}

class _EnrollmentScreenState extends State<EnrollmentScreen> {
  final _nameController = TextEditingController();
  final _rollController = TextEditingController();
  final _classController = TextEditingController();
  final _ageController = TextEditingController();
  final _phoneController = TextEditingController();
  String _selectedGender = 'Male';

  CameraController? _controller;
  late face_detection_module.FaceDetectionModule _faceDetector;
  late FaceEmbeddingModule _faceEmbedder;
  List<CameraDescription> _availableCameras = [];
  CameraDescription? _currentCamera;
  late DatabaseManager _dbManager;
  bool _isBackFlashOn = false;
  bool _isFrontLightOn = false;
  double? _previousAppBrightness;
  bool _isBrightnessBoostActive = false;

  int _capturedSamples = 0;
  final List<List<double>> _embeddings = [];
  bool _isCapturing = false;
  bool _autoCapturing = false;
  bool _embedderReady = false;

  // Performance tracking
  int _embeddingDim = 0;

  // Keyboard shortcuts & scrolling
  final ScrollController _scrollController = ScrollController();
  final FocusNode _focusNode = FocusNode();
  // Per-field focus nodes for hardware keyboard navigation
  final FocusNode _nameFocus = FocusNode();
  final FocusNode _rollFocus = FocusNode();
  final FocusNode _classFocus = FocusNode();
  final FocusNode _genderFocus = FocusNode();
  final FocusNode _ageFocus = FocusNode();
  final FocusNode _phoneFocus = FocusNode();
  final GlobalKey _nameFieldKey = GlobalKey();
  final GlobalKey _rollFieldKey = GlobalKey();
  final GlobalKey _classFieldKey = GlobalKey();
  final GlobalKey _genderFieldKey = GlobalKey();
  final GlobalKey _ageFieldKey = GlobalKey();
  final GlobalKey _phoneFieldKey = GlobalKey();
  final List<String> _genders = ['Male', 'Female', 'Other'];
  int _genderIndex = 0;

  @override
  void initState() {
    super.initState();
    _genderIndex = _genders.indexOf(_selectedGender);
    _initialize();
  }

  @override
  void dispose() {
    _disableBrightnessBoost();
    _controller?.dispose();
    _faceEmbedder.dispose();
    _faceDetector.dispose();
    _nameController.dispose();
    _rollController.dispose();
    _classController.dispose();
    _ageController.dispose();
    _phoneController.dispose();
    _scrollController.dispose();
    _focusNode.dispose();
    _nameFocus.dispose();
    _rollFocus.dispose();
    _classFocus.dispose();
    _genderFocus.dispose();
    _ageFocus.dispose();
    _phoneFocus.dispose();
    super.dispose();
  }

  Future<void> _initialize() async {
    try {
      _dbManager = DatabaseManager();
      await _dbManager.database;

      // Initialize modules
      _faceDetector = face_detection_module.FaceDetectionModule();
      await _faceDetector.initialize();

      _faceEmbedder = FaceEmbeddingModule();
      await _faceEmbedder.initialize();
      _embeddingDim = FaceEmbeddingModule.embeddingDimension;
      _embedderReady = _faceEmbedder.isReady;
      if (!_embedderReady) {
        throw Exception('FaceNet-128 interpreter failed to initialize');
      }

      debugPrint(
        '✅ Face recognition modules initialized (${_embeddingDim}D embeddings)',
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
      // Request camera permission
      final cameraStatus = await Permission.camera.request();
      if (!cameraStatus.isGranted) {
        debugPrint('❌ Camera permission denied');
        return;
      }
      _availableCameras = await availableCameras();
      if (_availableCameras.isEmpty) return;

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
      // Track current camera description for later use
      // Note: lens direction & sensor orientation not stored to keep state minimal
      // (overlay mirroring handled in attendance screen via painter parameters)
      _currentCamera = camera;

      await _controller?.dispose();
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
    if (_availableCameras.length < 2 || _currentCamera == null) return;
    final currentIndex = _availableCameras.indexOf(_currentCamera!);
    final nextIndex = (currentIndex + 1) % _availableCameras.length;
    final nextCamera = _availableCameras[nextIndex];
    await _initCameraFor(nextCamera);
  }

  bool get _isFrontCamera =>
      _currentCamera?.lensDirection == CameraLensDirection.front;

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

  Future<void> _captureFaceSample() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isCapturing) return;

    _isCapturing = true;

    try {
      final image = await _controller!.takePicture();
      final bytes = await image.readAsBytes();
      final rawImage = img.decodeImage(bytes);

      if (rawImage != null) {
        debugPrint(
          '📸 Captured image for enrollment: ${rawImage.width}x${rawImage.height}',
        );

        // Step 1: Detect face using ML Kit to ensure alignment with attendance screen
        final detections = await _detectFaceWithMlKit(bytes);

        if (detections.isEmpty) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('❌ No face detected. Please face the camera.'),
              ),
            );
          }
          return;
        }

        // Take the largest detected face
        detections.sort(
          (a, b) => (b.width * b.height).compareTo(a.width * a.height),
        );
        final face = detections.first;

        // Strict quality check for enrollment: require large, clear face
        if (face.width < 150 || face.height < 150) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(
                  '❌ Face too small! (${face.width.toInt()}x${face.height.toInt()})\n'
                  'Please move closer to camera. Need at least 150x150px',
                ),
                duration: const Duration(seconds: 2),
                backgroundColor: Colors.red,
              ),
            );
          }
          return;
        }

        // Check if face is centered (not too far to edges)
        final imageCenterX = rawImage.width / 2;
        final imageCenterY = rawImage.height / 2;
        final faceCenterX = face.x + (face.width / 2);
        final faceCenterY = face.y + (face.height / 2);
        
        final distanceFromCenter = 
            ((imageCenterX - faceCenterX).abs() + (imageCenterY - faceCenterY).abs()) / 2;
        final maxDeviation = rawImage.width * 0.25; // Allow 25% deviation from center
        
        if (distanceFromCenter > maxDeviation) {
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('❌ Face not centered. Please center your face in the frame.'),
                duration: Duration(seconds: 1),
                backgroundColor: Colors.orange,
              ),
            );
          }
          return;
        }

        debugPrint('✅ Face quality check passed: ${face.width.toInt()}x${face.height.toInt()}');

        // Step 2: Crop face (Identical logic to Attendance screen)
        final croppedFace = _cropFace(rawImage, face);

        // Step 3: Generate embedding using FaceNet-128
        final embedding = await _generateEmbedding(croppedFace);
        debugPrint('🧠 Generated embedding: ${embedding.length} dimensions');
        debugPrint('   Values: ${embedding.take(5).toList()}...');

        if (embedding.isNotEmpty) {
          _embeddings.add(embedding);
          debugPrint('✅ Added embedding #${_embeddings.length} to list');

          if (mounted) {
            setState(() => _capturedSamples++);
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('✅ Sample $_capturedSamples captured'),
                duration: const Duration(milliseconds: 500),
                backgroundColor: AppConstants.primaryColor,
              ),
            );
          }
        } else {
          debugPrint('❌ Embedding is empty!');
          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              const SnackBar(
                content: Text('❌ Failed to generate embedding'),
                backgroundColor: Colors.red,
              ),
            );
          }
        }
      }
    } catch (e) {
      debugPrint('Capture error: $e');
    } finally {
      if (mounted) setState(() => _isCapturing = false);
    }
  }

  Future<void> _startAutoCapture() async {
    if (_autoCapturing) return;
    _autoCapturing = true;
    if (mounted) setState(() {});

    try {
      while (_autoCapturing &&
          _capturedSamples < AppConstants.requiredEnrollmentSamples) {
        // If a capture is already in progress, wait a short while
        if (_isCapturing) {
          await Future.delayed(const Duration(milliseconds: 300));
          continue;
        }

        await _captureFaceSample();

        // Small delay between captures to allow user/head movement
        await Future.delayed(const Duration(milliseconds: 600));
      }
    } finally {
      _autoCapturing = false;
      if (mounted) setState(() {});
    }
  }

  void _stopAutoCapture() {
    _autoCapturing = false;
    if (mounted) setState(() {});
  }

  Future<void> _focusAndScrollTo(FocusNode? nextFocus, GlobalKey fieldKey) async {
    if (nextFocus != null) {
      FocusScope.of(context).requestFocus(nextFocus);
    } else {
      FocusScope.of(context).unfocus();
    }
  }

  Future<List<DetectedFace>> _detectFaceWithMlKit(Uint8List imageBytes) async {
    try {
      final faces = await _faceDetector.detectFaces(imageBytes);

      // Convert to legacy DetectedFace format for compatibility
      return faces
          .map(
            (face) => DetectedFace(
              x: face.boundingBox.left.toDouble(),
              y: face.boundingBox.top.toDouble(),
              width: face.boundingBox.width.toDouble(),
              height: face.boundingBox.height.toDouble(),
              confidence: 1.0,
            ),
          )
          .toList();
    } catch (e) {
      debugPrint('Face detection error: $e');
      return [];
    }
  }

  // Enrollment uses ML Kit for detection; NMS not required
  img.Image _cropFace(img.Image fullImage, DetectedFace face) {
    final x = face.x.toInt().clamp(0, fullImage.width - 1);
    final y = face.y.toInt().clamp(0, fullImage.height - 1);
    final w = face.width.toInt().clamp(1, fullImage.width - x);
    final h = face.height.toInt().clamp(1, fullImage.height - y);
    debugPrint('Cropping face at ($x, $y) with size ($w x $h) from ${fullImage.width}x${fullImage.height}');
    return img.copyCrop(fullImage, x: x, y: y, width: w, height: h);
  }

  Future<List<double>> _generateEmbedding(img.Image faceImage) async {
    try {
      debugPrint('🔄 Generating embedding from face ${faceImage.width}x${faceImage.height}');
      // Convert to bytes for the embedding module
      final faceBytes = Uint8List.fromList(
        img.encodeJpg(faceImage, quality: 100),
      );
      debugPrint('   Encoded to ${faceBytes.length} bytes');
      final embedding = await _faceEmbedder.generateEmbedding(faceBytes);
      if (embedding == null) {
        debugPrint('❌ Embedding generation returned null');
        return [];
      }
      if (!_faceEmbedder.isValidEmbedding(embedding)) {
        debugPrint('❌ Invalid embedding dimension: ${embedding.length} (expected ${FaceEmbeddingModule.embeddingDimension})');
        return [];
      }
      debugPrint('✅ Embedding generated: ${embedding.length}D vector');
      return embedding;
    } catch (e) {
      debugPrint('❌ Embedding generation error: $e');
      return [];
    }
  }

  Future<void> _saveStudent() async {
    if (_nameController.text.isEmpty ||
        _rollController.text.isEmpty ||
        _classController.text.isEmpty ||
        _ageController.text.isEmpty ||
        _phoneController.text.isEmpty) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(const SnackBar(content: Text('Please fill all fields')));
      return;
    }

    final age = int.tryParse(_ageController.text);
    if (age == null || age <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a valid age')),
      );
      return;
    }

    if (_embeddings.length < AppConstants.requiredEnrollmentSamples) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Need ${AppConstants.requiredEnrollmentSamples} samples. Have ${_embeddings.length}',
          ),
        ),
      );
      return;
    }

    final invalid = _embeddings.where((e) => e.length != FaceEmbeddingModule.embeddingDimension).length;
    if (invalid > 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Found $invalid invalid embedding(s). Please recapture samples.'),
        ),
      );
      return;
    }

    try {
      debugPrint('💾 Saving student with ${_embeddings.length} embeddings');
      // Insert student
      final student = Student(
        name: _nameController.text,
        rollNumber: _rollController.text,
        className: _classController.text,
        gender: _selectedGender,
        age: age,
        phoneNumber: _phoneController.text,
        enrollmentDate: DateTime.now(),
      );
      debugPrint('   Name: ${student.name}, Roll: ${student.rollNumber}, Class: ${student.className}, Gender: ${student.gender}, Age: ${student.age}, Phone: ${student.phoneNumber}');

      final studentId = await _dbManager.insertStudent(student);
      debugPrint('   ✅ Student inserted with ID: $studentId');

      // Insert embeddings
      for (var i = 0; i < _embeddings.length; i++) {
        final embedding = _embeddings[i];
        await _dbManager.insertEmbedding(
          FaceEmbedding(
            studentId: studentId,
            vector: embedding,
            captureDate: DateTime.now(),
          ),
        );
        debugPrint('   ✅ Embedding $i inserted (${embedding.length}D)');
      }
      debugPrint('✅ Student enrolled successfully!');
      
      // Verify data was actually saved
      final savedStudent = await _dbManager.getStudentById(studentId);
      if (savedStudent != null) {
        debugPrint('🔍 VERIFICATION: Student found in database!');
        debugPrint('   ID: ${savedStudent.id}');
        debugPrint('   Name: ${savedStudent.name}');
        debugPrint('   Roll: ${savedStudent.rollNumber}');
        debugPrint('   Class: ${savedStudent.className}');
        debugPrint('   Gender: ${savedStudent.gender}');
        debugPrint('   Age: ${savedStudent.age}');
        debugPrint('   Phone: ${savedStudent.phoneNumber}');
        
        // Verify embeddings were saved
        final savedEmbeddings = await _dbManager.getEmbeddingsForStudent(studentId);
        debugPrint('   Embeddings saved: ${savedEmbeddings.length}');
        for (var i = 0; i < savedEmbeddings.length; i++) {
          debugPrint('      Embedding $i: ${savedEmbeddings[i].vector.length}D');
        }
        
        final allStudents = await _dbManager.getAllStudents();
        debugPrint('   Total students in DB: ${allStudents.length}');
      } else {
        debugPrint('❌ VERIFICATION FAILED: Student NOT found after insert!');
      }

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('✅ Student enrolled successfully!')),
        );
        _stopAutoCapture();
        _nameController.clear();
        _rollController.clear();
        _classController.clear();
        setState(() {
          _capturedSamples = 0;
          _embeddings.clear();
        });
        Navigator.pop(context);
      }
    } catch (e) {
      debugPrint('❌ Save error: $e');
      if (mounted) {
        String errorMsg = 'Error: $e';
        if (e.toString().contains('UNIQUE')) {
          errorMsg = 'Roll number already exists!';
        }
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text(errorMsg)));
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final isReady = _controller != null && _controller!.value.isInitialized;
    final canCapture = isReady && _embedderReady;

    return KeyboardListener(
      focusNode: _focusNode,
      autofocus: true,
      onKeyEvent: _handleKeyEvent,
      child: WillPopScope(
        onWillPop: () async {
          if (_autoCapturing) {
            _stopAutoCapture();
          }
          return true;
        },
        child: Scaffold(
          appBar: AppBar(
            title: const Text('Enroll Student'),
            flexibleSpace: Container(
              decoration: BoxDecoration(gradient: AppConstants.blueGradient),
            ),
          ),
          bottomNavigationBar: Container(
            decoration: BoxDecoration(
              color: AppConstants.secondaryColor,
              border: Border(top: BorderSide(color: AppConstants.cardBorder)),
            ),
            padding: EdgeInsets.fromLTRB(
              AppConstants.paddingMedium,
              AppConstants.paddingSmall,
              AppConstants.paddingMedium,
              AppConstants.paddingSmall + MediaQuery.of(context).padding.bottom,
            ),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: canCapture && !_autoCapturing
                        ? _startAutoCapture
                        : (_autoCapturing ? _stopAutoCapture : null),
                    icon: Icon(_autoCapturing ? Icons.stop_circle : Icons.videocam),
                    label: Text(
                      _autoCapturing ? 'Stop Capture' : 'Start Capture',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
                const SizedBox(width: AppConstants.paddingMedium),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _capturedSamples >= AppConstants.requiredEnrollmentSamples &&
                            _embedderReady
                        ? _saveStudent
                        : null,
                    icon: const Icon(Icons.check_circle),
                    label: const Text(
                      'Save Student',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                ),
              ],
            ),
          ),
          body: AnimatedBackground(
            child: SafeArea(
              child: Padding(
                padding: const EdgeInsets.all(AppConstants.paddingMedium),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                  final cameraHeight =
                    (constraints.maxHeight * 0.48).clamp(240.0, 360.0);

                    Widget cameraSection = Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 500),
                        child: Container(
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(
                              AppConstants.borderRadiusLarge,
                            ),
                            border: Border.all(
                              color: AppConstants.cardBorder,
                              width: 2,
                            ),
                            boxShadow: [AppConstants.cardShadow],
                          ),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(
                              AppConstants.borderRadiusLarge,
                            ),
                            child: Container(
                              color: AppConstants.secondaryColor,
                              child: !isReady
                                  ? const Center(
                                      child: Column(
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        children: [
                                          CircularProgressIndicator(),
                                          SizedBox(
                                            height: AppConstants.paddingMedium,
                                          ),
                                          Text(
                                            'Initializing Camera...',
                                            style: TextStyle(
                                              color: AppConstants.textSecondary,
                                            ),
                                          ),
                                        ],
                                      ),
                                    )
                                  : Stack(
                                      fit: StackFit.expand,
                                      children: [
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
                                                  _controller!
                                                      .value
                                                      .previewSize
                                                      ?.width ??
                                                  1,
                                              child: CameraPreview(_controller!),
                                            ),
                                          ),
                                        ),
                                        if (_isFrontCamera && _isFrontLightOn)
                                          IgnorePointer(
                                            child: CustomPaint(
                                              painter:
                                                  _CenterOvalFlashMaskPainter(),
                                              child: const SizedBox.expand(),
                                            ),
                                          ),
                                        if (_availableCameras.length > 1)
                                          Positioned(
                                            left: 12,
                                            top: 12,
                                            child: Container(
                                              decoration: BoxDecoration(
                                                color: Colors.black.withAlpha(
                                                  153,
                                                ),
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
                                          left: 12,
                                          top: 72,
                                          child: Container(
                                            decoration: BoxDecoration(
                                              color: Colors.black.withAlpha(
                                                153,
                                              ),
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
                                    ),
                            ),
                          ),
                        ),
                      ),
                    );

                    Widget detailsSection = Card(
                      child: Padding(
                        padding: const EdgeInsets.all(AppConstants.paddingSmall),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                const Text(
                                  'Enrollment Progress',
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                                Text(
                                  '$_capturedSamples/${AppConstants.requiredEnrollmentSamples}',
                                  style: const TextStyle(
                                    fontSize: 13,
                                    fontWeight: FontWeight.w700,
                                    color: AppConstants.primaryColor,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(8),
                              child: LinearProgressIndicator(
                                value:
                                    _capturedSamples /
                                    AppConstants.requiredEnrollmentSamples,
                                minHeight: 8,
                                backgroundColor: AppConstants.inputFill,
                                valueColor: AlwaysStoppedAnimation(
                                  _capturedSamples >=
                                          AppConstants.requiredEnrollmentSamples
                                      ? AppConstants.successColor
                                      : AppConstants.primaryColor,
                                ),
                              ),
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            Row(
                              children: [
                                Expanded(
                                  child: TextField(
                                    key: _nameFieldKey,
                                    controller: _nameController,
                                    focusNode: _nameFocus,
                                    textInputAction: TextInputAction.next,
                                    onSubmitted:
                                        (_) => _focusAndScrollTo(
                                          _rollFocus,
                                          _rollFieldKey,
                                        ),
                                    decoration: InputDecoration(
                                      isDense: true,
                                      labelText: 'Name',
                                      prefixIcon: const Icon(
                                        Icons.person_outline,
                                      ),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(
                                          AppConstants.borderRadius,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: AppConstants.paddingSmall),
                                Expanded(
                                  child: TextField(
                                    key: _rollFieldKey,
                                    controller: _rollController,
                                    focusNode: _rollFocus,
                                    textInputAction: TextInputAction.next,
                                    onSubmitted:
                                        (_) => _focusAndScrollTo(
                                          _classFocus,
                                          _classFieldKey,
                                        ),
                                    decoration: InputDecoration(
                                      isDense: true,
                                      labelText: 'Roll Number',
                                      prefixIcon: const Icon(Icons.numbers),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(
                                          AppConstants.borderRadius,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            Row(
                              children: [
                                Expanded(
                                  child: TextField(
                                    key: _classFieldKey,
                                    controller: _classController,
                                    focusNode: _classFocus,
                                    textInputAction: TextInputAction.next,
                                    onSubmitted:
                                        (_) => _focusAndScrollTo(
                                          _genderFocus,
                                          _genderFieldKey,
                                        ),
                                    decoration: InputDecoration(
                                      isDense: true,
                                      labelText: 'Class/Section',
                                      prefixIcon: const Icon(Icons.school),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(
                                          AppConstants.borderRadius,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: AppConstants.paddingSmall),
                                Expanded(
                                  child: DropdownButtonFormField<String>(
                                    key: _genderFieldKey,
                                    value: _selectedGender,
                                    focusNode: _genderFocus,
                                    decoration: InputDecoration(
                                      isDense: true,
                                      labelText: 'Gender',
                                      prefixIcon: const Icon(Icons.wc),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(
                                          AppConstants.borderRadius,
                                        ),
                                      ),
                                    ),
                                    items:
                                        _genders
                                            .map(
                                              (value) => DropdownMenuItem<
                                                String
                                              >(
                                                value: value,
                                                child: Text(value),
                                              ),
                                            )
                                            .toList(),
                                    onChanged: (newValue) {
                                      if (newValue == null) return;
                                      setState(() {
                                        _selectedGender = newValue;
                                        _genderIndex = _genders.indexOf(
                                          newValue,
                                        );
                                      });
                                      _focusAndScrollTo(_ageFocus, _ageFieldKey);
                                    },
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: AppConstants.paddingSmall),
                            Row(
                              children: [
                                Expanded(
                                  child: TextField(
                                    key: _ageFieldKey,
                                    controller: _ageController,
                                    focusNode: _ageFocus,
                                    textInputAction: TextInputAction.next,
                                    keyboardType: TextInputType.number,
                                    onSubmitted:
                                        (_) => _focusAndScrollTo(
                                          _phoneFocus,
                                          _phoneFieldKey,
                                        ),
                                    decoration: InputDecoration(
                                      isDense: true,
                                      labelText: 'Age',
                                      prefixIcon: const Icon(
                                        Icons.calendar_today,
                                      ),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(
                                          AppConstants.borderRadius,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                const SizedBox(width: AppConstants.paddingSmall),
                                Expanded(
                                  child: TextField(
                                    key: _phoneFieldKey,
                                    controller: _phoneController,
                                    focusNode: _phoneFocus,
                                    textInputAction: TextInputAction.done,
                                    keyboardType: TextInputType.phone,
                                    onSubmitted:
                                        (_) => _focusAndScrollTo(
                                          null,
                                          _phoneFieldKey,
                                        ),
                                    decoration: InputDecoration(
                                      isDense: true,
                                      labelText: 'Phone Number',
                                      prefixIcon: const Icon(Icons.phone),
                                      border: OutlineInputBorder(
                                        borderRadius: BorderRadius.circular(
                                          AppConstants.borderRadius,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            if (!_embedderReady)
                              Padding(
                                padding: const EdgeInsets.only(
                                  top: AppConstants.paddingSmall,
                                ),
                                child: Text(
                                  'Face embedding model unavailable. Check logs.',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: AppConstants.errorLight,
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                    );

                    return Column(
                      children: [
                        SizedBox(height: cameraHeight, child: cameraSection),
                        const SizedBox(height: AppConstants.paddingMedium),
                        Flexible(
                          child: SingleChildScrollView(
                            child: Align(
                              alignment: Alignment.topCenter,
                              child: ConstrainedBox(
                                constraints: const BoxConstraints(maxWidth: 500),
                                child: detailsSection,
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
      ),
    );
  }

  void _handleKeyEvent(KeyEvent event) {
    if (event is! KeyDownEvent) return;
    debugPrint('Keyboard event: ${event.logicalKey.debugName}');
    // If gender field has focus, handle arrow keys to change selection
    if (_genderFocus.hasFocus) {
      if (event.logicalKey == LogicalKeyboardKey.arrowDown) {
        setState(() {
          _genderIndex = (_genderIndex + 1) % _genders.length;
          _selectedGender = _genders[_genderIndex];
        });
        return;
      }
      if (event.logicalKey == LogicalKeyboardKey.arrowUp) {
        setState(() {
          _genderIndex = (_genderIndex - 1 + _genders.length) % _genders.length;
          _selectedGender = _genders[_genderIndex];
        });
        return;
      }
      if (event.logicalKey == LogicalKeyboardKey.enter || event.logicalKey == LogicalKeyboardKey.numpadEnter) {
        FocusScope.of(context).nextFocus();
        return;
      }
    }

    // Enter: move focus to next field
    if (event.logicalKey == LogicalKeyboardKey.enter || event.logicalKey == LogicalKeyboardKey.numpadEnter) {
      FocusScope.of(context).nextFocus();
      return;
    }

    // NOTE: single-letter 'F' shortcut removed to avoid accidental saves

    switch (event.logicalKey) {
      case LogicalKeyboardKey.f5:
        // F5: Start/stop auto capture
        if (_autoCapturing) {
          _stopAutoCapture();
        } else {
          _startAutoCapture();
        }
        break;
      case LogicalKeyboardKey.f6:
        // F6: Scroll down
        if (_scrollController.hasClients) {
          _scrollController.animateTo(
            (_scrollController.offset + 300)
                .clamp(0.0, _scrollController.position.maxScrollExtent),
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeInOut,
          );
        }
        break;
      case LogicalKeyboardKey.f7:
        // F7: Save student
        if (_capturedSamples >= AppConstants.requiredEnrollmentSamples && _embedderReady) {
          _saveStudent();
        }
        break;
      case LogicalKeyboardKey.f8:
        // F8: Open enrollment options (same behavior as the AppBar keyboard icon)
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('F5: Start Capture · F6: Scroll Down · F7: Save Student · F8: Enrollment Options'),
            duration: Duration(seconds: 3),
          ),
        );
        break;
      default:
        break;
    }
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
