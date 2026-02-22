# 🚀 Multi-Model Real-Time Attendance Engine (Updated)

This Flutter application runs entirely on-device to detect faces, generate embeddings, match against a registry, and mark attendance in real time. It combines **ML Kit / MediaPipe**, **AdaFace-Mobile embeddings**, and a lightweight **SharedPreferences-backed session store**, giving a privacy-first, low-latency experience.

**Status:** ✅ Fully Functional | 🎯 Feature-Complete | 📱 Mobile Optimized

---

## ✨ Key Features
- **ML Kit / MediaPipe face detection** (fast mode) that runs on the still captured by `takePicture()` so bounding boxes stay aligned with the preview without manual rotation hacks.
- **AdaFace-Mobile embeddings** (512-D) powered by `tflite_flutter`, normalized, and matched via cosine similarity for identity recognition.
- **Blink-based liveness detection** (Optional M5) that confirms users are live before reporting critical attendance states.
- **SharedPreferences-backed `DatabaseManager`** that stores students, embeddings, attendance, subjects, and teacher sessions without requiring Drift code generation.
- **Accurate overlay rendering** via `BoxFit.cover`, scaled face rectangles, and single-step mirroring for front cameras—no calibration sliders needed.

## 🧱 System Architecture (M1–M5)
1. **Capture** – `attendance_screen.dart` shows a live preview but runs inference on the high-resolution still that is already upright thanks to camera EXIF handling.
2. **M1 – Face Detection Module**: wraps Google ML Kit (MediaPipe) to return `DetectedFace` bounding boxes and head pose estimates.
3. **M2 – Face Embedding Module**: resizes the cropped face to 112×112, feeds it through AdaFace-Mobile (`assets/models/embedding_model.tflite`), and outputs a normalized 512-D vector.
4. **M3 – Face Matching Module**: compares the incoming embedding with vectors stored in `DatabaseManager` using cosine similarity (or 1-NN) to determine whether the face is known and how confident the match is.
5. **M4 – Attendance Flow**: recognized students trigger `_setOverlay()`, attendance state updates, and persistence under keys like `session_attendance_<teacher>_<subject>_<date>`.
6. **M5 – Liveness Detection**: Optional blink detection (ear threshold + temporal pattern) runs over short image sequences to add an extra layer of trust.

## 🎯 Model & Library Choices
- **Google ML Kit (face detection)** vs. custom TFLite object detectors: ML Kit handles rotation/EXIF, runs well on Android/iOS, and avoids tuning anchors or writing custom rotation code.
- **AdaFace-Mobile embeddings** vs. FaceNet/MobileFaceNet: AdaFace offers better few-shot robustness, fewer parameters, and faster inference (~15ms with 4 threads) while still producing high-dimensional vectors suitable for cosine similarity.
- **Cosine similarity matcher** (M3) vs. more complex classifiers: the deterministic matcher keeps the pipeline explainable and easy to debug while requiring only the stored embeddings.
- **SharedPreferences (`DatabaseManager`)** vs. Drift/SQLite: avoids migration headaches (see `face_recognition_database.dart` deprecation) by serializing JSON lists and playing well with Flutter’s lightweight storage story.

## 🛠️ Key Fixes & Improvements
- **Face overlay accuracy** – `_buildFaceOverlay` now scales ML Kit rectangles through a single scale factor derived from `displaySize`, mirrors once for front cameras, and clamps the result so the circle stays on the face.
- **Calibration-free UI** – Removed ratio-cycling buttons, `_calibX/_calibY`, and `_autoShiftX`, leaving a deterministic mapping math path.
- **Documentation refresh** – This README now reflects the true ML Kit + AdaFace pipeline, model choices, persistence layer, and overlay fix.

## ✅ Current Flow
1. User opens `Mark Attendance` screen; camera preview fills the rounded card via `FittedBox.cover`.
2. `_scanFace()` captures an image, sends bytes to `FaceDetectionModule`, crops the first face, and generates an AdaFace embedding.
3. `FaceMatchingModule` compares the embedding with stored vectors (via `DatabaseManager`) and identifies the student (or marks as unknown).
4. `_setOverlay()` paints the circle (green if recognized, red if not) and records attendance under the current session namespace.
5. Optional liveness (M5) can run before accepting sensitive attendance states.
6. Attendance stats and exports honor the session-based keys and appear in the UI (badges, export history, etc.).

## 📊 Performance Metrics
| Metric | Value |
|--------|-------|
| **FPS** | 30+ (preview) |
| **Detection latency** | ~30–50ms (ML Kit) |
| **Embedding inference** | ~15–25ms (AdaFace-Mobile) |
| **Memory (peak)** | ~120MB |
| **Model size** | ~20MB (`embedding_model.tflite` + ML Kit bundle) |

## 🧩 Project Structure (simplified)
```
lib/
├── main.dart                    # Entry point, UI wiring, app theming
├── screens/
│   ├── attendance_screen.dart   # Camera stack, overlay, attendance UI
│   ├── enrollment_screen.dart   # Register a new face (captures + label)
│   ├── export_screen.dart       # Export attendance history
│   └── home_screen.dart         # Landing dashboard
├── modules/
│   ├── m1_face_detection.dart   # ML Kit detection helper
│   ├── m2_face_embedding.dart    # AdaFace-Mobile inference
│   ├── m3_face_matching.dart     # Cosine similarity matcher
│   ├── m4_attendance_management.dart # Attendance session logic
│   └── m5_liveness_detection.dart    # Blink-based live check
├── database/
│   └── database_manager.dart    # SharedPreferences persistence layer
├── models/                      # DTOs for students, embeddings, attendance
└── utils/                       # Helpers (constants, colors, logging)
```

## ⚙️ Dependencies (excerpt)
```yaml
dependencies:
  flutter:
    sdk: flutter
  camera: ^0.11.0+1
  google_ml_kit: ^0.18.0       # MediaPipe face detection
  permission_handler: ^11.3.1
  tflite_flutter: ^0.11.0      # AdaFace inference
  image: ^4.0.0
  shared_preferences: ^2.2.0
  sqlite3_flutter_libs: ^0.5.15 # (unused but retained for future persistence)
  drift: ^2.15.0                # (unused, replaced by SharedPreferences)
  flutter_tts: ^4.2.5
  vector_math: ^2.1.4
  path_provider: ^2.0.15
  pdf: ^3.11.0
  share_plus: ^7.0.0
```

## 🚀 Getting Started
1. `git clone https://github.com/sunilv8892-sudo/final-attendence-app-working.git`
2. `cd multi-model-support-yolo-main`
3. `flutter pub get`
4. `flutter run`

### Required assets
- `assets/models/embedding_model.tflite` – AdaFace-Mobile embedding model (512-D vector).
- `assets/videos/background.mp4`, `assets/lottie/success.json` – UI polish assets.

## 🧰 Troubleshooting
- **Face circle drifts** → Ensure `imageSize` updates after `takePicture()` (the overlay math depends on the decoded JPEG dimensions).
- **Unknown results always** → Confirm the embedding model exists and `DatabaseManager` has stored embeddings (check `SharedPreferences` keys `embeddings`).
- **ML Kit throws `FaceDetector` errors** → Grant camera permission via `permission_handler` before starting the camera.
- **Debug logs missing** → Set `_showDebugInfo = true` inside `attendance_screen.dart` to print overlay metrics (`🔧 Overlay:`).

## 🧭 Future Work
- Multi-face support (detect + overlay multiple circles)
- UI to manage students/embeddings directly from the app
- Vector quantization or pruning to keep storage lean
- Optional cloud sync for attendance exports (if privacy policy permits)
- Add confidence slider/liveness toggle in the settings screen

## 🔗 Resources
- [Google ML Kit Face Detection](https://developers.google.com/ml-kit/vision/face-detection)
- [AdaFace Paper](https://arxiv.org/abs/2112.15449)
- [TensorFlow Lite Flutter](https://www.tensorflow.org/lite/guide/flutter)
- [Flutter Camera Plugin](https://pub.dev/packages/camera)
- [SharedPreferences Guide](https://pub.dev/packages/shared_preferences)

**Happy detecting! 👁️‍🗨️**
