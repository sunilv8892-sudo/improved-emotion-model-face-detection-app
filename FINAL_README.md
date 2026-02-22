# Face Recognition Attendance System — Final Technical README

## Table of Contents
1. Overview
2. What Was Implemented in This Final Version
3. End-to-End Runtime Pipeline
4. FaceNet + KNN Design (How It Works)
5. Why Threshold Exists Internally (and Why UI Control Was Removed)
6. Data Storage Model (SharedPreferences)
7. File-by-File Technical Reference
8. Enrollment Flow (Detailed)
9. Attendance Flow (Detailed)
10. Security / False-Positive Hardening
11. Build, Run, Test Commands
12. Operational Notes for Real-World Use
13. Known Non-Blocking Analyzer Warnings

---

## 1) Overview

This project is an offline Android-first attendance app using:
- **Face detection**: Google ML Kit
- **Face embedding**: FaceNet TFLite model (`assets/models/embedding_model.tflite`)
- **Identity classification**: KNN over enrolled embeddings
- **Attendance decision**: Multi-stage verification and anti-false-positive gating

The app is now configured for a **strict 128D embedding pipeline**.

---

## 2) What Was Implemented in This Final Version

### Core model/matching updates
- Switched pipeline to **FaceNet-128D embeddings**.
- Removed cosine-based matching path from active attendance decision flow.
- Implemented **KNN weighted voting** (k=5) using Euclidean distance.
- Added **second-stage verification** against candidate student’s saved embeddings.
- Added stronger unknown rejection checks:
  - minimum vote/support constraints,
  - ambiguity margin checks,
  - top-k verification consistency checks.

### Threshold behavior updates
- Removed user-facing threshold adjustment UI from Settings.
- Locked runtime matching threshold to a stable internal value.
- Kept internal threshold logic (required in open-set face recognition).

### Stability updates
- Added guards for camera dispose/scan loop interactions.
- Tightened dimension checks and invalid embedding rejection during enrollment.

---

## 3) End-to-End Runtime Pipeline

### Enrollment pipeline
1. Camera capture.
2. ML Kit detects face.
3. Face crop extraction.
4. FaceNet model generates embedding.
5. Embedding must be **exactly 128D** (strict validation).
6. Embedding saved in storage and linked to student.

### Attendance pipeline
1. Capture + detect face.
2. Generate 128D query embedding.
3. Build/query KNN training set from enrolled embeddings.
4. Run KNN weighted voting to get candidate student.
5. Verify candidate against candidate’s own saved templates.
6. Apply anti-ambiguity and support checks.
7. Mark present only if all checks pass.

---

## 4) FaceNet + KNN Design (How It Works)

### “Training” in this app
There is no gradient-based training step. KNN is memory-based:
- **Training set = enrolled student embeddings**.
- Each enrollment sample is a labeled vector (`studentId`, `embedding[128]`).

### KNN inference
For each query embedding:
- Compute Euclidean distance to each training sample.
- Convert distance to similarity-like score: `sim = 1 / (1 + distance)`.
- Take top-k nearest neighbors (`k=5`).
- Perform weighted vote (`weight = 1 / (distance + epsilon)`).

### Candidate verification stage
After KNN predicts student X:
- Compare query with X’s own saved embeddings.
- Require strong best match + sufficient support among templates.
- Reject if not consistent.

This two-stage approach is designed to reduce false positives for unknown faces.

---

## 5) Why Threshold Exists Internally (and Why UI Control Was Removed)

### Why threshold is necessary
Without a threshold, KNN always returns someone even for unknown faces.
Threshold is required to operate in **open-set** mode (unknown persons must be rejected).

### What changed
- Threshold slider/settings was removed from user UI.
- A fixed internal threshold is used for stable behavior.
- This avoids accidental misconfiguration while still allowing robust unknown rejection.

---

## 6) Data Storage Model (SharedPreferences)

Primary keys used:
- `students` (list of JSON records)
- `embeddings` (list of JSON records)
- `attendance` (list of JSON records)
- `subjects`
- `teacherSessions`
- `tts_enabled`

Each embedding record stores:
- embedding id
- student id
- vector values
- capture date

---

## 7) File-by-File Technical Reference

## Entry Points
- `lib/main.dart`
  - Main app bootstrap, route wiring, theme usage.
- `lib/main_app.dart`
  - Alternate/legacy entry path retained in project.

## Database Layer
- `lib/database/database_manager.dart`
  - SharedPreferences CRUD for students, embeddings, attendance, subjects, sessions.
  - Similarity helper and retrieval methods used by screens/modules.
- `lib/database/database_connection.dart`
  - Legacy/auxiliary DB connection utilities.
- `lib/database/face_recognition_database.dart`
  - Legacy structure retained.

## Data Models
- `lib/models/student_model.dart`
  - Student schema and serialization helpers.
- `lib/models/embedding_model.dart`
  - Face embedding entity.
- `lib/models/attendance_model.dart`
  - Attendance record and status enum.
- `lib/models/subject_model.dart`
  - Subject and teacher session entities.
- `lib/models/face_detection_model.dart`
  - Normalized detected face representation.
- `lib/models/match_result_model.dart`
  - Matching result envelope.

## ML Modules
- `lib/modules/m1_face_detection.dart`
  - ML Kit wrapper for face detection from bytes/path.
- `lib/modules/m2_face_embedding.dart`
  - FaceNet TFLite inference module.
  - Strict model checks:
    - input shape derived from tensor metadata,
    - output dimension must be 128.
  - Embedding normalization.
- `lib/modules/m3_face_matching.dart`
  - Standalone matching utilities (KNN/euclidean).
- `lib/modules/m4_attendance_management.dart`
  - Attendance report/statistics logic.
- `lib/modules/m5_liveness_detection.dart`
  - Liveness utility module (not primary gate in current flow).

## Screens
- `lib/screens/home_screen.dart`
  - App dashboard and navigation.
- `lib/screens/enrollment_screen.dart`
  - Enrollment UI, camera capture, embedding capture/validation, save logic.
- `lib/screens/attendance_prep_screen.dart`
  - Teacher/subject pre-attendance flow.
- `lib/screens/attendance_screen.dart`
  - Real-time attendance loop.
  - KNN training-set build and two-stage match verification.
- `lib/screens/database_screen.dart`
  - Attendance and data browsing.
- `lib/screens/export_screen.dart`
  - Data export screen.
- `lib/screens/export_screen_backup.dart`
  - Backup/older export implementation retained.
- `lib/screens/settings_screen.dart`
  - TTS and data-management controls.
  - Threshold UI removed in this final version.
- `lib/screens/attendance_screen_stub.dart`
  - Supporting/placeholder screen retained.
- `lib/screens/expression_detection_screen.dart`
  - Expression-detection related UI retained.

## Utilities
- `lib/utils/constants.dart`
  - Theme constants, strings, app constants.
- `lib/utils/csv_export_service.dart`
  - CSV export helpers.
- `lib/utils/export_utils.dart`
  - File export path and platform utilities.
- `lib/utils/theme.dart`
  - App theming utilities.

## Widgets
- `lib/widgets/animated_background.dart`
  - Shared animated background/glass containers.

## Assets
- `assets/models/embedding_model.tflite`
  - FaceNet embedding model file used by app.
- `assets/lottie/success.json`
  - Success animation asset.
- `assets/icons/*`
  - App icons.

---

## 8) Enrollment Flow (Detailed)

`EnrollmentScreen` performs:
1. Validate student form fields.
2. Capture camera frame.
3. Detect face using ML Kit.
4. Apply quality checks (size/position).
5. Crop face region.
6. Generate embedding through `FaceEmbeddingModule`.
7. Enforce `embedding.length == 128`.
8. Repeat until required sample count.
9. Save student + all embeddings.

If embedding generation fails or dimension mismatch occurs, sample is rejected.

---

## 9) Attendance Flow (Detailed)

`AttendanceScreen` performs:
1. Load enrolled students + valid 128D embeddings.
2. Build in-memory KNN sample list (`studentId`, vector).
3. Continuous scanning loop:
   - capture,
   - detect,
   - embed,
   - classify via KNN,
   - verify candidate templates,
   - apply cooldown + consecutive confirmation,
   - mark attendance.

### Decision gates before marking present
- KNN candidate exists.
- Candidate exceeds internal threshold.
- Candidate not ambiguous vs second best.
- Verification support from own templates passes.
- Top-k verification consistency passes.

---

## 10) Security / False-Positive Hardening

Implemented protections:
- Strict 128D embedding acceptance.
- Multi-neighbor KNN voting (not single nearest only).
- Distance-weighted voting.
- Candidate-only template verification.
- Minimum support checks.
- Ambiguity margin checks.
- Repeated frame confirmation and cooldown.

Goal: **mark enrolled faces reliably while minimizing unknown false accepts**.

---

## 11) Build, Run, Test Commands

```bash
flutter pub get
flutter analyze
flutter test
flutter build apk --debug
flutter run -d <device-id>
```

Install to specific phone:

```bash
flutter install -d ZXFAZ5855XXCJNAI --debug
```

---

## 12) Operational Notes for Real-World Use

- Re-enroll students if you change embedding model version or preprocessing.
- Keep consistent enrollment conditions (distance/lighting/pose).
- Use multiple enrollment samples per student from slightly varied angles.
- Unknown-face rejection is always a balance; current pipeline is tuned for safety.

---

## 13) Known Non-Blocking Analyzer Warnings

Current analyzer output contains mostly style/deprecation/unused-symbol warnings in non-critical files. These do not block APK build or runtime launch. They can be cleaned in a separate quality pass without changing the matching pipeline behavior.

---

## Final Status

This codebase now runs with:
- **FaceNet embedding + KNN classification**
- **strict 128D constraints**
- **multi-stage attendance verification**
- **settings threshold control removed from UI**
- successful Android build/install/run flow.
