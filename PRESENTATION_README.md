# FaceAttend — Complete System Documentation

## For Presentation & Guide Reference

> **App Name**: Face Recognition Attendance System  
> **Version**: 18.4.0  
> **Platform**: Flutter (Android / iOS / Desktop)  
> **All ML runs on-device** — no cloud, no internet required  
> **Research Paper**: "Synergistic Feature Fusion Approach Towards Real-Time Children Facial Emotion Recognition" — Shivaprasad D L & D S Guru

---

## Table of Contents

1. [What This App Does](#1-what-this-app-does)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [App Screens & User Flow](#4-app-screens--user-flow)
5. [Module-by-Module Deep Dive](#5-module-by-module-deep-dive)
   - [M1: Face Detection](#m1-face-detection)
   - [M2: Face Embedding](#m2-face-embedding)
   - [M3: Face Matching](#m3-face-matching)
   - [M4: Attendance Management](#m4-attendance-management)
   - [M5: Liveness Detection](#m5-liveness-detection)
   - [M6: Emotion Detection](#m6-emotion-detection)
6. [Data Models](#6-data-models)
7. [Database Layer](#7-database-layer)
8. [The Emotion Recognition Pipeline — In Depth](#8-the-emotion-recognition-pipeline--in-depth)
9. [How Face Recognition Works](#9-how-face-recognition-works)
10. [CSV Export System](#10-csv-export-system)
11. [File Structure Reference](#11-file-structure-reference)
12. [Problems We Solved](#12-problems-we-solved)
13. [Potential Guide Questions & Answers](#13-potential-guide-questions--answers)

---

## 1. What This App Does

**FaceAttend** is an offline mobile application that automates classroom attendance using face recognition and simultaneously detects student emotions. A teacher simply points the phone camera at students — the app automatically:

1. **Detects faces** in the camera frame using Google ML Kit
2. **Identifies who each face belongs to** by comparing face embeddings against enrolled students
3. **Records attendance** (Present / Absent) with timestamp
4. **Detects the student's emotion** (Happy, Sad, Angry, Neutral, Surprise, Disgust) using a machine learning pipeline
5. **Exports attendance reports** as CSV files with emotion data included

Everything runs **entirely on the device** — no internet, no cloud APIs, no server. This makes it fast, private, and usable in areas with no connectivity.

### Key Features
- **Student Enrollment**: Register students with name, photo (face embeddings), roll number, class
- **Face-Based Attendance**: Point camera → auto-detect → auto-identify → auto-record
- **Emotion Recognition**: Real-time emotion classification using EfficientNet + HOG + SVM
- **Text-to-Speech**: Announces student names when recognized
- **CSV Export**: Subject-wise attendance reports with emotion column
- **Database Viewer**: Browse all students, attendance records, statistics
- **Backup/Restore**: Export and import all data as JSON

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FaceAttend Application                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐              │
│  │  Home Screen  │───▶│ Enrollment   │    │  Attendance  │              │
│  │  (Dashboard)  │    │  Screen      │    │  Screen      │              │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘              │
│         │                   │                    │                       │
│         ▼                   ▼                    ▼                       │
│  ┌────────────────────────────────────────────────────────┐            │
│  │                    MODULE LAYER                         │            │
│  │                                                         │            │
│  │  M1: Face        M2: Face         M3: Face              │            │
│  │  Detection       Embedding        Matching              │            │
│  │  (ML Kit)        (FaceNet)        (KNN+Euclidean)       │            │
│  │                                                         │            │
│  │  M4: Attendance  M5: Liveness     M6: Emotion           │            │
│  │  Management      Detection        Detection             │            │
│  │  (CRUD+Stats)    (Blink EAR)      (EFN+HOG+SVM)        │            │
│  └────────────────────────┬───────────────────────────────┘            │
│                           │                                             │
│  ┌────────────────────────▼───────────────────────────────┐            │
│  │                    DATA LAYER                           │            │
│  │                                                         │            │
│  │  DatabaseManager      Models         CSV Export          │            │
│  │  (SharedPreferences)  (Student,      (Attendance         │            │
│  │                        Embedding,     Reports)           │            │
│  │                        Attendance)                       │            │
│  └────────────────────────────────────────────────────────┘            │
│                                                                         │
│  ┌────────────────────────────────────────────────────────┐            │
│  │                    ML MODELS (On-Device)                │            │
│  │                                                         │            │
│  │  • Google ML Kit Face Detector (built-in)               │            │
│  │  • FaceNet TFLite (128-d embeddings, ~1 MB)             │            │
│  │  • EfficientNet-B0 TFLite (1000-d features, ~5.6 MB)   │            │
│  │  • SVM Classifier (JSON params, ~640 KB)                │            │
│  └────────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | Flutter (Dart) | Cross-platform mobile UI |
| **Face Detection** | Google ML Kit | Detect face bounding boxes in camera frames |
| **Face Embeddings** | FaceNet (TFLite) | Convert face images to 128-d numerical vectors |
| **Face Matching** | K-Nearest Neighbors | Compare embeddings using Euclidean distance |
| **Emotion Features** | EfficientNet-B0 (TFLite) | Extract 1000-d deep features from face images |
| **Emotion Features** | HOG (Dart implementation) | Extract 1568-d gradient histogram features |
| **Emotion Classifier** | SVM (RBF kernel) | Classify emotions from combined features |
| **Dimensionality Reduction** | LDA | Reduce 2570 features to 5 discriminant components |
| **Database** | SharedPreferences | Store students, embeddings, attendance as JSON |
| **Camera** | camera plugin | Access device camera for live face capture |
| **Text-to-Speech** | flutter_tts | Announce student names during attendance |
| **Export** | Custom CSV generator | Create attendance reports for download/sharing |

### ML Models Used

| Model | File | Input | Output | Size |
|-------|------|-------|--------|------|
| ML Kit Face Detector | Built into Google Play Services | Camera frame | Face bounding boxes + landmarks | N/A |
| FaceNet | `embedding_model.tflite` | 160×160×3 RGB | 128-d embedding vector | ~1 MB |
| EfficientNet-B0 | `efficientnet_feature_extractor.tflite` | 224×224×3 RGB | 1000-d softmax probabilities | ~5.6 MB |
| SVM + LDA + Scaler | `emotion_runtime_params.json` | 2570-d feature vector | 6 emotion probabilities | ~640 KB |

---

## 4. App Screens & User Flow

### Screen Map

```
Home Screen (Dashboard)
├── Enroll Student → EnrollmentScreen
│   └── Camera → Capture 10+ face photos → Save to DB
├── Take Attendance → AttendancePrepScreen
│   └── Select teacher/subject → AttendanceScreen
│       └── Camera → Auto-detect → Auto-identify → Record + Emotion
├── Database → DatabaseScreen
│   └── View students, attendance history, statistics
├── Export → ExportScreen
│   └── CSV files, share, backup/restore
├── Expression Detection → ExpressionDetectionScreen
│   └── Standalone camera emotion monitoring
└── Settings → SettingsScreen
    └── TTS toggle, data stats, clear data, backup/restore
```

### Screen Details

#### 1. Home Screen (`home_screen.dart`)
- **Purpose**: Main dashboard and navigation hub
- **Shows**: Total enrolled students, students present today, total sessions
- **Navigation**: Buttons to all other screens
- **Design**: Premium navy-glass theme with animated background

#### 2. Enrollment Screen (`enrollment_screen.dart`)
- **Purpose**: Register new students into the system
- **Flow**: 
  1. Enter student details (name, roll number, class, gender, age, phone)
  2. Open camera (front or back)
  3. Auto-capture 10+ face photos from different angles
  4. For each photo: detect face → crop → generate 128-d embedding
  5. Save student profile + all embeddings to database
- **Why multiple photos?**: Multiple embeddings from different angles make matching more robust. During attendance, the K-NN compares against ALL stored embeddings for each student.

#### 3. Attendance Prep Screen (`attendance_prep_screen.dart`)
- **Purpose**: Select teacher name and subject before taking attendance
- **Flow**: Enter teacher name → select/create subject → proceed to camera

#### 4. Attendance Screen (`attendance_screen.dart`)
- **Purpose**: The core attendance-taking screen with live camera
- **Flow**:
  1. Camera feed shows live preview with face bounding boxes
  2. For each detected face: crop → generate embedding → match against all stored embeddings
  3. If match found (similarity ≥ 0.75): mark student present, announce name via TTS
  4. Simultaneously: run emotion detection on the face
  5. Display student name + emotion overlay on camera preview
  6. On completion: save attendance records + generate CSV report
- **Key Feature**: Requires 2 consecutive detections of the same person to confirm identity (prevents false positives)

#### 5. Database Screen (`database_screen.dart`)
- **Purpose**: View and manage all stored data
- **Tabs**: Students list + Attendance history
- **Shows**: Per-student attendance percentage, date-wise attendance summaries

#### 6. Export Screen (`export_screen.dart`)
- **Purpose**: Manage attendance CSV files
- **Features**: List saved exports, share files, view file contents, delete exports
- **Export includes**: Teacher name, subject, date, attendee/absentee lists, emotion data

#### 7. Expression Detection Screen (`expression_detection_screen.dart`)
- **Purpose**: Standalone emotion monitoring (not tied to attendance)
- **Flow**: Camera → detect face → run emotion pipeline → show real-time emotion with confidence
- **Uses**: Stability window of 3 frames to avoid flickering between emotions

#### 8. Settings Screen (`settings_screen.dart`)
- **Purpose**: App configuration and data management
- **Features**: TTS enable/disable, view data statistics, backup data as JSON, restore from backup, clear all data

---

## 5. Module-by-Module Deep Dive

The app's ML functionality is organized into 6 modules (M1–M6), each handling one specific concern. All modules are **singletons** — they're initialized once and reused across screens.

### M1: Face Detection
**File**: `lib/modules/m1_face_detection.dart`  
**Technology**: Google ML Kit Face Detection  
**What it does**: Takes a camera frame (image bytes) and returns a list of detected faces with bounding boxes, confidence scores, and head pose angles.

**How it works**:
1. Camera captures a frame as JPEG bytes
2. Bytes are written to a temp file (ML Kit requires file path)
3. ML Kit's `FaceDetector` processes the image
4. Returns: face bounding box (x, y, width, height), confidence, pose angles (yaw X, pitch Y)

**Key settings**:
- Performance mode: Fast (for real-time camera)
- Classification: Enabled (smiling probability, eye open probability)
- Contours: Disabled (not needed, saves computation)
- Minimum face size: 10% of image

**Output**: `DetectedFace` model with bounding box, confidence, pose angles

---

### M2: Face Embedding
**File**: `lib/modules/m2_face_embedding.dart`  
**Technology**: FaceNet TFLite model (128-dimensional)  
**What it does**: Takes a cropped face image and converts it into a 128-dimensional numerical vector (embedding) that uniquely represents that person's face.

**How it works**:
1. Receive cropped face image bytes
2. Decode and resize to 160×160 RGB
3. Normalize pixel values to [0, 1]
4. Run through FaceNet TFLite model
5. L2-normalize the output embedding (unit vector)
6. Return 128-d float vector

**Why embeddings?**: Two photos of the same person produce similar vectors (small Euclidean distance). Two different people produce distant vectors (large distance). This converts the "who is this person?" question into a simple distance calculation.

**Key concept — L2 normalization**: After FaceNet produces raw 128 values, we divide each by the total vector length (L2 norm). This ensures all embeddings lie on a unit hypersphere, making distance comparisons consistent regardless of image brightness or contrast.

---

### M3: Face Matching
**File**: `lib/modules/m3_face_matching.dart`  
**Technology**: K-Nearest Neighbors with Euclidean distance  
**What it does**: Compares an incoming face embedding against all stored embeddings in the database to find who it matches.

**How it works**:
1. Receive the incoming face's 128-d embedding
2. Compute Euclidean distance to every stored embedding in the database
3. Find the nearest neighbor (smallest distance)
4. Convert distance to similarity: `similarity = 1 / (1 + distance)`
5. If similarity ≥ threshold (0.75): return "known" with student ID
6. If below threshold: return "unknown"

**Why Euclidean distance?**: After L2 normalization, Euclidean distance on unit vectors is directly related to cosine similarity. FaceNet embeddings are designed so that same-person faces are < 1.0 apart and different-person faces are > 1.0 apart.

**K-NN with K=5**: During attendance, we use K=5 nearest neighbors and take the majority vote. This is more robust than K=1 because one bad enrollment photo won't cause a mismatch.

**Similarity threshold**: 0.75 (configurable). This balances:
- Too high (0.90): would miss legitimate students with slight appearance changes
- Too low (0.50): would falsely match different people

---

### M4: Attendance Management
**File**: `lib/modules/m4_attendance_management.dart`  
**What it does**: Business logic for recording attendance, computing statistics, generating reports.

**Key functions**:
- `recordAttendance()`: Save a record, prevents duplicates (same student same day)
- `getAttendanceDetails()`: Get full stats for a student (total, present, absent, late, percentage)
- `getDailyAttendanceReport()`: All records for a specific date
- `exportAsCSV()`: Generate formatted CSV with all students and dates

---

### M5: Liveness Detection
**File**: `lib/modules/m5_liveness_detection.dart`  
**Technology**: Eye Aspect Ratio (EAR) blink detection  
**What it does**: Detects if a face is a real live person (not a printed photo or screen).

**How it works**:
1. Track face landmarks across multiple frames
2. Calculate Eye Aspect Ratio (EAR) for each frame: `EAR = (vertical1 + vertical2) / (2 × horizontal)`
3. When EAR drops below 0.3, that's a blink. When it recovers above 0.3, the blink is complete
4. Require 2 blinks within 10 seconds to confirm liveness

**Why it matters**: Without liveness detection, someone could hold up a photo of a student and trick the system.

---

### M6: Emotion Detection
**File**: `lib/modules/m6_emotion_detection.dart`  
**Technology**: EfficientNet + HOG + LDA + SVM pipeline  
**What it does**: Classifies a face image into one of 6 emotions.

**This is the most complex module — see [Section 8](#8-the-emotion-recognition-pipeline--in-depth) for the full deep dive.**

**Quick summary**:
1. Face image → EfficientNet-B0 → 1000 log-probability features
2. Face image → HOG descriptor → 1568 gradient histogram features
3. Append 2 pose features (head yaw + pitch)
4. Combine: 2570 features total
5. MinMaxScaler → LDA (reduce to 5 components) → RBF SVM (6 binary classifiers)
6. Output: emotion label + confidence + full probability map

**6 Emotions**: Angry, Disgust, Happy, Neutral, Sad, Surprise

---

## 6. Data Models

### Student (`models/student_model.dart`)
```
Student
├── id: int (auto-generated)
├── name: String
├── rollNumber: String
├── className: String
├── gender: String
├── age: int
├── phoneNumber: String
└── enrollmentDate: DateTime
```

### FaceEmbedding (`models/embedding_model.dart`)
```
FaceEmbedding
├── id: int
├── studentId: int (foreign key → Student)
├── vector: List<double> (128 values)
└── captureDate: DateTime
```
Each student has **10-15 embeddings** from different angles captured during enrollment.

### AttendanceRecord (`models/attendance_model.dart`)
```
AttendanceRecord
├── id: int
├── studentId: int (foreign key → Student)
├── date: DateTime (YYYY-MM-DD)
├── time: String (HH:MM:SS)
├── status: AttendanceStatus (present / absent / late)
├── recordedAt: DateTime
└── emotion: String? (e.g., "Happy", "Neutral")
```

### Subject (`models/subject_model.dart`)
```
Subject
├── id: int
├── name: String
└── createdAt: DateTime

TeacherSession
├── id: int
├── teacherName: String
├── subjectId: int
├── subjectName: String
├── date: DateTime
└── createdAt: DateTime
```

### DetectedFace (`models/face_detection_model.dart`)
```
DetectedFace
├── x, y, width, height: double (bounding box)
├── confidence: double (0-1)
├── expression: String
├── poseX: double? (head yaw angle)
└── poseY: double? (head pitch angle)
```

### MatchResult (`models/match_result_model.dart`)
```
MatchResult
├── identityType: String ("known" or "unknown")
├── studentId: int?
├── studentName: String?
├── similarity: double (0-1)
└── timestamp: DateTime
```

---

## 7. Database Layer

**File**: `lib/database/database_manager.dart`  
**Technology**: SharedPreferences (JSON-based key-value storage)

### Why SharedPreferences (not SQLite)?
The app uses Flutter's SharedPreferences for data persistence. Data is stored as JSON string lists under keys like `students`, `embeddings`, `attendance`, `subjects`, `teacherSessions`.

### How It Works
- Each data type (students, embeddings, etc.) is stored as a `List<String>` where each string is a JSON-encoded record
- IDs are auto-generated by finding the max existing ID + 1
- Embeddings store vectors as comma-separated doubles in a single string
- All operations are async (SharedPreferences is async on first load)

### Operations
| Operation | Students | Embeddings | Attendance |
|-----------|----------|------------|------------|
| Create | `insertStudent()` | `insertEmbedding()` | `recordAttendance()` |
| Read one | `getStudentById()` | `getEmbeddingsForStudent()` | `getAttendanceForDate()` |
| Read all | `getAllStudents()` | `getAllEmbeddings()` | `getAllAttendance()` |
| Update | `updateStudent()` | — | — |
| Delete | `deleteStudent()` | `deleteEmbeddingsForStudent()` | `deleteAttendance()` |

### Data Integrity
- Deleting a student also deletes their embeddings and attendance records
- Attendance prevents duplicates: same student + same date = rejected
- Backup/restore exports ALL SharedPreferences keys as a single JSON file

---

## 8. The Emotion Recognition Pipeline — In Depth

This is the most technically complex part of the app. It implements the research paper's approach of **"Synergistic Feature Fusion"** — combining deep learning features (EfficientNet) with classical computer vision features (HOG) for better accuracy than either alone.

### The Complete Flow

```
Face Image (camera crop, ~200×200 JPEG)
│
├──────────────────────────────────┐
│                                  │
▼                                  ▼
EfficientNet-B0 (TFLite)          HOG Feature Extractor (Dart)
│                                  │
│ Input: 224×224×3 RGB             │ Input: 256×256 grayscale
│ Model: ImageNet pre-trained      │ Algorithm: skimage.feature.hog
│ Output: 1000 softmax probs       │ Params: 32px cells, 8 bins, 2×2 blocks
│                                  │ Output: 1568-d descriptor
│ ── log(clamp(prob, 1e-7)) ──    │
│                                  │
│ 1000 features                    │ 1568 features
│                                  │
└──────────┬───────────────────────┘
           │
           │ + Pose angles (yaw, pitch) = 2 features
           │
           ▼ 2570-d combined feature vector
           │
    ┌──────▼──────┐
    │ MinMaxScaler │  Scale each feature to [0, 1]
    │ (2570 params) │  Formula: x_scaled = x * scale + min
    └──────┬──────┘
           │ 2570 scaled features
           │
    ┌──────▼──────┐
    │   LDA (5)   │  Linear Discriminant Analysis
    │  (2570→5)   │  Projects to 5 most discriminative directions
    └──────┬──────┘
           │ 5 discriminant features
           │
    ┌──────▼──────┐
    │  RBF SVM    │  6 binary One-vs-Rest classifiers
    │  (6 models) │  Each: "is this class X?" with RBF kernel
    │  C=0.05     │  Decision: argmax of all 6 scores
    └──────┬──────┘
           │
           ▼
    Emotion Label + Confidence
    e.g., "Happy" (87.3%)
```

### Stage-by-Stage Explanation

#### Stage 1: EfficientNet-B0 Feature Extraction

**What**: A pre-trained deep neural network that turns a face image into 1000 numbers.

**How**: EfficientNet-B0 was trained on ImageNet (1.2 million images, 1000 object categories). Even though it was trained to recognize objects like "cat", "dog", "car", etc., the features it produces are highly descriptive of visual content. The 1000 output values represent the probability that the image belongs to each of the 1000 ImageNet classes.

**The log transform**: The model outputs probabilities (0 to 1). But the training data was extracted using `log(probability)`, so all training features are negative numbers (-14.3 to -0.1). Without the log transform, the runtime features would be incompatible with the trained scaler and SVM.

```dart
// In emotion_feature_extractor.dart
final clamped = prob < 1e-7 ? 1e-7 : prob;
return math.log(clamped);  // [0, 1] → [-16.1, 0]
```

**Why it works for emotions**: Even though EfficientNet wasn't trained on emotions, the deep features it extracts (edges, textures, shapes, spatial patterns) are transferable to emotion recognition. A smiling face activates different ImageNet features than a frowning face.

#### Stage 2: HOG Feature Extraction

**What**: Histogram of Oriented Gradients — a classical descriptor that captures edge directions.

**How**: 
1. Resize face image to 256×256 grayscale
2. Compute gradient (edge direction + strength) at every pixel using central differences: `gx[y][x] = img[y][x+1] - img[y][x-1]`
3. Divide into 8×8 grid of 32×32-pixel cells
4. In each cell: build histogram of 8 orientation bins (0°, 22.5°, 45°, ..., 157.5°)
5. Group cells into 7×7 overlapping 2×2-cell blocks
6. Normalize each block using L2-Hys (L2 norm → clip at 0.2 → renormalize)
7. Concatenate: 7×7×2×2×8 = 1568 values

**Why it helps**: HOG captures the **physical shape** of facial features:
- Raised eyebrows (Surprise) → strong horizontal gradients above eyes
- Downturned mouth corners (Sad) → specific gradient patterns around mouth
- Furrowed brow (Angry) → dense vertical gradients between eyebrows

This is **complementary** to EfficientNet's semantic features, which is why combining them improves accuracy.

#### Stage 3: Feature Combination

Concatenate: [1000 EFN features] + [1568 HOG features] + [2 pose angles] = **2570-d vector**

The pose angles (head yaw and pitch from ML Kit) capture whether the person is looking left/right/up/down, which correlates with certain emotions (e.g., looking down while sad).

#### Stage 4: MinMaxScaler

Scale every feature to [0, 1] range using training data statistics:
```
x_scaled = x * scale_factor + min_adjustment
```
This prevents features with large ranges (like EFN log-probs: -14 to 0) from dominating features with small ranges (like pose: -30° to +30°).

#### Stage 5: LDA (Linear Discriminant Analysis)

**What**: Projects 2570 features down to just 5 features that maximally separate the 6 emotion classes.

**How**: LDA finds directions in 2570-d space where different emotion classes are most separated. With 6 classes, there can be at most 5 discriminant directions (n_classes - 1).

**Formula**: `z = (x - mean) × projection_matrix`  
Where `mean` is 2570-d and `projection_matrix` is 2570×5.

**Why 5 is enough**: LDA specifically optimizes for class separation. These 5 features encode things like:
- "How happy vs. sad does this look?"
- "How surprised vs. angry does this look?"
- etc.

#### Stage 6: RBF SVM Classification

**What**: Support Vector Machine with Radial Basis Function kernel — a classical ML classifier.

**Strategy**: One-vs-Rest (OvR) — train 6 separate binary classifiers:
1. "Is this Angry?" (Angry vs. all others)
2. "Is this Disgust?" (Disgust vs. all others)
3. "Is this Happy?" (Happy vs. all others)
4. "Is this Neutral?" (Neutral vs. all others)
5. "Is this Sad?" (Sad vs. all others)
6. "Is this Surprise?" (Surprise vs. all others)

**RBF Kernel**: `K(x,y) = exp(-γ × ||x-y||²)` — measures similarity between points in the LDA space. Points close together get kernel value near 1, distant points get kernel value near 0.

**Decision function**: For each binary model:
```
score = Σ(αᵢ × K(xᵢ, x)) + b
```
Where `xᵢ` are support vectors (boundary training samples), `αᵢ` are their weights, and `b` is the intercept.

**Final prediction**: The class with the highest score wins. Scores are converted to probabilities using softmax with temperature=0.5.

### Model Parameters (Stored in JSON)

The entire trained pipeline is stored in `assets/models/emotion_runtime_params.json` (~640 KB):

| Component | What's Stored | Size |
|-----------|--------------|------|
| Scaler | 2570 min values + 2570 scale values | ~40 KB |
| LDA | 2570-d mean + 2570×5 projection matrix | ~100 KB |
| SVM | 6 models × (support vectors + coefficients + intercept) | ~500 KB |
| Config | Labels, feature layout, gamma, class biases | ~1 KB |

---

## 9. How Face Recognition Works

### Enrollment (One-Time Setup Per Student)

```
Student stands in front of camera
│
▼ [10 photos captured from slightly different angles]
│
For each photo:
│
├── M1: Detect face → bounding box
├── Crop face from image
├── M2: Generate 128-d embedding via FaceNet
└── Save embedding to database (linked to student ID)
│
Result: Student has 10 embedding vectors in database
```

### Attendance (Real-Time Matching)

```
Camera frame captured
│
├── M1: Detect all faces → list of bounding boxes
│
For each detected face:
│
├── Crop face from frame
├── M2: Generate 128-d embedding (query)
├── M3: Compare query against ALL stored embeddings
│   │
│   ├── Compute Euclidean distance to each stored embedding
│   ├── Find K=5 nearest neighbors
│   ├── Convert distances to similarity: sim = 1/(1+dist)
│   └── If best similarity ≥ 0.75 → "Known" (student ID)
│       Else → "Unknown"
│
├── If known for 2 consecutive frames:
│   ├── M4: Record attendance (present)
│   ├── M6: Detect emotion
│   ├── TTS: Speak student name
│   └── Show overlay: name + emotion on camera preview
│
└── Mark remaining students as absent
```

### Why 2 Consecutive Detections?
To prevent false positives. If the model briefly misidentifies Student A as Student B in one frame, requiring 2 consecutive matches filters this out. Only consistent identification is accepted.

---

## 10. CSV Export System

### Attendance Report Format
```csv
Teacher Name,Subject
"John Doe","Mathematics"

Date: 2026-03-12

"Attendees = 5, Absentees = 3, Total = 8"

"Attendees","Absentees","Emotion"
"Student B","Student A","Happy"
"Student D","Student C","Neutral"
"Student E","","Surprise"
```

### Export Flow
1. After attendance, the system generates a CSV file
2. File is saved to `/FaceAttendanceExports/` directory on device
3. User can share files via the Export screen (email, WhatsApp, etc.)
4. Filenames include teacher name, subject, and timestamp

### Backup/Restore
The Settings screen provides full data backup:
- **Backup**: Exports ALL data (students, embeddings, attendance, subjects, sessions) as a single JSON file
- **Restore**: Imports a backup JSON file, replacing all current data
- **File Picker**: User can select backup file from device storage

---

## 11. File Structure Reference

```
lib/
├── main.dart                           ← App entry point, route configuration
├── main_app.dart                       ← Alternative entry point
│
├── models/                             ← Data models (plain Dart classes)
│   ├── student_model.dart              ← Student: name, roll, class, etc.
│   ├── embedding_model.dart            ← FaceEmbedding: 128-d vector
│   ├── attendance_model.dart           ← AttendanceRecord: date, status, emotion
│   ├── face_detection_model.dart       ← DetectedFace: bounding box, pose
│   ├── match_result_model.dart         ← MatchResult: known/unknown + similarity
│   └── subject_model.dart              ← Subject + TeacherSession
│
├── modules/                            ← ML and business logic modules
│   ├── m1_face_detection.dart          ← Google ML Kit face detection
│   ├── m2_face_embedding.dart          ← FaceNet TFLite 128-d embeddings
│   ├── m3_face_matching.dart           ← KNN Euclidean distance matching
│   ├── m4_attendance_management.dart   ← Attendance CRUD + reports
│   ├── m5_liveness_detection.dart      ← Blink-based liveness (EAR)
│   ├── m6_emotion_detection.dart       ← High-level emotion API + fallback
│   ├── emotion_engine.dart             ← Pipeline orchestrator (EFN+HOG→SVM)
│   ├── emotion_feature_extractor.dart  ← EfficientNet TFLite + log transform
│   ├── hog_feature_extractor.dart      ← HOG descriptor (skimage-compatible)
│   ├── emotion_model_parameters.dart   ← Load JSON params (scaler, LDA, SVM)
│   ├── svm_classifier.dart             ← RBF SVM decision function + softmax
│   └── efficientnet_emotion_classifier.dart ← Utility class
│
├── database/                           ← Data persistence layer
│   ├── database_manager.dart           ← SharedPreferences CRUD operations
│   ├── database_connection.dart        ← Connection helper
│   └── face_recognition_database.dart  ← Additional DB utilities
│
├── screens/                            ← UI screens
│   ├── home_screen.dart                ← Dashboard with navigation buttons
│   ├── enrollment_screen.dart          ← Student registration + face capture
│   ├── attendance_prep_screen.dart     ← Select teacher + subject
│   ├── attendance_screen.dart          ← Live camera attendance taking
│   ├── database_screen.dart            ← View students + attendance records
│   ├── export_screen.dart              ← CSV file management
│   ├── expression_detection_screen.dart← Standalone emotion monitoring
│   └── settings_screen.dart            ← App settings + backup/restore
│
├── widgets/                            ← Reusable UI components
│   └── animated_background.dart        ← Animated gradient background
│
└── utils/                              ← Utility functions
    ├── constants.dart                  ← App theme, colors, routes, sizing
    ├── csv_export_service.dart         ← CSV generation for attendance
    ├── export_utils.dart               ← Export directory management
    └── theme.dart                      ← Additional theme configuration

assets/models/
├── embedding_model.tflite              ← FaceNet 128-d (face identity)
├── efficientnet_feature_extractor.tflite ← EfficientNet-B0 (emotion features)
└── emotion_runtime_params.json         ← SVM+LDA+Scaler parameters

training/                               ← Python training scripts
├── EfficientNetb0_HOG_pose_FM (1).csv  ← Pre-extracted 8234-sample dataset
├── train_emotion_model.py              ← Full training script (CLI)
├── hog_compat.py                       ← Python HOG matching skimage exactly
├── Mood Prediction.ipynb               ← Original research notebook
└── Emotion - Deep2.pdf                 ← Published research paper
```

---

## 12. Problems We Solved

### Problem 1: Emotion Always "Angry"
**Cause**: EfficientNet outputs softmax probabilities (0–1) but training data used log-probabilities (-14 to 0). The scaler produced garbage values.  
**Fix**: Apply `log(clamp(prob, 1e-7))` to EFN output.  
**Impact**: 0% → 80.6% accuracy.

### Problem 2: HOG Parameters Wrong
**Cause**: Dart used 64×64 image with 8×8 cells. Training used 256×256 with 32×32 cells. Both produce 1568 features by coincidence but with completely different spatial meaning.  
**Fix**: Match the training notebook exactly: 256×256, 32px cells.  
**Impact**: HOG features became usable → 80.6% → 85.8% accuracy.

### Problem 3: skimage Algorithm Mismatches
**Cause**: Three subtle differences between our Dart HOG and the real skimage algorithm: (1) border gradients should be 0 not edge-copied, (2) hard spatial assignment not interpolation, (3) L2-Hys uses epsilon²=1e-10 not epsilon=1e-5.  
**Fix**: Read the actual skimage Cython source code and match it exactly.  
**Validation**: Correlation = 1.0000 between our reimplementation and skimage.

### Problem 4: Neutral → Angry Confusion
**Cause**: Neutral and Angry faces share similar relaxed/tense features. SVM decision boundary is tight. Runtime conditions (camera, lighting) push borderline cases toward Angry.  
**Fix**: Post-hoc class bias correction — Neutral +0.3, Angry -0.1 on raw SVM scores.  
**Impact**: Neutral recall improved from 80.6% → 84.5%.

---

## 13. Potential Guide Questions & Answers

### Q: "Why not use a pre-built emotion recognition API?"
**A**: The system runs entirely offline with no internet dependency. This ensures it works in classrooms with poor connectivity. It also keeps student facial data private — no images are sent to external servers. Additionally, the research paper specifically proposes this EFN+HOG fusion approach as a novel contribution.

### Q: "Why EfficientNet + HOG? Why not just one of them?"
**A**: EfficientNet captures high-level semantic features (what the face "means" in the context of 1000 ImageNet categories), while HOG captures low-level geometric features (edge directions and shapes). They provide **complementary information**:
- EFN alone: 80.6% accuracy
- HOG alone: would be much lower (HOG can't capture deep semantics)
- **EFN + HOG together: 85.8%** — the combination is better than either alone

This is the core insight of the research paper — "synergistic feature fusion."

### Q: "Why use SVM instead of a neural network for classification?"
**A**: After LDA reduces from 2570 to just 5 features, the classification problem is low-dimensional. SVMs excel at low-dimensional classification where the decision boundaries are complex (RBF kernel handles non-linear boundaries). A neural network would be overkill for 5 input features and would be harder to export/deploy. The SVM with all its parameters fits in a 640 KB JSON file.

### Q: "Why LDA? Why not PCA?"
**A**: PCA finds directions of maximum **variance** in the data — it doesn't use class labels. LDA finds directions that maximize the **separation between classes**. Since we know the 6 emotion labels, LDA gives us 5 features that are specifically optimized for telling emotions apart. PCA might preserve variation that's irrelevant to emotion (like lighting changes).

### Q: "How does face recognition work without training on your students?"
**A**: FaceNet is pre-trained on millions of face images to produce embeddings where same-person faces are close and different-person faces are far apart. We don't retrain FaceNet — we just use it as a feature extractor. During enrollment, we store the student's embeddings. During attendance, we compare the new embedding to stored ones using Euclidean distance. This is called **metric learning** — the model learns a distance metric, not class labels.

### Q: "What's the similarity threshold and why 0.75?"
**A**: The threshold of 0.75 (where `similarity = 1/(1+distance)`) means a Euclidean distance of about 0.33. With FaceNet's L2-normalized 128-d embeddings, same-person pairs typically have distance < 0.3 and different-person pairs > 0.5. The 0.75 threshold sits in the safe zone between these distributions.

### Q: "Can someone trick the system with a photo?"
**A**: The M5 Liveness Detection module uses blink detection (Eye Aspect Ratio) to verify the face is real. A printed photo can't blink. However, this module requires face mesh landmarks which aren't always available — it's an additional security layer, not a guarantee.

### Q: "Why SharedPreferences instead of SQLite?"
**A**: SharedPreferences was chosen for simplicity and reliability across platforms. The data volume is manageable (hundreds of students, not millions). SQLite would be more efficient for complex queries, but SharedPreferences avoids codegen dependencies, native build issues, and keeps the codebase simple. All operations are simple CRUD — no complex joins or aggregations.

### Q: "What's the model accuracy and how was it tested?"
**A**: The emotion model achieves **85.8% accuracy** on a 20% hold-out test set (1647 samples) with stratified random split. Per-class: Surprise 95%, Disgust 93%, Happy 91%, Neutral 81%, Angry 80%, Sad 76%. The training dataset has 8234 pre-extracted feature samples across 6 balanced classes (~1200-1500 per class).

### Q: "What happens if the face is too dark or blurry?"
**A**: If ML Kit can't detect a face, no box appears — the system simply skips that frame. If a face is detected but the embedding quality is poor, the Euclidean distance to all stored embeddings will be large, similarity will be below 0.75, and it won't match anyone (treated as "unknown"). The system is inherently robust to poor-quality inputs because bad images produce distant embeddings.

### Q: "How fast is the emotion detection?"
**A**: Each emotion prediction takes approximately 100-200ms on a modern phone:
- EfficientNet inference: ~50ms (TFLite, 4 threads)
- HOG extraction: ~30ms (Dart, 256×256)
- Scaler + LDA + SVM: ~5ms (simple matrix math)
- Total: ~85-200ms depending on device

This is fast enough for real-time display (5-10 FPS on the emotion pipeline).

### Q: "What is the L2-Hys normalization in HOG?"
**A**: L2-Hys stands for "L2 norm with Hysteresis clipping." Steps:
1. Compute L2 norm of the block: `||block||₂`
2. Divide every element by this norm: `block / ||block||₂`
3. Clip any value above 0.2 to exactly 0.2
4. Renormalize by the new L2 norm
This prevents any single gradient direction from dominating the descriptor and makes HOG robust to local contrast variations.

### Q: "Why 128-dimensional face embeddings?"
**A**: 128 dimensions provide a good balance: enough to uniquely represent millions of different faces (2^128 possible states), but compact enough for fast distance calculations. FaceNet was specifically designed and trained to produce discriminative 128-d vectors. Google's original FaceNet paper showed 128-d achieves 99.63% on the LFW face verification benchmark.

### Q: "Can the system handle spectacles/masks?"
**A**: If a student enrolls wearing glasses and later appears without them (or vice versa), the face embedding will differ. However, by capturing 10+ enrollment photos (some with glasses, some without), the K-NN search with K=5 provides robustness — at least some stored embeddings will match. Masks that cover the mouth/nose area significantly reduce accuracy since they hide key facial features.

### Q: "How is the attendance data exported?"
**A**: The system generates CSV files with teacher name, subject name, date, and two columns: attendees and absentees with names. An optional "Emotion" column records the detected emotion at the time of attendance. Files are saved to a dedicated export folder and can be shared via the device's share sheet (email, WhatsApp, Google Drive, etc.).

---

## Summary

FaceAttend is a complete offline face recognition attendance system with real-time emotion detection. The key technical contributions are:

1. **On-device face recognition** using FaceNet embeddings + Euclidean KNN matching
2. **Synergistic emotion recognition** combining EfficientNet deep features with HOG classical features
3. **Exact skimage-compatible HOG implementation** in Dart (validated to correlation = 1.0)
4. **Full ML pipeline in JSON** — scaler, LDA, SVM all stored as numerical parameters, no Python needed at runtime
5. **Practical deployment** — works offline, handles real-world camera conditions, includes CSV export for institutional use

The emotion recognition pipeline implements the paper "Synergistic Feature Fusion Approach Towards Real-Time Children Facial Emotion Recognition" by Shivaprasad D L and D S Guru, achieving 85.8% accuracy on a 6-class balanced dataset.
