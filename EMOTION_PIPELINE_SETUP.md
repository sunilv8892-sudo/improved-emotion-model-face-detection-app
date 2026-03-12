**Emotion Pipeline Setup**

This repo is now wired so the remaining work is only to generate the real model artifacts from your dataset and copy them into Flutter assets.

**What is already implemented**

- Flutter identity pipeline remains unchanged: ML Kit face detect -> crop -> FaceNet -> KNN.
- Flutter emotion pipeline is implemented in parallel: EfficientNet features + HOG + pose -> MinMaxScaler -> LDA(5) -> RBF SVM.
- Attendance UI now stores and displays `Name | Emotion | Time | Attendance Marked`.
- Expression detection screen also uses the new emotion pipeline.

**Files to run**

- Training code: [training/train_emotion_model.py](training/train_emotion_model.py)
- Python deps: [training/requirements.txt](training/requirements.txt)
- Training helper: [scripts/run_emotion_training.ps1](scripts/run_emotion_training.ps1)
- Asset copy helper: [scripts/copy_emotion_assets.ps1](scripts/copy_emotion_assets.ps1)

**Expected CSV layout**

The trainer expects one row per image with:

- EfficientNet feature columns
- HOG feature columns
- pose x/y columns
- emotion label column
- optional image-name column

It will auto-detect columns by prefix when possible. If your CSV uses different names, pass explicit options like `--efficientnet-dim`, `--hog-dim`, and `--pose-cols`.

**One-time training flow**

From the repo root in PowerShell:

```powershell
./scripts/run_emotion_training.ps1 -CsvPath "C:\path\to\emotion_features.csv" -ExportTflite
./scripts/copy_emotion_assets.ps1
```

This produces:

- [models/README.md](models/README.md) described artifacts in `models/`
- `emotion_runtime_params.json` and `efficientnet_feature_extractor.tflite` copied into `assets/models/`

**Then run Flutter**

```powershell
flutter pub get
flutter run
```

**Important constraint**

I cannot guarantee `100%` correctness before the real CSV and generated assets exist, because the trained feature dimensions must match the exported TFLite feature extractor exactly. The code is prepared for that and will fail fast with a clear dimension-mismatch error if the training artifacts do not match the runtime extractor.

**If your CSV uses the paper's original dimensions**

- EfficientNet features: about 1000
- HOG features: 1568
- Pose features: 2
- Total before LDA: about 2570

The Dart HOG extractor has been aligned to 1568 features to match that layout.