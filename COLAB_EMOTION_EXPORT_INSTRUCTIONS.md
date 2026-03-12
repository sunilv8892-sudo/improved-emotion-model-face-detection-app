**Colab Export Guide**

This file gives the fastest path to generate the emotion-model outputs in Google Colab and copy them back into this Flutter app.

The Flutter code is already prepared to consume these files:

- `efficientnet_feature_extractor.tflite`
- `emotion_runtime_params.json`

Optional training artifacts for reference:

- `scaler.pkl`
- `lda.pkl`
- `svm.pkl`

Only the two runtime files above are required by Flutter.

**What You Need To Upload To Colab**

Upload these two files from this repo into Colab:

- `training/train_emotion_model.py`
- `training/EfficientNetb0_HOG_pose_FM (1).csv`

Fastest option:

- Rename the CSV to `emotion_features.csv` before uploading to avoid quoting issues.

**Colab Step 1: Install Packages**

Run this in a Colab cell:

```python
!pip install -q joblib numpy pandas scikit-learn tensorflow
```

**Colab Step 2: Upload Files**

Run this in a Colab cell and upload:

- `train_emotion_model.py`
- your CSV file

```python
from google.colab import files
uploaded = files.upload()
print(uploaded.keys())
```

If your uploaded CSV still has spaces/parentheses in the name, normalize it with:

```python
import os

for name in os.listdir('.'):
    if name.endswith('.csv'):
        os.rename(name, 'emotion_features.csv')
        print('Renamed CSV to emotion_features.csv')
        break
```

**Colab Step 3: Run Training + Export**

Run this exact command in a Colab cell:

```python
!python train_emotion_model.py emotion_features.csv --output-dir models --export-tflite
```

This should generate:

- `models/scaler.pkl`
- `models/lda.pkl`
- `models/svm.pkl`
- `models/emotion_runtime_params.json`
- `models/efficientnet_feature_extractor.tflite`

**If Column Names In The CSV Are Not Auto-Detected**

Use the fallback mode with explicit dimensions.

The Flutter runtime is currently aligned to this paper-style layout:

- EfficientNet features: around `1000`
- HOG features: `1568`
- Pose features: `2`

If auto-detection fails, inspect the CSV first:

```python
import pandas as pd

df = pd.read_csv('emotion_features.csv')
print(df.columns.tolist())
print(df.shape)
df.head()
```

Then run the trainer with explicit options, for example:

```python
!python train_emotion_model.py emotion_features.csv \
    --output-dir models \
    --export-tflite \
    --efficientnet-dim 1000 \
    --hog-dim 1568 \
    --pose-cols pose_x pose_y \
    --label-column emotion_label \
    --image-column image_name
```

If your CSV uses different column names, replace:

- `pose_x pose_y`
- `emotion_label`
- `image_name`

with the real names from the CSV.

**Colab Step 4: Verify Outputs Exist**

Run this in a Colab cell:

```python
import os

expected = [
    'models/scaler.pkl',
    'models/lda.pkl',
    'models/svm.pkl',
    'models/emotion_runtime_params.json',
    'models/efficientnet_feature_extractor.tflite',
]

for path in expected:
    print(path, 'OK' if os.path.exists(path) else 'MISSING')
```

**Colab Step 5: Download Outputs**

Fastest way is to zip the whole models folder.

```python
!zip -r emotion_models.zip models
```

Then download:

```python
from google.colab import files
files.download('emotion_models.zip')
```

If you prefer downloading only the Flutter runtime files:

```python
from google.colab import files
files.download('models/emotion_runtime_params.json')
files.download('models/efficientnet_feature_extractor.tflite')
```

**What To Copy Back Into This Flutter Repo**

Copy these two generated files into:

- `assets/models/emotion_runtime_params.json`
- `assets/models/efficientnet_feature_extractor.tflite`

Keep these optional files in the repo `models/` folder if you want reproducibility:

- `models/scaler.pkl`
- `models/lda.pkl`
- `models/svm.pkl`

**After You Paste The Files Here**

Run from the Flutter repo root:

```powershell
flutter pub get
flutter run
```

**Important Constraint**

The generated `emotion_runtime_params.json` must match the exported `efficientnet_feature_extractor.tflite` from the same training/export run.

Do not mix:

- one run's JSON
- another run's TFLite file

If they do not match, the app will throw a feature-dimension mismatch at runtime.

**If You Want The Safest Colab Workflow**

Use this order only:

1. Upload `train_emotion_model.py`
2. Upload the CSV
3. Run training/export once
4. Download the generated files from the same run
5. Copy both runtime files into `assets/models`

**Files Already Wired In Flutter**

These repo files already use the exported assets:

- [lib/modules/emotion_feature_extractor.dart](c:/Users/sunil.v/Downloads/final-attendence-app-working-feature-glass-matching-fix/lib/modules/emotion_feature_extractor.dart)
- [lib/modules/emotion_model_parameters.dart](c:/Users/sunil.v/Downloads/final-attendence-app-working-feature-glass-matching-fix/lib/modules/emotion_model_parameters.dart)
- [lib/modules/emotion_engine.dart](c:/Users/sunil.v/Downloads/final-attendence-app-working-feature-glass-matching-fix/lib/modules/emotion_engine.dart)
- [lib/modules/m6_emotion_detection.dart](c:/Users/sunil.v/Downloads/final-attendence-app-working-feature-glass-matching-fix/lib/modules/m6_emotion_detection.dart)

Once the two runtime files are in `assets/models`, the app side is ready.