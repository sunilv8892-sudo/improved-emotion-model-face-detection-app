Generated emotion-model artifacts live here.

Expected outputs from training/train_emotion_model.py:

- scaler.pkl
- lda.pkl
- svm.pkl
- emotion_runtime_params.json
- efficientnet_feature_extractor.tflite

Only these two files need to be copied into Flutter assets/models:

- emotion_runtime_params.json
- efficientnet_feature_extractor.tflite

The .pkl files are kept here for reproducibility and offline inspection.