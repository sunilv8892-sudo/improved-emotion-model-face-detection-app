"""Quick grid search to improve Neutral recall."""
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.svm import SVC

LABELS = ['Angry','Disgust','Happy','Neutral','Sad','Surprise']
df = pd.read_csv('training/EfficientNetb0_HOG_pose_FM (1).csv')
features = df[[str(i) for i in range(1000)]].values.astype(np.float64)
labels = df['Class'].to_numpy(dtype=str)

X_tr, X_te, y_tr, y_te = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)

sc = MinMaxScaler()
X_trs = sc.fit_transform(X_tr)
X_tes = sc.transform(X_te)

lda = LinearDiscriminantAnalysis(n_components=5)
X_trl = lda.fit_transform(X_trs, y_tr)
X_tel = lda.transform(X_tes)

y_bin = label_binarize(y_tr, classes=LABELS)

print("Grid search for better Neutral recall:")
print("-" * 80)

best_neutral = 0
best_config = None

for C in [5, 10, 15, 20, 50]:
    for g in ['scale', 0.05, 0.1, 0.2, 0.5, 1.0]:
        ovr = OneVsRestClassifier(SVC(kernel='rbf', C=C, gamma=g), n_jobs=-1)
        ovr.fit(X_trl, y_bin)
        yp = [LABELS[np.argmax(r)] for r in ovr.predict(X_tel)]
        acc = accuracy_score(y_te, yp)
        rec = {}
        for l in LABELS:
            mask = y_te == l
            if mask.sum() > 0:
                rec[l] = np.mean(np.array(yp)[mask] == l)
        nr = rec.get('Neutral', 0)
        parts = ' '.join(f'{k[:3]}:{v:.2f}' for k, v in rec.items())
        print(f'C={C:>3} g={str(g):>5} acc={acc:.3f} Neutral={nr:.2f} | {parts}')
        if nr > best_neutral and acc > 0.78:
            best_neutral = nr
            best_config = (C, g, acc, nr, rec)

if best_config:
    C, g, acc, nr, rec = best_config
    print(f'\nBest for Neutral: C={C} gamma={g} acc={acc:.3f} Neutral={nr:.2f}')
    print(f'All recalls: {rec}')
