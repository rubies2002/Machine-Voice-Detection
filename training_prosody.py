import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Define constants
HUMAN_DATA_DIR = "D:/Reuben/SpeechLogix Project/human"
MACHINE_DATA_DIR = "D:/Reuben/SpeechLogix Project/machine"
AUDIO_DURATION = 3
FEATURES = ['tempo', 'chroma', 'mfcc', 'rms']

# Load data
X, y = [], []
for label, data_dir in [('human', HUMAN_DATA_DIR), ('machine', MACHINE_DATA_DIR)]:
    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(data_dir, filename)
            y.append(label)
            X.append(filepath)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Extract prosody features from audio files
FEATURES = ['tempo', 'chroma', 'mfcc', 'rms', 'spectral_centroid']

# Extract prosody features from audio files
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    features = []
    if 'tempo' in FEATURES:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
    if 'chroma' in FEATURES:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
    if 'mfcc' in FEATURES:
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features.extend(np.mean(mfcc, axis=1))
    if 'rms' in FEATURES:
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
    if 'spectral_centroid' in FEATURES:
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid))
    return features

X_train_features = [extract_features(x) for x in X_train]
X_test_features = [extract_features(x) for x in X_test]

# Train SVM classifier
clf = SVC(kernel='linear', C=1, random_state=42)
clf.fit(X_train_features, y_train)

# # Evaluate classifier
# y_pred_train = clf.predict(X_train_features)
# train_accuracy = accuracy_score(y_train, y_pred_train)
# print('Train accuracy:', train_accuracy)
# y_pred_test = clf.predict(X_test_features)
# test_accuracy = accuracy_score(y_test, y_pred_test)
# print('Test accuracy:', test_accuracy)

# Save classifier as joblib file
joblib.dump(clf, 'prosody_1.joblib')