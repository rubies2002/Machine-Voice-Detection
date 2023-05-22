import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

def extract_spectral_flux(audio_file):
    y, sr = librosa.load(audio_file)
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    return spectral_flux

# Path to the directory containing human voice audio files
human_voice_dir = "D:/Reuben/SpeechLogix Project/human"

# Path to the directory containing machine voice audio files
machine_voice_dir = "D:/Reuben/SpeechLogix Project/machine"

max_length = 0  # Maximum length of spectral flux feature vectors

X = []  # Feature matrix
y = []  # Labels


# Extract features for human voice samples
for root, dirs, files in os.walk(human_voice_dir):
    for file in files:
        audio_file = os.path.join(root, file)
        spectral_flux = extract_spectral_flux(audio_file)  # Function to extract spectral flux feature
        max_length = max(max_length, len(spectral_flux))  # Update maximum length
        X.append(spectral_flux)
        y.append('human')

# Extract features for machine voice samples
for root, dirs, files in os.walk(machine_voice_dir):
    for file in files:
        audio_file = os.path.join(root, file)
        spectral_flux = extract_spectral_flux(audio_file)  # Function to extract spectral flux feature
        max_length = max(max_length, len(spectral_flux))  # Update maximum length
        X.append(spectral_flux)
        y.append('machine')

# Pad the feature vectors to ensure consistent length
X_padded = []
for spectral_flux in X:
    pad_width = max_length - len(spectral_flux)
    padded_flux = np.pad(spectral_flux, (0, pad_width), mode='constant')
    X_padded.append(padded_flux)

X = np.array(X_padded)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


joblib.dump(model, 'spectral_flux.joblib')