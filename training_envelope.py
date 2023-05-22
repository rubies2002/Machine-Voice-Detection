import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import *

# Define function to extract amplitude envelope features from audio files
def extract_amplitude_envelope(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    hop_length = int(sr * 0.010)  # 10 ms hop length
    n_fft = 2048  # 2048-point FFT
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    return np.mean(S, axis=1)

# Define paths to human and machine voice datasets
human_path = "D:/Reuben/SpeechLogix Project/human"
machine_path = "D:/Reuben/SpeechLogix Project/machine"

# Load human and machine voice datasets
human_files = os.listdir(human_path)
machine_files = os.listdir(machine_path)

human_features = [extract_amplitude_envelope(os.path.join(human_path, f)) for f in human_files]
machine_features = [extract_amplitude_envelope(os.path.join(machine_path, f)) for f in machine_files]

# Remove the middle dimension from the feature arrays
human_features = np.squeeze(human_features)
machine_features = np.squeeze(machine_features)

# Create feature matrix and label vector
X = np.vstack((human_features, machine_features))
y = np.concatenate((np.zeros(len(human_features)), np.ones(len(machine_features))))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model on the training data
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model on the testing data
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# Save the trained SVM model to a file
joblib.dump(model, 'svm_model_amp_envelope.pkl')