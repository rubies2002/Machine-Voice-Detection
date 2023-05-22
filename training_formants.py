import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import *

# Define function to extract formant features from audio files
def extract_formants(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    lpc_coeffs = librosa.lpc(y=y, order=12)
    roots = np.roots(lpc_coeffs)
    roots = roots[np.imag(roots) >= 0]
    frequencies = np.sort(np.arctan2(np.imag(roots), np.real(roots)) * (sr / (2 * np.pi)))
    formants = frequencies[:3]
    return formants

# Define paths to human and machine voice datasets
human_path = "D:/Reuben/SpeechLogix Project/human"
machine_path = "D:/Reuben/SpeechLogix Project/machine"

# Load human and machine voice datasets
human_files = os.listdir(human_path)
machine_files = os.listdir(machine_path)

human_features = [extract_formants(os.path.join(human_path, f)) for f in human_files]
machine_features = [extract_formants(os.path.join(machine_path, f)) for f in machine_files]

# Create feature matrix and label vector
X = human_features + machine_features
y = np.concatenate((np.zeros(len(human_features)), np.ones(len(machine_features))))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the feature matrices
X_train = np.array(X_train).reshape(-1, 3)
X_test = np.array(X_test).reshape(-1, 3)

# Train an SVM model on the training data
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the trained SVM model to a file
joblib.dump(model, 'svm_model_formants.pkl')