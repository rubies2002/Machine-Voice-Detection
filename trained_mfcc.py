import os
import librosa
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define paths to human and machine voice datasets
human_path = "D:/Reuben/SpeechLogix Project/human"
machine_path = "D:/Reuben/SpeechLogix Project/machine"

# Define function to extract MFCC features from audio files
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# Load human and machine voice datasets
human_files = os.listdir(human_path)
machine_files = os.listdir(machine_path)

human_features = [extract_features(os.path.join(human_path, f)) for f in human_files]
machine_features = [extract_features(os.path.join(machine_path, f)) for f in machine_files]

# Create feature matrix and label vector
X = human_features + machine_features
y = np.concatenate((np.zeros(len(human_features)), np.ones(len(machine_features))))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model on the training data
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")



# Save the trained SVM model to a file
joblib.dump(model, 'svm_model_mfcc.pkl')