def form(path):
    import librosa
    import numpy as np
    import joblib
    import training_formants
    

    # Load the trained SVM model
    svm_model = joblib.load('svm_model_formants.pkl')
    formants = training_formants.extract_formants(path)

    formants = np.array(formants).reshape(1, -1) 
    if svm_model.predict(formants) == 1:
        i=1
    else:
        i=0

    return(i)

def spec_centroid(path):
    import numpy as np
    import joblib
    import training_spectral_centroid 

    # Extract prosody features for the audio file
    features = training_spectral_centroid.extract_features(path)

    # Load pre-trained model
    model = joblib.load('spec_centroid.joblib')

    # Predict whether the voice is machine or human
    prediction = model.predict([features])[0]

    if prediction=="machine":
        a=1
    else:a=0
    return a


def spec_flatness(path):

    import librosa
    import numpy as np
   
        # Load audio file
        

    y, sr = librosa.load(path, sr=None)

    # Compute spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)

    # Calculate the mean of spectral flatness
    mean_spectral_flatness = np.mean(spectral_flatness)
    print(mean_spectral_flatness)

    # Set a threshold to differentiate machine and normal voice
    threshold = 0.05

    # Check if the mean spectral flatness is below the threshold
    if mean_spectral_flatness < threshold:
        # print("machine voice.")
        a=1
    else:
        # print("normal voice.")
        a=0
    return (a)

def zero_cross(path):
    import librosa

    signal, sr = librosa.load(path, sr=None)

    # Calculate zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(signal)

    # Compute the mean of the zero crossing rate for each frame
    zcr_mean = zcr.mean(axis=1)

    # Set a threshold to distinguish between machine and normal voice
    threshold = 0.05

    # If the mean zero crossing rate is above the threshold, it's likely a machine voice
    if zcr_mean > threshold:
        a=1
    else:
        a=0
    return(a)

def pitch(path):
    import librosa
    import numpy as np


    # Load the audio file
    y, sr = librosa.load(path, sr=None)

    # Extract the fundamental frequency (pitch) using the YIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # Calculate the mean pitch
    mean_pitch = np.mean(f0[voiced_flag])
    # print(mean_pitch)
    # Use the mean pitch to differentiate between machine and human voice
    if mean_pitch > 127:
        a=1
    else:
        a=0
    return(a)

def mfc(path):
    import joblib
    import librosa
    import numpy as np

    # Load the trained SVM model
    svm_model = joblib.load('svm_model_mfcc.pkl')

    # Load the audio file
    y, sr = librosa.load(path,sr=None)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # Calculate the mean of each MFCC coefficient
    mfcc_mean = np.mean(mfcc, axis=1)

    # Create a feature vector by concatenating the mean values of all MFCC coefficients
    feature_vector = np.concatenate((mfcc_mean.reshape(20,1)))

    # Use the SVM model to predict whether the audio sample is machine or normal voice
    if svm_model.predict([feature_vector])[0] == 1:
        a=1
    else:
        a=0
    return(a)

def pitc_contour(path):
    import librosa
    import training_pitch_contour
    import joblib

    # Load the trained SVM model
    svm_model = joblib.load('svm_model_pitch_contour.joblib')
    
    y, sr = librosa.load(path,sr=None)

    feature_vector = training_pitch_contour.extract_pitch(path)


    # Check if the median pitch value is above or below the threshold to classify the voice
    prediction = svm_model.predict([feature_vector])[0]

    if prediction=="machine":
        a=1
    else:a=0
    return a

def prosody(path):
    import librosa
    import numpy as np
    import joblib
    import training_prosody

    # Extract prosody features for the audio file
    features = training_prosody.extract_features(path)

    # Load pre-trained model
    model = joblib.load('prosody_1.joblib')

    # Predict whether the voice is machine or human
    prediction = model.predict([features])[0]

    if prediction=="machine":
        a=1
    else:a=0
    return a


def amplitude_env(path):
    import numpy as np
    import joblib
    import training_envelope

    # Load the trained SVM model
    svm_model = joblib.load('svm_model_amp_envelope.pkl')
    enve = training_envelope.extract_amplitude_envelope(path)

    enve = np.array(enve).reshape(1, -1) 
    if svm_model.predict(enve) == 1:
        i=1
    else:
        i=0
    
    return(i)

def spec_flux(path):
    import librosa
    import numpy as np
    import joblib
    # from sklearn.linear_model import LogisticRegression
    model = joblib.load('spectral_flux.joblib')

    y, sr = librosa.load(path)
    max_length=model.coef_.shape[1]
    spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
    pad_width = max_length - len(spectral_flux)
    padded_flux = np.pad(spectral_flux, (0, pad_width), mode='constant')

    prediction = model.predict(padded_flux.reshape(1, -1))[0]


    if prediction == "machine":
        a = 1
    else:
        a = 0

    return a