import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from librosa.feature import mfcc
import joblib
import librosa
import os

# CONSTANTS
CATEGORIES = ['real', 'fake']

# Used to get spectral characteristics of a .wav file
# Use MFCC to train model
def get_mfcc(filename):
    # waveform (y), sampling rate (sr)
    y, sr = librosa.load(filename)
    #each row represents a different coefficient
    #each column represents a frame in the audio signal
    mfcc_array = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    #Scale down to 1D vector
    mfcc_array = np.mean(mfcc_array.T, axis=0)
    return mfcc_array


# x: stores mfcc vectors from audio training data
# y: stores labels from audio training data (real or fake)
x = []
y = []


# Loop thru audio data and classify in y array
# y = 0 (real)
# y = 1 (fake)
for label, category in enumerate(CATEGORIES):
    folder = os.path.join('audio', category)
    for filename in os.listdir(folder): 
        if filename.endswith('wav') or filename.endswith('mp3'):
            file_path = os.path.join(folder, filename)
            try:
                mfcc_vector = get_mfcc(file_path)
                x.append(mfcc_vector)
                y.append(label)
            except Exception as Error:
                print(Error)
                print("ERROR!")

x = np.array(x)
y = np.array(y)

# split arrays into random train and test subsets
# 25% testing, 75% training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#train model using RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

#test
predictions = model.predict(x_test)
print(classification_report(y_test, predictions))

#save model to path
joblib.dump(model, 'voicebuster.pk1')


