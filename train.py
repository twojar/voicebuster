import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from librosa.feature import mfcc
import librosa

# Used to get spectral characteristics of a .wav file
# Use MFCC to train model
def get_mfcc(filename):
    # waveform (y), sampling rate (sr)
    y, sr = librosa.load(filename)
    #each row represents a different coefficient
    #each column represents a frame in the audio signal
    mfcc_array = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return mfcc_array

mfcc_array = get_mfcc('audio/a1.mp3')
print(mfcc_array)
