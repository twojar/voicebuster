import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from librosa.feature import mfcc
from train import get_mfcc
import joblib
import librosa
import os
import sys

def predict(filepath):
    model = joblib.load('voicebuster.pk1')
    mfcc_array = get_mfcc(filepath)
    mfcc_array = mfcc_array.reshape(1,-1)
    # [prob_real, prob_fake]
    probs = model.predict_proba(mfcc_array)[0]
    prediction = np.argmax(probs)
    prediction_label = "REAL" if prediction == 0 else "FAKE"
    confidence = probs[prediction]
    return prediction_label, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: test.py youraudiofile.wav")
        sys.exit(1)
    prediction_label, confidence = predict(sys.argv[1])
    print(f"Analysis: {prediction_label} {confidence*100:.2f}%")

    
