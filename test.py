import numpy as np
from train import get_mfcc
import joblib
import sys

def predict(filepath):
    model = joblib.load('voicebuster_mlp.pkl')
    mfcc_array = get_mfcc(filepath)
    mfcc_array = mfcc_array.reshape(1,-1)
    
    # [prob_real, prob_fake]
    probs = model.predict_proba(mfcc_array)[0]
    prediction = np.argmax(probs)
    truth_value = "REAL" if prediction == 0 else "FAKE"
    percent = probs[prediction]

    #Returns truth value with confidence percentage
    return truth_value, percent

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test.py youraudiofile.wav")
        sys.exit(1)
    prediction_label, confidence = predict(sys.argv[1])
    print(f"Analysis on {sys.argv[1]}: {prediction_label} {confidence*100:.2f}%")

    
