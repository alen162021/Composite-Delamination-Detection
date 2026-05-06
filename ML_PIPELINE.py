import streamlit as st
import numpy as np
import librosa
import os
import joblib

st.set_page_config(page_title="Composite Delamination Detection", layout="wide")

st.title("🔍 Composite Delamination Detection System")

# ------------------------------------------------------------
# Load model (placeholder - RF recommended for deployment)
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ------------------------------------------------------------
# Feature extraction (same as your pipeline)
# ------------------------------------------------------------

def extract_features(signal, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    energy = np.mean(signal ** 2)

    return np.hstack([mfcc, zcr, energy])

# ------------------------------------------------------------
# Upload file
# ------------------------------------------------------------

uploaded_file = st.file_uploader("Upload impact audio (.wav)", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    signal, sr = librosa.load(uploaded_file, sr=48000)

    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)

    features = extract_features(signal, sr).reshape(1, -1)

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    st.subheader("Prediction Result")

    st.success(f"Predicted Class: {prediction}")

    st.write("0 = Tight | 1 = Loose (example mapping)")
