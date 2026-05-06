# ============================================================
# STREAMLIT APP — FLANGE ACOUSTIC CLASSIFICATION SYSTEM
# ============================================================

# Run using:
# streamlit run app.py

# ============================================================
# 1. IMPORTS
# ============================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flange Acoustic AI",
    layout="wide"
)

st.title("🔩 Flange Acoustic Classification System")

st.markdown("""
This application performs:

- Signal Processing
- FFT Analysis
- Mel Spectrogram Analysis
- MFCC Feature Extraction
- Machine Learning Classification
- Deep Learning Prediction
- Ensemble Voting
""")

# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def peak_to_peak(signal):

    return np.max(signal) - np.min(signal)


def crest_factor(signal):

    rms = np.sqrt(np.mean(signal ** 2)) + 1e-9

    return np.max(np.abs(signal)) / rms


def fft_features(signal, sr):

    fft = np.fft.rfft(signal)

    mag = np.abs(fft)

    freqs = np.fft.rfftfreq(
        len(signal),
        1 / sr
    )

    centroid = np.sum(freqs * mag) / (
        np.sum(mag) + 1e-9
    )

    bandwidth = np.sqrt(
        np.sum(
            ((freqs - centroid) ** 2) * mag
        ) / (np.sum(mag) + 1e-9)
    )

    return centroid, bandwidth


def extract_features(signal, sr):

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=13
        ),
        axis=1
    )

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_mels=40
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_mean = np.mean(
        log_mel,
        axis=1
    )

    energy = np.mean(signal ** 2)

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    p2p = peak_to_peak(signal)

    crest = crest_factor(signal)

    centroid, bandwidth = fft_features(
        signal,
        sr
    )

    spec_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sr
        )
    )

    spec_rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=signal,
            sr=sr
        )
    )

    return np.hstack([

        mfcc,

        mel_mean,

        energy,
        zcr,
        p2p,
        crest,

        centroid,
        bandwidth,

        spec_centroid,
        spec_rolloff
    ])

# ============================================================
# PLOTTING FUNCTIONS
# ============================================================

def plot_waveform(signal):

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(signal)

    ax.set_title("Percussion Waveform")

    ax.set_xlabel("Samples")

    ax.set_ylabel("Amplitude")

    st.pyplot(fig)


def plot_fft(signal, sr):

    fft = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        1/sr
    )

    fig, ax = plt.subplots(figsize=(12,4))

    ax.plot(freqs, np.abs(fft))

    ax.set_title("FFT Spectrum")

    ax.set_xlabel("Frequency (Hz)")

    ax.set_ylabel("Magnitude")

    st.pyplot(fig)


def plot_mel(signal, sr):

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    fig, ax = plt.subplots(figsize=(10,4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=ax
    )

    ax.set_title("Mel Spectrogram")

    st.pyplot(fig)


def plot_mfcc(signal, sr):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    fig, ax = plt.subplots(figsize=(10,4))

    librosa.display.specshow(
        mfcc,
        x_axis='time',
        ax=ax
    )

    ax.set_title("MFCC")

    st.pyplot(fig)

# ============================================================
# MODEL TRAINING
# ============================================================

@st.cache_resource

def train_models():

    # --------------------------------------------------------
    # DUMMY TRAINING DATA
    # Replace with your real saved dataset
    # --------------------------------------------------------

    X = np.random.randn(300, 59)

    y = np.random.randint(0, 3, 300)

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # RF
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf.fit(X_scaled, y)

    # SVM
    svm = SVC(kernel='rbf', probability=True)

    svm.fit(X_scaled, y)

    # CNN
    cnn = models.Sequential([

        layers.Reshape(
            (59,1),
            input_shape=(59,)
        ),

        layers.Conv1D(
            32,
            3,
            activation='relu'
        ),

        layers.MaxPooling1D(2),

        layers.Conv1D(
            64,
            3,
            activation='relu'
        ),

        layers.GlobalAveragePooling1D(),

        layers.Dense(
            64,
            activation='relu'
        ),

        layers.Dense(
            3,
            activation='softmax'
        )
    ])

    cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    cnn.fit(
        X_scaled,
        y,
        epochs=5,
        verbose=0
    )

    return scaler, rf, svm, cnn

# ============================================================
# LOAD MODELS
# ============================================================

scaler, rf, svm, cnn = train_models()

# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav","mp3","m4a","mp4"]
)

# ============================================================
# PROCESS AUDIO
# ============================================================

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    signal, sr = librosa.load(
        temp_path,
        sr=48000
    )

    signal = (
        signal - np.mean(signal)
    ) / (np.std(signal) + 1e-9)

    st.success("Audio Loaded Successfully")

    # --------------------------------------------------------
    # SIGNAL VISUALIZATION
    # --------------------------------------------------------

    st.header("📈 Signal Analysis")

    plot_waveform(signal)

    plot_fft(signal, sr)

    plot_mel(signal, sr)

    plot_mfcc(signal, sr)

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    features = extract_features(
        signal,
        sr
    )

    X = scaler.transform(
        [features]
    )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    st.header("🤖 Model Predictions")

    rf_pred = rf.predict(X)[0]

    svm_pred = svm.predict(X)[0]

    cnn_pred = np.argmax(
        cnn.predict(X),
        axis=1
    )[0]

    preds = [rf_pred, svm_pred, cnn_pred]

    ensemble = max(
        set(preds),
        key=preds.count
    )

    label_map = {

        0: "0 ft-lbs (Loose)",
        1: "25 ft-lbs",
        2: "50 ft-lbs (Tight)"
    }

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Random Forest",
        label_map[rf_pred]
    )

    col2.metric(
        "SVM",
        label_map[svm_pred]
    )

    col3.metric(
        "CNN",
        label_map[cnn_pred]
    )

    col4.metric(
        "Ensemble",
        label_map[ensemble]
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    st.header("✅ Final Prediction")

    st.success(
        f"Predicted Torque Condition: {label_map[ensemble]}"
    )

    # --------------------------------------------------------
    # FEATURE TABLE
    # --------------------------------------------------------

    st.header("📊 Extracted Features")

    feature_df = pd.DataFrame(
        features.reshape(1,-1)
    )

    st.dataframe(feature_df)

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    os.remove(temp_path)
