# ============================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================

!apt-get install ffmpeg -y
!pip install librosa scikit-learn tensorflow seaborn hmmlearn

# ============================================================
# 2. IMPORTS
# ============================================================

import os
import zipfile
import numpy as np
import librosa
import librosa.display
import re
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

from hmmlearn import hmm

# ============================================================
# 3. UPLOAD ZIP FILE
# ============================================================

uploaded = files.upload()

zip_file = list(uploaded.keys())[0]

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall("data")

base_path = "data"

print("Folders Found:")
print(os.listdir(base_path))

# ============================================================
# 4. PARSE FILE NAMES
# ============================================================

def parse_train(filename):

    filename = filename.replace(".mp4", "")
    filename = filename.replace(".m4a", "")

    match = re.match(
        r"(\d+)ftlbF(\d)A(\d)",
        filename
    )

    if match:

        torque = match.group(1)
        flange = "F" + match.group(2)
        area = "A" + match.group(3)

        return torque, flange, area

    return None, None, None


def parse_test(filename):

    filename = filename.replace(".mp4", "")
    filename = filename.replace(".m4a", "")

    match = re.match(r"F(\d)A(\d)", filename)

    if match:

        flange = "F" + match.group(1)
        area = "A" + match.group(2)

        return flange, area

    return None, None

# ============================================================
# 5. LOAD AUDIO
# ============================================================

def load_audio(path):

    signal, sr = librosa.load(path, sr=48000)

    signal = (
        signal - np.mean(signal)
    ) / (np.std(signal) + 1e-9)

    return signal, sr

# ============================================================
# 6. SPLIT AUDIO INTO INDIVIDUAL HITS
# ============================================================

def split_hits(signal, sr):

    energy = librosa.feature.rms(y=signal)[0]

    threshold = (
        np.mean(energy)
        + 0.5 * np.std(energy)
    )

    frames = np.where(energy > threshold)[0]

    if len(frames) < 5:
        return [signal]

    segments = np.split(
        frames,
        np.where(np.diff(frames) > 2)[0] + 1
    )

    hits = []

    for s in segments:

        start = s[0] * 512
        end = min(len(signal), s[-1] * 512)

        hit = signal[start:end]

        if len(hit) > 1000:
            hits.append(hit)

    return hits

# ============================================================
# 7. FEATURE EXTRACTION
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
# 8. VISUALIZATION FUNCTIONS
# ============================================================

def plot_waveform(signal):

    plt.figure(figsize=(12,4))

    plt.plot(signal)

    plt.title("Percussion Waveform")

    plt.xlabel("Samples")

    plt.ylabel("Amplitude")

    plt.show()


def plot_fft(signal, sr):

    fft = np.fft.rfft(signal)

    freqs = np.fft.rfftfreq(
        len(signal),
        1/sr
    )

    plt.figure(figsize=(12,4))

    plt.plot(freqs, np.abs(fft))

    plt.title("FFT Spectrum")

    plt.xlabel("Frequency (Hz)")

    plt.ylabel("Magnitude")

    plt.show()


def plot_mel(signal, sr):

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    plt.figure(figsize=(10,4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis='time',
        y_axis='mel'
    )

    plt.colorbar()

    plt.title("Mel Spectrogram")

    plt.show()


def plot_mfcc(signal, sr):

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=13
    )

    plt.figure(figsize=(10,4))

    librosa.display.specshow(
        mfcc,
        x_axis='time'
    )

    plt.colorbar()

    plt.title("MFCC")

    plt.show()

# ============================================================
# 9. LOAD DATASETS
# ============================================================

datasets = {}

for folder in os.listdir(base_path):

    if not folder.startswith("FLANGE_"):
        continue

    flange_id = "F" + folder.split("_")[1]

    X = []
    y = []

    folder_path = os.path.join(
        base_path,
        folder
    )

    print("\nLoading:", folder)

    for file in os.listdir(folder_path):

        if not (
            file.endswith(".mp4")
            or file.endswith(".m4a")
        ):
            continue

        file_path = os.path.join(
            folder_path,
            file
        )

        torque, fl, area = parse_train(file)

        if torque is None:
            continue

        signal, sr = load_audio(file_path)

        hits = split_hits(signal, sr)

        for h in hits:

            X.append(
                extract_features(h, sr)
            )

            y.append(torque)

    datasets[flange_id] = (
        np.array(X),
        np.array(y)
    )

print("\nLoaded:")
print(datasets.keys())

# ============================================================
# 10. SIGNAL VISUALIZATION
# ============================================================

print("\n################################")
print("SIGNAL ANALYSIS")
print("################################")

example_folder = "FLANGE_1"

folder_path = os.path.join(
    base_path,
    example_folder
)

example_file = os.listdir(folder_path)[0]

example_path = os.path.join(
    folder_path,
    example_file
)

signal, sr = load_audio(example_path)

hits = split_hits(signal, sr)

example_hit = hits[0]

plot_waveform(example_hit)

plot_fft(example_hit, sr)

plot_mel(example_hit, sr)

plot_mfcc(example_hit, sr)

# ============================================================
# 11. LABEL ENCODING
# ============================================================

le = LabelEncoder()

all_labels = np.concatenate(
    [datasets[k][1] for k in datasets]
)

le.fit(all_labels)

for k in datasets:

    X, y = datasets[k]

    datasets[k] = (
        X,
        le.transform(y)
    )

print("Classes:")
print(le.classes_)

# ============================================================
# 12. EVALUATION FUNCTIONS
# ============================================================

def plot_cm(cm, labels, title):

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(title)

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.show()


def evaluate_model(
    y_true,
    y_pred,
    labels,
    title
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    acc = accuracy_score(
        y_true,
        y_pred
    )

    print("\n===================")
    print(title)

    print("Accuracy:", acc)

    class_acc = {}

    for i in range(len(cm)):

        class_acc[labels[i]] = (
            cm[i][i] / np.sum(cm[i])
        )

    print("Per-Class Accuracy:")
    print(class_acc)

    plot_cm(cm, labels, title)

# ============================================================
# 13. MODEL DEFINITIONS
# ============================================================

def train_rf(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_svm(X_train, y_train):

    model = SVC(kernel='rbf')

    model.fit(X_train, y_train)

    return model


def train_dt(X_train, y_train):

    model = DecisionTreeClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_lr(X_train, y_train):

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(X_train, y_train)

    return model


def train_bpnn(X_train, y_train):

    model = MLPClassifier(
        hidden_layer_sizes=(128,64),
        max_iter=500,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def build_cnn(input_dim, num_classes):

    model = models.Sequential([

        layers.Reshape(
            (input_dim, 1),
            input_shape=(input_dim,)
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
            num_classes,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_lstm(input_dim, num_classes):

    model = models.Sequential([

        layers.Reshape(
            (input_dim, 1),
            input_shape=(input_dim,)
        ),

        layers.LSTM(64),

        layers.Dense(
            64,
            activation='relu'
        ),

        layers.Dense(
            num_classes,
            activation='softmax'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ============================================================
# 14. DEPENDENT TEST
# ============================================================

print("\n################################")
print("DEPENDENT TEST")
print("################################")

X_all = np.concatenate(
    [datasets[k][0] for k in datasets]
)

y_all = np.concatenate(
    [datasets[k][1] for k in datasets]
)

X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.3,
    random_state=42,
    stratify=y_all
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

rf = train_rf(X_train_scaled, y_train)
rf_pred = rf.predict(X_test_scaled)
evaluate_model(y_test, rf_pred, le.classes_, "Dependent Test - RF")

svm = train_svm(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
evaluate_model(y_test, svm_pred, le.classes_, "Dependent Test - SVM")

dt = train_dt(X_train_scaled, y_train)
dt_pred = dt.predict(X_test_scaled)
evaluate_model(y_test, dt_pred, le.classes_, "Dependent Test - DT")

lr = train_lr(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
evaluate_model(y_test, lr_pred, le.classes_, "Dependent Test - LR")

bpnn = train_bpnn(X_train_scaled, y_train)
bpnn_pred = bpnn.predict(X_test_scaled)
evaluate_model(y_test, bpnn_pred, le.classes_, "Dependent Test - BPNN")

cnn = build_cnn(
    X_train_scaled.shape[1],
    len(np.unique(y_train))
)

cnn.fit(
    X_train_scaled,
    y_train,
    epochs=10,
    verbose=0
)

cnn_pred = np.argmax(
    cnn.predict(X_test_scaled),
    axis=1
)

evaluate_model(
    y_test,
    cnn_pred,
    le.classes_,
    "Dependent Test - CNN"
)

lstm = build_lstm(
    X_train_scaled.shape[1],
    len(np.unique(y_train))
)

lstm.fit(
    X_train_scaled,
    y_train,
    epochs=10,
    verbose=0
)

lstm_pred = np.argmax(
    lstm.predict(X_test_scaled),
    axis=1
)

evaluate_model(
    y_test,
    lstm_pred,
    le.classes_,
    "Dependent Test - LSTM"
)

preds = np.array([
    rf_pred,
    svm_pred,
    dt_pred,
    lr_pred,
    bpnn_pred,
    cnn_pred,
    lstm_pred
])

ensemble_pred = np.apply_along_axis(
    lambda x: np.bincount(x).argmax(),
    axis=0,
    arr=preds
)

evaluate_model(
    y_test,
    ensemble_pred,
    le.classes_,
    "Dependent Test - FINAL ENSEMBLE"
)

# ============================================================
# 15. BINARY FUNCTION
# ============================================================

def to_binary(y):

    return np.where(
        y == le.transform(["0"])[0],
        0,
        1
    )

# ============================================================
# 16. INDEPENDENT TEST
# ============================================================

for test_key in datasets:

    print("\n################################")
    print("TEST ON:", test_key)
    print("################################")

    X_test, y_test = datasets[test_key]

    X_train = []
    y_train = []

    for k in datasets:

        if k != test_key:

            X_train.append(datasets[k][0])

            y_train.append(datasets[k][1])

    X_train = np.concatenate(X_train)

    y_train = np.concatenate(y_train)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    rf = train_rf(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)

    svm = train_svm(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)

    dt = train_dt(X_train_scaled, y_train)
    dt_pred = dt.predict(X_test_scaled)

    lr = train_lr(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    bpnn = train_bpnn(X_train_scaled, y_train)
    bpnn_pred = bpnn.predict(X_test_scaled)

    cnn = build_cnn(
        X_train_scaled.shape[1],
        len(np.unique(y_train))
    )

    cnn.fit(
        X_train_scaled,
        y_train,
        epochs=10,
        verbose=0
    )

    cnn_pred = np.argmax(
        cnn.predict(X_test_scaled),
        axis=1
    )

    lstm = build_lstm(
        X_train_scaled.shape[1],
        len(np.unique(y_train))
    )

    lstm.fit(
        X_train_scaled,
        y_train,
        epochs=10,
        verbose=0
    )

    lstm_pred = np.argmax(
        lstm.predict(X_test_scaled),
        axis=1
    )

    evaluate_model(
        y_test,
        rf_pred,
        le.classes_,
        f"{test_key} - RF"
    )

    evaluate_model(
        y_test,
        svm_pred,
        le.classes_,
        f"{test_key} - SVM"
    )

    evaluate_model(
        y_test,
        dt_pred,
        le.classes_,
        f"{test_key} - DT"
    )

    evaluate_model(
        y_test,
        lr_pred,
        le.classes_,
        f"{test_key} - LR"
    )

    evaluate_model(
        y_test,
        bpnn_pred,
        le.classes_,
        f"{test_key} - BPNN"
    )

    evaluate_model(
        y_test,
        cnn_pred,
        le.classes_,
        f"{test_key} - CNN"
    )

    evaluate_model(
        y_test,
        lstm_pred,
        le.classes_,
        f"{test_key} - LSTM"
    )

    preds = np.array([
        rf_pred,
        svm_pred,
        dt_pred,
        lr_pred,
        bpnn_pred,
        cnn_pred,
        lstm_pred
    ])

    ensemble_pred = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(),
        axis=0,
        arr=preds
    )

    evaluate_model(
        y_test,
        ensemble_pred,
        le.classes_,
        f"{test_key} - FINAL ENSEMBLE"
    )

    y_test_bin = to_binary(y_test)

    ensemble_bin = to_binary(
        ensemble_pred
    )

    evaluate_model(
        y_test_bin,
        ensemble_bin,
        ["Loose", "Tight"],
        f"{test_key} - Ensemble (2-Class)"
    )

# ============================================================
# 17. EXPERIMENTAL TEST
# ============================================================

print("\n################################")
print("EXPERIMENTAL TEST")
print("################################")

scaler = StandardScaler()

X_all_scaled = scaler.fit_transform(X_all)

final_model = train_rf(
    X_all_scaled,
    y_all
)

results = {}

for folder in os.listdir(base_path):

    if not folder.endswith("_test"):
        continue

    flange_id = "F" + folder.split("_")[1]

    folder_path = os.path.join(
        base_path,
        folder
    )

    X_exp = []

    print("\nPredicting:", flange_id)

    for file in os.listdir(folder_path):

        if not (
            file.endswith(".mp4")
            or file.endswith(".m4a")
        ):
            continue

        file_path = os.path.join(
            folder_path,
            file
        )

        signal, sr = load_audio(file_path)

        hits = split_hits(signal, sr)

        for h in hits:

            features = extract_features(h, sr)

            X_exp.append(features)

    if len(X_exp) == 0:

        print("No audio found.")

        continue

    X_exp = np.array(X_exp)

    X_exp = scaler.transform(X_exp)

    preds = final_model.predict(X_exp)

    final = np.bincount(preds).argmax()

    label = le.inverse_transform([final])[0]

    results[flange_id] = label

    print(f"{flange_id} → {label} ft-lbs")

plt.figure(figsize=(8,5))

flanges = list(results.keys())

values = [
    int(results[k])
    for k in flanges
]

bars = plt.bar(
    flanges,
    values
)

plt.title(
    "Experimental Prediction Results"
)

plt.ylabel(
    "Predicted Torque (ft-lbs)"
)

for i, v in enumerate(values):

    plt.text(
        i,
        v + 1,
        str(v),
        ha='center'
    )

plt.show()

print("\n################################")
print("FINAL SUBMISSION FORMAT")
print("################################")

print(
    results.get("F1", "?"),
    results.get("F2", "?"),
    results.get("F3", "?"),
    results.get("F4", "?")
)

# ============================================================
# 18. OPTIONAL HMM BONUS MODEL
# ============================================================

print("\n################################")
print("HMM BONUS MODEL")
print("################################")

def train_hmm_models(X_train, y_train):

    hmm_models = {}

    for label in np.unique(y_train):

        class_data = X_train[
            y_train == label
        ]

        lengths = [1] * len(class_data)

        model = hmm.GaussianHMM(
            n_components=3,
            covariance_type='diag',
            n_iter=100,
            random_state=42
        )

        try:

            model.fit(class_data, lengths)

            hmm_models[label] = model

        except:

            pass

    return hmm_models


def hmm_predict(hmm_models, X_test):

    preds = []

    for x in X_test:

        scores = []

        labels = []

        for label in hmm_models:

            try:

                score = hmm_models[label].score(
                    x.reshape(1,-1)
                )

            except:

                score = -999999

            scores.append(score)

            labels.append(label)

        preds.append(
            labels[np.argmax(scores)]
        )

    return np.array(preds)

hmm_models = train_hmm_models(
    X_train_scaled,
    y_train
)

hmm_pred = hmm_predict(
    hmm_models,
    X_test_scaled
)

evaluate_model(
    y_test,
    hmm_pred,
    le.classes_,
    "HMM MODEL"
)
