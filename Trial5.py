import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import zipfile
import os
import re
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Flange Lab",
    page_icon="🔩",
    layout="wide"
)

st.title("🔩 Smart Flange Identification System")

LABELS = ["0", "25", "50"]

# ============================================================
# AUDIO
# ============================================================

def load_audio(path):
    try:
        signal, sr = librosa.load(path, sr=48000)
    except:
        signal, sr = sf.read(path)
        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
    return signal, sr


def split_hits(signal, sr):
    energy = librosa.feature.rms(y=signal)[0]

    threshold = np.mean(energy) + 0.5 * np.std(energy)
    frames = np.where(energy > threshold)[0]

    if len(frames) < 5:
        return [signal]

    segments = np.split(frames, np.where(np.diff(frames) > 2)[0] + 1)

    hits = []
    for s in segments:
        start = s[0] * 512
        end = min(len(signal), s[-1] * 512)
        hit = signal[start:end]
        if len(hit) > 1000:
            hits.append(hit)

    return hits


def extract_features(signal, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    energy = np.mean(signal ** 2)

    return np.hstack([mfcc, centroid, zcr, energy])


def parse_filename(filename):
    filename = filename.lower()

    filename = filename.replace(".wav", "")
    filename = filename.replace(".mp4", "")
    filename = filename.replace(".m4a", "")

    train_match = re.search(r"(\d+)ftlbf(\d)a(\d)", filename)

    if train_match:
        return {
            "type": "train",
            "torque": train_match.group(1),
            "flange": "F" + train_match.group(2),
            "area": "A" + train_match.group(3)
        }

    unknown_match = re.search(r"f(\d)a(\d)", filename)

    if unknown_match:
        return {
            "type": "unknown",
            "flange": "F" + unknown_match.group(1),
            "area": "A" + unknown_match.group(2)
        }

    return None


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(cm)

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))

    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)


# ============================================================
# UPLOAD ZIP
# ============================================================

uploaded_zip = st.file_uploader("Upload ZIP dataset", type=["zip"])

if uploaded_zip:

    with tempfile.TemporaryDirectory() as tmp:

        zip_path = os.path.join(tmp, "data.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        X, y = [], []
        unknown = []

        files = []

        for root, _, fs in os.walk(tmp):
            for f in fs:
                if f.endswith((".wav", ".mp4", ".m4a")):
                    files.append(os.path.join(root, f))

        st.success(f"Loaded {len(files)} files")

        # ====================================================
        # PROCESS DATA
        # ====================================================

        for path in files:

            parsed = parse_filename(os.path.basename(path))
            if not parsed:
                continue

            signal, sr = load_audio(path)
            hits = split_hits(signal, sr)

            if parsed["type"] == "train":
                for h in hits:
                    X.append(extract_features(h, sr))
                    y.append(parsed["torque"])

            else:
                unknown.append((path, parsed["flange"], parsed["area"]))

        X = np.array(X)
        y = np.array(y)

        st.success(f"Training samples: {len(X)}")

        # ====================================================
        # SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ====================================================
        # MODELS
        # ====================================================

        models = {
            "RF": RandomForestClassifier(),
            "SVM": SVC(),
            "DT": DecisionTreeClassifier(),
            "LR": LogisticRegression(max_iter=1000),
            "MLP": MLPClassifier(max_iter=500)
        }

        st.header("🤖 Model Results")

        results = {}

        best_model = None
        best_acc = 0

        for name, model in models.items():

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            results[name] = acc

            st.subheader(name)
            st.metric("Accuracy", round(acc, 4))

            cm = confusion_matrix(y_test, pred, labels=LABELS)
            plot_cm(cm, f"{name} Confusion Matrix")

            if acc > best_acc:
                best_acc = acc
                best_model = model

        # ====================================================
        # OVERALL COMPARISON
        # ====================================================

        st.header("📊 Model Comparison")
        st.dataframe(pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": list(results.values())
        }))

        # ====================================================
        # FLANGE LEVEL SUMMARY (IMPORTANT PART YOU WANTED)
        # ====================================================

        st.header("🔩 Flange-Level Prediction Summary")

        flange_map = {}

        for path, flange, area in unknown:

            signal, sr = load_audio(path)
            hits = split_hits(signal, sr)

            feats = np.array([extract_features(h, sr) for h in hits])
            feats = scaler.transform(feats)

            preds = best_model.predict(feats)

            final = int(round(np.mean([int(p) for p in preds])))

            if flange not in flange_map:
                flange_map[flange] = []

            flange_map[flange].append(final)

        final_flange = []

        for f, vals in flange_map.items():

            avg = int(round(np.mean(vals)))
            confidence = 1 - np.std(vals)

            final_flange.append({
                "Flange": f,
                "Estimated Torque": avg,
                "Confidence": round(confidence, 3)
            })

        st.dataframe(pd.DataFrame(final_flange))
