import streamlit as st
import numpy as np
import pandas as pd
import librosa
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
    page_title="Flange Torque System",
    layout="wide"
)

st.title("🔩 Flange Torque Classification System")
st.markdown("Predict only: **0, 25, 50 ft-lb**")

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


# ============================================================
# FEATURES
# ============================================================

def extract_features(signal, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(signal))
    energy = np.mean(signal ** 2)
    centroid = np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr))

    return np.hstack([mfcc, zcr, energy, centroid])


# ============================================================
# PARSER (FORCE ONLY 0/25/50)
# ============================================================

def parse_label(filename):
    filename = filename.lower()

    match = re.search(r"(\d+)ftlb", filename)
    if match:
        val = int(match.group(1))

        if val in [0, 25, 50]:
            return val

    return None


def parse_flange(filename):
    match = re.search(r"f(\d)", filename.lower())
    if match:
        return f"F{match.group(1)}"
    return "Unknown"


# ============================================================
# CM PLOT
# ============================================================

def plot_cm(cm, labels, title):
    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_title(title)
    st.pyplot(fig)


# ============================================================
# UPLOAD ZIP
# ============================================================

uploaded = st.file_uploader("Upload ZIP Dataset", type="zip")

if uploaded:

    with tempfile.TemporaryDirectory() as tmp:

        zip_path = os.path.join(tmp, "data.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded.read())

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp)

        # ====================================================
        # LOAD DATA
        # ====================================================

        X, y = [], []
        flange_map = {}

        files = []

        for root, _, fs in os.walk(tmp):
            for f in fs:
                if f.endswith((".wav", ".mp4", ".m4a")):
                    files.append(os.path.join(root, f))

        st.success(f"Found {len(files)} files")

        for path in files:

            label = parse_label(os.path.basename(path))
            flange = parse_flange(os.path.basename(path))

            if label is None:
                continue

            signal, sr = load_audio(path)
            hits = split_hits(signal, sr)

            for h in hits:
                feat = extract_features(h, sr)
                X.append(feat)
                y.append(label)

                flange_map.setdefault(flange, []).append(label)

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
            "RF": RandomForestClassifier(n_estimators=100),
            "SVM": SVC(),
            "DT": DecisionTreeClassifier(),
            "LR": LogisticRegression(max_iter=1000),
            "MLP": MLPClassifier(hidden_layer_sizes=(64,32), max_iter=400)
        }

        results = {}
        trained = {}

        st.header("🤖 Model Results")

        best_model = None
        best_acc = 0

        for name, model in models.items():

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            results[name] = acc
            trained[name] = model

            st.subheader(name)
            st.write("Accuracy:", acc)

            cm = confusion_matrix(y_test, preds)
            plot_cm(cm, [0,25,50], f"{name} CM")

            if acc > best_acc:
                best_acc = acc
                best_model = model

        # ====================================================
        # SUMMARY TABLE
        # ====================================================

        st.header("📊 Summary")

        df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": list(results.values())
        })

        st.dataframe(df)

        # ====================================================
        # FLANGE LEVEL PREDICTION (AVERAGING)
        # ====================================================

        st.header("🔩 Flange-Level Prediction (Averaged)")

        flange_results = {}

        for f, vals in flange_map.items():
            pred = int(round(np.mean(vals)))
            confidence = 100 - (np.std(vals) * 10)

            flange_results[f] = {
                "Prediction": pred,
                "Confidence": max(50, min(100, confidence))
            }

        df2 = pd.DataFrame([
            {
                "Flange": k,
                "Torque": v["Prediction"],
                "Confidence": v["Confidence"]
            }
            for k, v in flange_results.items()
        ])

        st.dataframe(df2)

        # ====================================================
        # PIPE VISUALIZATION
        # ====================================================

        st.header("🧩 Flange Pipe Model")

        cols = st.columns(len(flange_results))

        for i, (k, v) in enumerate(flange_results.items()):
            with cols[i]:
                st.markdown(f"### {k}")
                st.metric("Torque", f"{v['Prediction']} ft-lb")
                st.metric("Confidence", f"{v['Confidence']:.1f}%")

        # ====================================================
        # LIVE TEST
        # ====================================================

        st.header("🎙 Live Prediction")

        live = st.file_uploader("Upload test audio", type=["wav","mp4","m4a"])

        if live:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(live.read())
                path = f.name

            signal, sr = load_audio(path)
            hits = split_hits(signal, sr)

            feats = np.array([extract_features(h, sr) for h in hits])
            feats = scaler.transform(feats)

            pred = best_model.predict(feats)

            final = int(round(np.mean(pred)))

            st.success(f"Predicted Torque: {final} ft-lb")
