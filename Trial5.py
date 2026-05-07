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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Flange ML Lab",
    page_icon="🔩",
    layout="wide"
)

st.title("🔩 Smart Flange Identification & Torque Classification")

st.markdown("""
Upload an entire ZIP dataset.

The app will automatically:
- detect labeled training files
- train ML models
- compare performance
- generate confusion matrices
- predict unknown flange torque
""")

# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    try:
        signal, sr = librosa.load(path, sr=48000)

    except:

        signal, sr = sf.read(path)

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

    signal = (
        signal - np.mean(signal)
    ) / (np.std(signal) + 1e-9)

    return signal, sr

# ============================================================
# HIT SPLITTING
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
# FEATURE EXTRACTION
# ============================================================

def extract_features(signal, sr):

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=13
        ),
        axis=1
    )

    spectral_centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sr
        )
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    energy = np.mean(signal ** 2)

    return np.hstack([
        mfcc,
        spectral_centroid,
        zcr,
        energy
    ])

# ============================================================
# PARSE FILENAMES
# ============================================================

def parse_filename(filename):

    filename = filename.lower()

    filename = filename.replace(".wav", "")
    filename = filename.replace(".mp4", "")
    filename = filename.replace(".m4a", "")

    # TRAINING FILES
    train_match = re.search(
        r"(\d+)ftlbf(\d)a(\d)",
        filename
    )

    if train_match:

        torque = train_match.group(1)

        flange = f"F{train_match.group(2)}"
        area = f"A{train_match.group(3)}"

        return {
            "type": "train",
            "torque": torque,
            "flange": flange,
            "area": area
        }

    # UNKNOWN FILES
    unknown_match = re.search(
        r"f(\d)a(\d)",
        filename
    )

    if unknown_match:

        flange = f"F{unknown_match.group(1)}"
        area = f"A{unknown_match.group(2)}"

        return {
            "type": "unknown",
            "flange": flange,
            "area": area
        }

    return None

# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_cm(cm, labels, title):

    fig, ax = plt.subplots(figsize=(5,5))

    ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    st.pyplot(fig)

# ============================================================
# ZIP UPLOAD
# ============================================================

uploaded_zip = st.file_uploader(
    "Upload ZIP Dataset",
    type=["zip"]
)

if uploaded_zip:

    with tempfile.TemporaryDirectory() as temp_dir:

        zip_path = os.path.join(
            temp_dir,
            uploaded_zip.name
        )

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)

        # ====================================================
        # LOAD DATA
        # ====================================================

        X = []
        y = []

        unknown_data = []

        audio_files = []

        for root, dirs, files in os.walk(temp_dir):

            for file in files:

                if file.endswith((
                    ".wav",
                    ".mp4",
                    ".m4a"
                )):

                    audio_files.append(
                        os.path.join(root, file)
                    )

        st.success(f"Detected {len(audio_files)} audio files")

        # ====================================================
        # PROCESS FILES
        # ====================================================

        for path in audio_files:

            filename = os.path.basename(path)

            parsed = parse_filename(filename)

            if parsed is None:
                continue

            signal, sr = load_audio(path)

            hits = split_hits(signal, sr)

            if parsed["type"] == "train":

                for h in hits:

                    X.append(
                        extract_features(h, sr)
                    )

                    y.append(parsed["torque"])

            else:

                unknown_data.append({
                    "path": path,
                    "parsed": parsed
                })

        X = np.array(X)
        y = np.array(y)

        st.success(f"Training Samples: {len(X)}")

        # ====================================================
        # TRAIN TEST SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # ====================================================
        # MODELS
        # ====================================================

        models = {

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),

            "SVM":
                SVC(probability=True),

            "Decision Tree":
                DecisionTreeClassifier(),

            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "BPNN":
                MLPClassifier(
                    hidden_layer_sizes=(128,64),
                    max_iter=500
                )
        }

        st.header("🤖 Model Evaluation")

        best_model = None
        best_acc = 0

        for name, model in models.items():

            st.subheader(name)

            model.fit(X_train, y_train)

            preds = model.predict(X_test)

            acc = accuracy_score(
                y_test,
                preds
            )

            st.metric(
                "Accuracy",
                f"{acc:.4f}"
            )

            cm = confusion_matrix(
                y_test,
                preds
            )

            labels = sorted(list(set(y)))

            plot_cm(
                cm,
                labels,
                f"{name} Confusion Matrix"
            )

            if acc > best_acc:

                best_acc = acc
                best_model = model

        # ====================================================
        # UNKNOWN FLANGE PREDICTION
        # ====================================================

        st.header("🛠 Overall Flange Identification")

        flange_results = []

        cols = st.columns(4)

        for idx, item in enumerate(unknown_data):

            signal, sr = load_audio(
                item["path"]
            )

            hits = split_hits(signal, sr)

            features = np.array([
                extract_features(h, sr)
                for h in hits
            ])

            features = scaler.transform(features)

            preds = best_model.predict(features)

            unique, counts = np.unique(
                preds,
                return_counts=True
            )

            final_label = unique[np.argmax(counts)]

            confidence = (
                np.max(counts)
                / np.sum(counts)
            ) * 100

            flange_name = (
                item["parsed"]["flange"]
                + item["parsed"]["area"]
            )

            flange_results.append({
                "Flange": flange_name,
                "Prediction": final_label,
                "Confidence": confidence
            })

            with cols[idx % 4]:

                st.markdown("""
                # 🔩
                ### PIPE FLANGE
                """)

                st.write(flange_name)

                st.metric(
                    "Estimated Torque",
                    f"{final_label} ft-lb"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )

                if int(final_label) == 0:

                    st.error("LOOSE")

                elif int(final_label) == 25:

                    st.warning("MODERATE")

                else:

                    st.success("TIGHT")

        # ====================================================
        # PIPE VISUALIZATION
        # ====================================================

        st.header("🔗 Pipe Configuration")

        pipe = ""

        for item in flange_results:

            pipe += (
                f"[ {item['Flange']} "
                f"→ {item['Prediction']} ft-lb ]"
            )

            pipe += " ===== "

        st.code(pipe)

        # ====================================================
        # FINAL TABLE
        # ====================================================

        st.header("📋 Final Flange Assessment")

        st.dataframe(
            pd.DataFrame(flange_results)
        )
