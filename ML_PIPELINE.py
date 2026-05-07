import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import zipfile
import tempfile
import os
import re
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flange Looseness Detection",
    page_icon="🔩",
    layout="wide"
)

st.title("🔩 Flange Looseness Detection & Machine Learning")

st.caption("""
Upload a ZIP dataset containing percussion recordings.

Training files:
- 0ftlbF1A1.m4a
- 25ftlbF2A3.mp4
- 50ftlbF4A2.m4a

Unknown prediction files:
- F1A1.m4a
- F2A2.mp4
""")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📘 About")

    st.write("""
This application performs:

- Audio signal processing
- Feature extraction
- Multi-class torque classification
- Confusion matrix evaluation
- Unknown flange prediction

Supported labels:
- 0 ft-lb
- 25 ft-lb
- 50 ft-lb
""")

# ============================================================
# FILE PARSING
# ============================================================

def parse_training_filename(filename):

    filename = filename.replace(".mp4", "")
    filename = filename.replace(".m4a", "")
    filename = filename.replace(".wav", "")

    match = re.match(
        r"(\d+)ftlbF(\d)A(\d)",
        filename
    )

    if match:

        torque = int(match.group(1))
        flange = f"F{match.group(2)}"
        area = f"A{match.group(3)}"

        return torque, flange, area

    return None


def parse_unknown_filename(filename):

    filename = filename.replace(".mp4", "")
    filename = filename.replace(".m4a", "")
    filename = filename.replace(".wav", "")

    match = re.match(r"F(\d)A(\d)", filename)

    if match:

        flange = f"F{match.group(1)}"
        area = f"A{match.group(2)}"

        return flange, area

    return None

# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    try:

        signal, sr = librosa.load(path, sr=22050)

    except:

        signal, sr = sf.read(path)

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

    signal = signal.astype(np.float32)

    signal = (
        signal - np.mean(signal)
    ) / (np.std(signal) + 1e-9)

    return signal, sr

# ============================================================
# SPLIT HITS
# ============================================================

def split_hits(signal, sr):

    signal = signal / (
        np.max(np.abs(signal)) + 1e-9
    )

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

    for seg in segments:

        start = seg[0] * 512
        end = min(len(signal), seg[-1] * 512)

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

    spectral_rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=signal,
            sr=sr
        )
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    rms = np.mean(signal ** 2)

    fft = np.abs(np.fft.rfft(signal))
    fft_mean = np.mean(fft)

    return np.hstack([

        mfcc,

        spectral_centroid,
        spectral_rolloff,

        zcr,
        rms,
        fft_mean
    ])

# ============================================================
# PLOTTING
# ============================================================

def plot_waveform(signal):

    fig, ax = plt.subplots(figsize=(10,3))

    ax.plot(signal)

    ax.set_title("Waveform")

    ax.set_xlabel("Samples")

    ax.set_ylabel("Amplitude")

    st.pyplot(fig)


def plot_fft(signal, sr):

    fft = np.abs(np.fft.rfft(signal))

    freqs = np.fft.rfftfreq(
        len(signal),
        1/sr
    )

    fig, ax = plt.subplots(figsize=(10,3))

    ax.plot(freqs, fft)

    ax.set_title("FFT Spectrum")

    ax.set_xlabel("Frequency (Hz)")

    ax.set_ylabel("Magnitude")

    st.pyplot(fig)


def plot_confusion_matrix(cm, labels, title):

    fig, ax = plt.subplots(figsize=(6,5))

    ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_title(title)

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
    "📦 Upload ZIP Dataset",
    type=["zip"]
)

if uploaded_zip:

    temp_dir = tempfile.mkdtemp()

    zip_path = os.path.join(
        temp_dir,
        uploaded_zip.name
    )

    with open(zip_path, "wb") as f:
        f.write(uploaded_zip.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    X = []
    y = []

    metadata = []

    unknown_files = []

    audio_extensions = (
        ".m4a",
        ".mp4",
        ".wav"
    )

    # ========================================================
    # DATASET BUILDING
    # ========================================================

    for root, dirs, files in os.walk(temp_dir):

        for file in files:

            if not file.endswith(audio_extensions):
                continue

            full_path = os.path.join(root, file)

            parsed = parse_training_filename(file)

            if parsed is not None:

                torque, flange, area = parsed

                try:

                    signal, sr = load_audio(full_path)

                    hits = split_hits(signal, sr)

                    for h in hits:

                        features = extract_features(h, sr)

                        X.append(features)

                        y.append(torque)

                        metadata.append({

                            "file": file,
                            "flange": flange,
                            "area": area,
                            "torque": torque
                        })

                except:

                    st.warning(f"Skipping corrupted file: {file}")

            else:

                unknown = parse_unknown_filename(file)

                if unknown is not None:

                    unknown_files.append({

                        "file": file,
                        "path": full_path,
                        "flange": unknown[0],
                        "area": unknown[1]
                    })

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(X) == 0:

        st.error("No valid training files found.")
        st.stop()

    X = np.array(X)
    y = np.array(y)

    unique_classes = np.unique(y)

    if len(unique_classes) < 2:

        st.error("Need at least 2 torque classes.")
        st.stop()

    st.success(f"Loaded {len(X)} training samples")

    st.write("Detected Classes:")
    st.write(sorted(unique_classes))

    # ========================================================
    # SIGNAL ANALYSIS
    # ========================================================

    st.header("📈 Signal Analysis")

    sample_signal, sample_sr = load_audio(full_path)

    sample_hits = split_hits(
        sample_signal,
        sample_sr
    )

    if len(sample_hits) > 0:

        sample_hit = sample_hits[0]

        col1, col2 = st.columns(2)

        with col1:
            plot_waveform(sample_hit)

        with col2:
            plot_fft(sample_hit, sample_sr)

    # ========================================================
    # SPLIT
    # ========================================================

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

    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Random Forest":

            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),

        "SVM":

            SVC(
                probability=True
            ),

        "Decision Tree":

            DecisionTreeClassifier(
                random_state=42
            ),

        "Logistic Regression":

            LogisticRegression(
                max_iter=2000
            ),

        "KNN":

            KNeighborsClassifier()
    }

    # ========================================================
    # MODEL TRAINING
    # ========================================================

    st.header("🤖 Model Evaluation")

    results = []

    best_model = None
    best_acc = 0

    for name, model in models.items():

        try:

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            acc = accuracy_score(
                y_test,
                pred
            )

            cm = confusion_matrix(
                y_test,
                pred
            )

            results.append({

                "Model": name,
                "Accuracy": acc
            })

            if acc > best_acc:

                best_acc = acc
                best_model = model

            st.subheader(name)

            st.write(f"Accuracy: {acc:.4f}")

            plot_confusion_matrix(

                cm,

                sorted(unique_classes),

                f"{name} Confusion Matrix"
            )

            report = classification_report(

                y_test,
                pred,
                output_dict=True
            )

            st.dataframe(
                pd.DataFrame(report).transpose()
            )

        except Exception as e:

            st.warning(f"{name} failed: {e}")

    # ========================================================
    # RESULTS
    # ========================================================

    st.header("📊 Overall Results")

    results_df = pd.DataFrame(results)

    st.dataframe(results_df)

    if len(results_df) > 0:

        best_name = results_df.sort_values(

            "Accuracy",
            ascending=False

        ).iloc[0]["Model"]

        st.success(
            f"⭐ Best Model: {best_name}"
        )

    # ========================================================
    # UNKNOWN PREDICTION
    # ========================================================

    if len(unknown_files) > 0 and best_model is not None:

        st.header("🧪 Unknown Flange Prediction")

        prediction_rows = []

        for item in unknown_files:

            try:

                signal, sr = load_audio(
                    item["path"]
                )

                hits = split_hits(signal, sr)

                features = []

                for h in hits:

                    feat = extract_features(h, sr)

                    features.append(feat)

                if len(features) == 0:
                    continue

                features = scaler.transform(features)

                preds = best_model.predict(features)

                final_pred = np.bincount(preds).argmax()

                confidence = np.mean(
                    preds == final_pred
                )

                prediction_rows.append({

                    "File":
                        item["file"],

                    "Flange":
                        item["flange"],

                    "Area":
                        item["area"],

                    "Predicted Torque":
                        f"{final_pred} ft-lb",

                    "Confidence":
                        f"{confidence*100:.1f}%"
                })

            except Exception as e:

                st.warning(
                    f"Could not process {item['file']}"
                )

        if len(prediction_rows) > 0:

            pred_df = pd.DataFrame(
                prediction_rows
            )

            st.dataframe(pred_df)

    # ========================================================
    # FLANGE DISTRIBUTION
    # ========================================================

    st.header("🔩 Flange Distribution")

    metadata_df = pd.DataFrame(metadata)

    if len(metadata_df) > 0:

        flange_counts = (

            metadata_df
            .groupby(["flange", "torque"])
            .size()
            .unstack(fill_value=0)
        )

        st.bar_chart(flange_counts)
