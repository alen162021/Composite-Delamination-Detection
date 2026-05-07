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
    page_title="Smart Flange Identification Lab",
    page_icon="🔩",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🔩 Smart Flange Identification & Torque Classification")

st.markdown("""
This application performs:

- Percussion hit detection
- Torque classification
- Multi-model ML comparison
- Confusion matrix generation
- Unknown flange prediction
- LIVE demonstration prediction
""")

# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(path):

    try:

        signal, sr = librosa.load(
            path,
            sr=48000
        )

    except:

        signal, sr = sf.read(path)

        if len(signal.shape) > 1:

            signal = np.mean(
                signal,
                axis=1
            )

    signal = (
        signal - np.mean(signal)
    ) / (np.std(signal) + 1e-9)

    return signal, sr

# ============================================================
# SPLIT HITS
# ============================================================

def split_hits(signal, sr):

    energy = librosa.feature.rms(
        y=signal
    )[0]

    threshold = (
        np.mean(energy)
        + 0.5 * np.std(energy)
    )

    frames = np.where(
        energy > threshold
    )[0]

    if len(frames) < 5:

        return [signal]

    segments = np.split(
        frames,
        np.where(
            np.diff(frames) > 2
        )[0] + 1
    )

    hits = []

    for s in segments:

        start = s[0] * 512

        end = min(
            len(signal),
            s[-1] * 512
        )

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

        librosa.feature.zero_crossing_rate(
            signal
        )
    )

    energy = np.mean(
        signal ** 2
    )

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

        return {

            "type": "train",

            "torque":
                train_match.group(1),

            "flange":
                "F" + train_match.group(2),

            "area":
                "A" + train_match.group(3)
        }

    # UNKNOWN FILES
    unknown_match = re.search(
        r"f(\d)a(\d)",
        filename
    )

    if unknown_match:

        return {

            "type": "unknown",

            "flange":
                "F" + unknown_match.group(1),

            "area":
                "A" + unknown_match.group(2)
        }

    return None

# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_cm(cm, labels, title):

    fig, ax = plt.subplots(
        figsize=(5,5)
    )

    ax.imshow(cm)

    ax.set_xticks(
        np.arange(len(labels))
    )

    ax.set_yticks(
        np.arange(len(labels))
    )

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
# MAIN
# ============================================================

uploaded_zip = st.file_uploader(
    "Upload DATA ZIP",
    type=["zip"]
)

if uploaded_zip:

    with tempfile.TemporaryDirectory() as temp_dir:

        zip_path = os.path.join(
            temp_dir,
            uploaded_zip.name
        )

        with open(zip_path, "wb") as f:

            f.write(
                uploaded_zip.read()
            )

        with zipfile.ZipFile(
            zip_path,
            "r"
        ) as zip_ref:

            zip_ref.extractall(temp_dir)

        # ====================================================
        # LOAD DATA
        # ====================================================

        X = []
        y = []

        unknown_files = []

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

        st.success(
            f"Detected {len(audio_files)} audio files"
        )

        # ====================================================
        # PROCESS DATA
        # ====================================================

        for path in audio_files:

            filename = os.path.basename(path)

            parsed = parse_filename(
                filename
            )

            if parsed is None:
                continue

            signal, sr = load_audio(path)

            hits = split_hits(
                signal,
                sr
            )

            if parsed["type"] == "train":

                for h in hits:

                    X.append(
                        extract_features(h, sr)
                    )

                    y.append(
                        parsed["torque"]
                    )

            else:

                unknown_files.append({

                    "path": path,

                    "flange":
                        parsed["flange"],

                    "area":
                        parsed["area"]
                })

        X = np.array(X)
        y = np.array(y)

        st.success(
            f"Training Samples: {len(X)}"
        )

        # ====================================================
        # TRAIN / TEST SPLIT
        # ====================================================

        X_train, X_test, y_train, y_test = train_test_split(

            X,
            y,

            test_size=0.3,

            random_state=42,

            stratify=y
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(
            X_train
        )

        X_test = scaler.transform(
            X_test
        )

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
                LogisticRegression(
                    max_iter=1000
                ),

            "BPNN":
                MLPClassifier(
                    hidden_layer_sizes=(128,64),
                    max_iter=500
                )
        }

        st.header("🤖 Model Evaluation")

        best_model = None
        best_acc = 0

        results = []

        for name, model in models.items():

            st.subheader(name)

            model.fit(
                X_train,
                y_train
            )

            preds = model.predict(
                X_test
            )

            acc = accuracy_score(
                y_test,
                preds
            )

            results.append({

                "Model": name,
                "Accuracy": round(acc,4)
            })

            st.metric(
                "Accuracy",
                round(acc,4)
            )

            cm = confusion_matrix(
                y_test,
                preds
            )

            labels = sorted(
                list(set(y))
            )

            plot_cm(
                cm,
                labels,
                f"{name} Confusion Matrix"
            )

            if acc > best_acc:

                best_acc = acc
                best_model = model

        # ====================================================
        # MODEL COMPARISON TABLE
        # ====================================================

        st.header("📊 Overall Model Comparison")

        st.dataframe(
            pd.DataFrame(results)
        )

        # ====================================================
        # TEST FILE PREDICTION
        # ====================================================

        st.header("🔩 Unknown Flange Prediction")

        if len(unknown_files) == 0:

            st.warning("""
            No unknown flange files found.

            Unknown files should look like:
            - F1A1.wav
            - F2A3.mp4
            """)

        else:

            flange_results = []

            for item in unknown_files:

                signal, sr = load_audio(
                    item["path"]
                )

                hits = split_hits(
                    signal,
                    sr
                )

                features = np.array([

                    extract_features(
                        h,
                        sr
                    )

                    for h in hits
                ])

                features = scaler.transform(
                    features
                )

                preds = best_model.predict(
                    features
                )

                unique, counts = np.unique(
                    preds,
                    return_counts=True
                )

                final_prediction = unique[
                    np.argmax(counts)
                ]

                confidence = (
                    np.max(counts)
                    / np.sum(counts)
                ) * 100

                flange_name = (
                    item["flange"]
                    + item["area"]
                )

                flange_results.append({

                    "Flange":
                        flange_name,

                    "Predicted Torque":
                        f"{final_prediction} ft-lb",

                    "Confidence":
                        f"{confidence:.1f}%"
                })

            # ================================================
            # RESULTS TABLE
            # ================================================

            st.dataframe(
                pd.DataFrame(flange_results)
            )

            # ================================================
            # PIPE VISUALIZATION
            # ================================================

            st.header("🛠 Pipe Configuration")

            cols = st.columns(
                len(flange_results)
            )

            for idx, result in enumerate(flange_results):

                with cols[idx]:

                    st.markdown("# 🔩")

                    st.markdown(
                        f"### {result['Flange']}"
                    )

                    st.metric(
                        "Torque",
                        result["Predicted Torque"]
                    )

                    st.metric(
                        "Confidence",
                        result["Confidence"]
                    )

        # ====================================================
        # LIVE DEMONSTRATION
        # ====================================================

        st.header("🎙 Live Demonstration")

        st.markdown("""
        Record or upload a NEW percussion test.

        The trained model will estimate:
        - torque level
        - flange condition
        - confidence
        """)

        live_audio = st.audio_input(
            "Record Percussion Test"
        )

        uploaded_live = st.file_uploader(
            "OR Upload Live Test",
            type=["wav","mp4","m4a"],
            key="live"
        )

        live_file = None

        if live_audio is not None:

            live_file = live_audio

        elif uploaded_live is not None:

            live_file = uploaded_live

        if live_file is not None:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(
                    live_file.read()
                )

                temp_audio_path = tmp.name

            signal, sr = load_audio(
                temp_audio_path
            )

            hits = split_hits(
                signal,
                sr
            )

            st.success(
                f"Detected {len(hits)} percussion hits"
            )

            # ================================================
            # WAVEFORM
            # ================================================

            fig, ax = plt.subplots(
                figsize=(10,3)
            )

            ax.plot(signal)

            ax.set_title(
                "Live Percussion Signal"
            )

            st.pyplot(fig)

            # ================================================
            # PREDICTION
            # ================================================

            features = np.array([

                extract_features(
                    h,
                    sr
                )

                for h in hits
            ])

            features = scaler.transform(
                features
            )

            preds = best_model.predict(
                features
            )

            unique, counts = np.unique(
                preds,
                return_counts=True
            )

            final_prediction = unique[
                np.argmax(counts)
            ]

            confidence = (
                np.max(counts)
                / np.sum(counts)
            ) * 100

            st.header("🔍 LIVE RESULT")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Estimated Torque",
                    f"{final_prediction} ft-lb"
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )

            # ================================================
            # CONDITION
            # ================================================

            if int(final_prediction) == 0:

                st.error("""
                LIKELY LOOSE FLANGE
                """)

            elif int(final_prediction) == 25:

                st.warning("""
                MODERATELY TIGHT
                """)

            else:

                st.success("""
                TIGHT / HEALTHY
                """)

else:

    st.info("""
    Upload a ZIP containing:

    TRAINING:
    - 0ftlbF1A1.wav
    - 25ftlbF2A2.wav
    - 50ftlbF3A1.wav

    TESTING:
    - F1A1.wav
    - F2A2.wav

    Then:
    - models train automatically
    - confusion matrices are generated
    - unknown flanges are predicted
    - live demonstrations become available
    """)
