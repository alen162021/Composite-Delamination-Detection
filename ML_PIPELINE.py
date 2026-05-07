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
    confusion_matrix,
    classification_report
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

st.title("🔩 Smart Flange Identification & Torque Classification Lab")

st.markdown("""
This application performs:

- Automatic percussion hit detection
- Audio feature extraction
- Torque classification (0 / 25 / 50 ft-lb)
- Multi-model machine learning comparison
- Confusion matrix evaluation
- Unknown flange prediction
- Overall flange health estimation
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
# SPLIT HITS
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
# PARSE FILENAMES
# ============================================================

def parse_filename(filename):

    filename = filename.replace(".mp4", "")
    filename = filename.replace(".m4a", "")
    filename = filename.replace(".wav", "")

    train_match = re.match(
        r"(\d+)ftlbF(\d)A(\d)",
        filename
    )

    if train_match:

        torque = train_match.group(1)
        flange = "F" + train_match.group(2)
        area = "A" + train_match.group(3)

        return {
            "type": "train",
            "torque": torque,
            "flange": flange,
            "area": area
        }

    test_match = re.match(
        r"F(\d)A(\d)",
        filename
    )

    if test_match:

        flange = "F" + test_match.group(1)
        area = "A" + test_match.group(2)

        return {
            "type": "test",
            "flange": flange,
            "area": area
        }

    return None

# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(cm, labels, title):

    fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(cm)

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
# ZIP EXTRACTION
# ============================================================

uploaded_zip = st.file_uploader(
    "Upload ZIP Dataset",
    type=["zip"]
)

if uploaded_zip is not None:

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

        unknown_files = []

        flange_summary = {}

        st.header("📂 Dataset Processing")

        all_audio_files = []

        for root, dirs, files in os.walk(temp_dir):

            for file in files:

                if file.endswith((".mp4", ".m4a", ".wav")):

                    all_audio_files.append(
                        os.path.join(root, file)
                    )

        progress = st.progress(0)

        for idx, path in enumerate(all_audio_files):

            progress.progress(
                (idx + 1) / len(all_audio_files)
            )

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

                unknown_files.append({
                    "path": path,
                    "info": parsed
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

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ====================================================
        # MODELS
        # ====================================================

        models = {

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=150,
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
                    max_iter=1000
                ),

            "BPNN":
                MLPClassifier(
                    hidden_layer_sizes=(128,64),
                    max_iter=500,
                    random_state=42
                )
        }

        st.header("🤖 Model Evaluation")

        results = []

        best_model = None
        best_acc = 0

        for name, model in models.items():

            st.subheader(name)

            model.fit(
                X_train_scaled,
                y_train
            )

            preds = model.predict(
                X_test_scaled
            )

            acc = accuracy_score(
                y_test,
                preds
            )

            results.append({
                "Model": name,
                "Accuracy": acc
            })

            st.metric(
                "Accuracy",
                f"{acc:.4f}"
            )

            cm = confusion_matrix(
                y_test,
                preds
            )

            labels = sorted(list(set(y)))

            plot_confusion_matrix(
                cm,
                labels,
                f"{name} Confusion Matrix"
            )

            report = classification_report(
                y_test,
                preds,
                output_dict=True
            )

            st.dataframe(
                pd.DataFrame(report).transpose()
            )

            if acc > best_acc:

                best_acc = acc
                best_model = model

        # ====================================================
        # OVERALL MODEL COMPARISON
        # ====================================================

        st.header("📊 Overall Model Comparison")

        result_df = pd.DataFrame(results)

        st.dataframe(result_df)

        fig, ax = plt.subplots()

        ax.bar(
            result_df["Model"],
            result_df["Accuracy"]
        )

        ax.set_ylim([0,1])

        ax.set_ylabel("Accuracy")

        plt.xticks(rotation=15)

        st.pyplot(fig)

        # ====================================================
        # UNKNOWN FLANGE PREDICTION
        # ====================================================

        st.header("🔩 Intelligent Flange Identification")

        st.markdown("""
        The system now estimates the most likely
        torque condition for each unknown flange.
        """)

        pipe_cols = st.columns(4)

        flange_results = []

        for idx, unknown in enumerate(unknown_files):

            signal, sr = load_audio(
                unknown["path"]
            )

            hits = split_hits(signal, sr)

            if len(hits) == 0:
                continue

            features = np.array([
                extract_features(h, sr)
                for h in hits
            ])

            features_scaled = scaler.transform(
                features
            )

            preds = best_model.predict(
                features_scaled
            )

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
                unknown["info"]["flange"]
                + unknown["info"]["area"]
            )

            flange_results.append({

                "Flange": flange_name,
                "Prediction": final_label,
                "Confidence": confidence
            })

            with pipe_cols[idx % 4]:

                st.markdown("## 🔩")

                st.markdown(f"""
                ### {flange_name}

                **Estimated Torque:**  
                `{final_label} ft-lb`

                **Confidence:**  
                `{confidence:.1f}%`
                """)

                if int(final_label) == 0:

                    st.error(
                        "Likely Loose"
                    )

                elif int(final_label) == 25:

                    st.warning(
                        "Moderately Tight"
                    )

                else:

                    st.success(
                        "Tight"
                    )

        # ====================================================
        # PIPE VISUALIZATION
        # ====================================================

        st.header("🛠 Overall Pipe Configuration")

        pipe_text = ""

        for item in flange_results:

            pipe_text += (
                f"[ {item['Flange']} → "
                f"{item['Prediction']} ft-lb ]"
            )

            pipe_text += " ===== "

        st.code(pipe_text)

        # ====================================================
        # FINAL TABLE
        # ====================================================

        st.header("📋 Final Flange Assessment")

        flange_df = pd.DataFrame(flange_results)

        st.dataframe(flange_df)

        # ====================================================
        # SIGNAL VISUALIZATION
        # ====================================================

        st.header("📈 Example Signal Analysis")

        if len(all_audio_files) > 0:

            example = all_audio_files[0]

            signal, sr = load_audio(example)

            fig, ax = plt.subplots(
                figsize=(10,3)
            )

            ax.plot(signal)

            ax.set_title(
                "Percussion Waveform"
            )

            st.pyplot(fig)

            fft = np.fft.rfft(signal)

            freqs = np.fft.rfftfreq(
                len(signal),
                1/sr
            )

            fig2, ax2 = plt.subplots(
                figsize=(10,3)
            )

            ax2.plot(
                freqs,
                np.abs(fft)
            )

            ax2.set_title(
                "FFT Spectrum"
            )

            st.pyplot(fig2)

            mel = librosa.feature.melspectrogram(
                y=signal,
                sr=sr
            )

            mel_db = librosa.power_to_db(
                mel,
                ref=np.max
            )

            fig3, ax3 = plt.subplots(
                figsize=(10,4)
            )

            img = librosa.display.specshow(
                mel_db,
                sr=sr,
                x_axis='time',
                y_axis='mel',
                ax=ax3
            )

            plt.colorbar(img)

            ax3.set_title(
                "Mel Spectrogram"
            )

            st.pyplot(fig3)

else:

    st.info("""
    Upload a ZIP dataset containing files such as:

    TRAINING:
    - 0ftlbF1A1.wav
    - 25ftlbF2A2.mp4
    - 50ftlbF4A3.m4a

    TESTING:
    - F1A1.wav
    - F2A2.wav

    The system will:
    - learn torque patterns
    - compare ML models
    - generate confusion matrices
    - estimate unknown flange conditions
    """)
