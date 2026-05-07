import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf
import zipfile
import os
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Flange Detection",
    page_icon="🔩",
    layout="wide"
)

st.title("🔩 Smart Flange Loosening Detection")

st.write("""
Upload ONE zip file containing:
- labeled training files
- unlabeled experimental files

Example:
- 0ftlbF1A1.wav
- 25ftlbF2A2.wav
- 50ftlbF3A1.wav
- F1A2.wav
- F4A1.wav
""")

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

    signal = signal / (np.max(np.abs(signal)) + 1e-9)

    return signal, sr

# ============================================================
# SPLIT IMPACTS
# ============================================================

def split_hits(signal, sr):

    energy = librosa.feature.rms(y=signal)[0]

    threshold = np.mean(energy) * 1.5

    frames = np.where(energy > threshold)[0]

    if len(frames) == 0:
        return []

    segments = np.split(
        frames,
        np.where(np.diff(frames) > 2)[0] + 1
    )

    hits = []

    for seg in segments:

        start = seg[0] * 512
        end = seg[-1] * 512

        hit = signal[start:end]

        if len(hit) > 1000:
            hits.append(hit)

    return hits

# ============================================================
# FEATURES
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

    centroid = np.mean(
        librosa.feature.spectral_centroid(
            y=signal,
            sr=sr
        )
    )

    rolloff = np.mean(
        librosa.feature.spectral_rolloff(
            y=signal,
            sr=sr
        )
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    energy = np.mean(signal**2)

    return np.hstack([
        mfcc,
        centroid,
        rolloff,
        zcr,
        energy
    ])

# ============================================================
# PARSE FILENAME
# ============================================================

def parse_filename(name):

    name = name.lower()

    # TRAINING FILE
    match = re.search(
        r"(\d+)ftlb",
        name
    )

    if match:

        torque = int(match.group(1))

        return torque

    # EXPERIMENTAL FILE
    return None

# ============================================================
# DRAW FLANGE
# ============================================================

def draw_flange(prediction):

    fig, ax = plt.subplots(figsize=(4,4))

    circle = plt.Circle((0,0),1,fill=False,linewidth=6)

    ax.add_artist(circle)

    angles = np.linspace(0,2*np.pi,8,endpoint=False)

    for a in angles:

        x = np.cos(a)
        y = np.sin(a)

        if prediction <= 10:
            color = "red"
            size = 450

        elif prediction <= 30:
            color = "orange"
            size = 380

        else:
            color = "green"
            size = 300

        ax.scatter(x,y,s=size,c=color)

    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)

    ax.axis("off")

    ax.set_aspect("equal")

    st.pyplot(fig)

# ============================================================
# ZIP UPLOAD
# ============================================================

uploaded_zip = st.file_uploader(
    "Upload ZIP Dataset",
    type=["zip"]
)

if uploaded_zip:

    with tempfile.TemporaryDirectory() as tmpdir:

        zip_path = os.path.join(tmpdir, "data.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        extract_dir = os.path.join(tmpdir, "data")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        st.success("ZIP Extracted")

        X_train = []
        y_train = []

        experimental = []

        # ====================================================
        # READ FILES
        # ====================================================

        for root, dirs, files in os.walk(extract_dir):

            for file in files:

                if not (
                    file.endswith(".wav")
                    or file.endswith(".mp4")
                    or file.endswith(".m4a")
                ):
                    continue

                full_path = os.path.join(root, file)

                torque = parse_filename(file)

                try:

                    signal, sr = load_audio(full_path)

                    hits = split_hits(signal, sr)

                    features = [
                        extract_features(h, sr)
                        for h in hits
                    ]

                except:

                    st.warning(f"Could not process {file}")
                    continue

                # TRAINING FILE
                if torque is not None:

                    for feat in features:

                        X_train.append(feat)
                        y_train.append(torque)

                # EXPERIMENTAL FILE
                else:

                    experimental.append(
                        (file, features)
                    )

        # ====================================================
        # TRAIN MODEL
        # ====================================================

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        st.subheader("Training Data")

        st.write("Samples:", len(X_train))

        unique = np.unique(y_train)

        st.write("Torque Classes:", unique)

        if len(unique) < 2:

            st.error("""
            Need at least TWO torque classes.

            Example:
            - 0ftlb
            - 25ftlb
            """)

            st.stop()

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X_train)

        Xtr, Xte, ytr, yte = train_test_split(
            X_scaled,
            y_train,
            test_size=0.3,
            random_state=42
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        model.fit(Xtr, ytr)

        pred = model.predict(Xte)

        acc = accuracy_score(yte, pred)

        st.success(f"Validation Accuracy: {acc:.4f}")

        # ====================================================
        # EXPERIMENTAL PREDICTIONS
        # ====================================================

        st.header("Experimental Predictions")

        for file, features in experimental:

            if len(features) == 0:
                continue

            features = scaler.transform(features)

            preds = model.predict(features)

            final = int(np.bincount(preds).argmax())

            st.subheader(file)

            if final <= 10:

                st.error(f"⚠️ VERY LOOSE → {final} ft-lbs")

            elif final <= 30:

                st.warning(f"⚠️ MODERATELY TIGHT → {final} ft-lbs")

            else:

                st.success(f"✅ TIGHT → {final} ft-lbs")

            draw_flange(final)

            fig, ax = plt.subplots()

            ax.hist(preds)

            ax.set_title("Impact Predictions")

            st.pyplot(fig)
