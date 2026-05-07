import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flange Loosening Detection",
    page_icon="🔩",
    layout="wide"
)

st.title("🔩 Smart Flange Loosening Detection")

st.caption("""
Machine Learning + Acoustic Percussion Analysis
University of Houston
""")

# ============================================================
# AUDIO LOADING
# ============================================================

def load_audio(file):

    suffix = "." + file.name.split(".")[-1]

    file.seek(0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        tmp_path = tmp.name

    try:
        signal, sr = librosa.load(tmp_path, sr=22050)

    except Exception:

        signal, sr = sf.read(tmp_path)

        if len(signal.shape) > 1:
            signal = np.mean(signal, axis=1)

    signal = signal / (np.max(np.abs(signal)) + 1e-9)

    return signal, sr

# ============================================================
# SPLIT IMPACT HITS
# ============================================================

def split_hits(signal, sr):

    frame_length = int(0.02 * sr)
    hop_length = int(0.01 * sr)

    energy = librosa.feature.rms(
        y=signal,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    threshold = np.mean(energy) * 1.5

    indices = np.where(energy > threshold)[0]

    if len(indices) == 0:
        return [], []

    segments = np.split(
        indices,
        np.where(np.diff(indices) > 2)[0] + 1
    )

    hits = []
    boundaries = []

    for seg in segments:

        start = seg[0] * hop_length
        end = seg[-1] * hop_length

        hit = signal[start:end]

        if len(hit) > 200:

            hits.append(hit)
            boundaries.append((start, end))

    return hits, boundaries

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

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
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

    rms = np.mean(signal**2)

    return np.hstack([
        mfcc,
        zcr,
        centroid,
        rolloff,
        rms
    ])

# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(files):

    X = []
    y = []

    for file in files:

        try:
            signal, sr = load_audio(file)

        except:
            st.warning(f"Could not load {file.name}")
            continue

        hits, _ = split_hits(signal, sr)

        # GOOD if filename contains TIGHT
        # BAD if filename contains LOOSE

        label = 1 if "loose" in file.name.lower() else 0

        for h in hits:

            X.append(extract_features(h, sr))
            y.append(label)

    return np.array(X), np.array(y)

# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_cm(cm, title):

    fig, ax = plt.subplots()

    ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j],
                    ha="center",
                    va="center")

    ax.set_title(title)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    st.pyplot(fig)

# ============================================================
# FLANGE VISUALIZATION
# ============================================================

def draw_flange(loose_probability):

    fig, ax = plt.subplots(figsize=(5,5))

    circle = plt.Circle((0,0), 1, fill=False, linewidth=8)

    ax.add_artist(circle)

    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)

    for a in angles:

        x = np.cos(a)
        y = np.sin(a)

        if loose_probability > 0.5:
            color = "red"
            size = 450
        else:
            color = "green"
            size = 300

        ax.scatter(x, y, s=size, c=color)

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    ax.set_aspect("equal")

    ax.axis("off")

    st.pyplot(fig)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📘 About")

    st.write("""
    This app detects flange loosening using:

    - Percussion acoustics
    - Signal processing
    - Machine learning
    - MFCC acoustic fingerprints
    """)

    st.write("""
    Upload percussion recordings of:
    - Tight flanges
    - Loose flanges

    The app automatically:
    1. Splits impacts
    2. Extracts features
    3. Trains ML models
    4. Predicts flange health
    """)

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs([
    "📂 Upload",
    "🤖 Training",
    "🧪 Testing"
])

# ============================================================
# TAB 1
# ============================================================

with tab1:

    train_files = st.file_uploader(
        "Upload Training Audio Files",
        accept_multiple_files=True
    )

    if train_files:

        X, y = build_dataset(train_files)

        st.success(f"Extracted {len(X)} impact samples")

        st.session_state["X"] = X
        st.session_state["y"] = y

        fig, ax = plt.subplots()

        unique, counts = np.unique(y, return_counts=True)

        labels = ["Tight", "Loose"]

        ax.bar(labels, counts)

        st.pyplot(fig)

# ============================================================
# TAB 2
# ============================================================

with tab2:

    if "X" not in st.session_state:

        st.warning("Upload data first.")

    else:

        X = st.session_state["X"]
        y = st.session_state["y"]

        scaler = StandardScaler()

        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )

        models = {

            "KNN": KNeighborsClassifier(),

            "Decision Tree": DecisionTreeClassifier(),

            "Logistic Regression": LogisticRegression(max_iter=1000),

            "SVM": SVC(probability=True),

            "Random Forest": RandomForestClassifier()
        }

        best_acc = 0
        best_model = None

        for name, model in models.items():

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)

            st.subheader(name)

            st.write("Accuracy:", round(acc, 4))

            cm = confusion_matrix(y_test, pred)

            plot_cm(cm, name)

            if acc > best_acc:
                best_acc = acc
                best_model = model

        st.success(f"Best Model Accuracy: {best_acc:.4f}")

        st.session_state["best_model"] = best_model
        st.session_state["scaler"] = scaler

# ============================================================
# TAB 3
# ============================================================

with tab3:

    if "best_model" not in st.session_state:

        st.warning("Train model first.")

    else:

        test_files = st.file_uploader(
            "Upload Test Audio",
            accept_multiple_files=True,
            key="test"
        )

        if test_files:

            model = st.session_state["best_model"]
            scaler = st.session_state["scaler"]

            for file in test_files:

                st.subheader(file.name)

                signal, sr = load_audio(file)

                hits, boundaries = split_hits(signal, sr)

                if len(hits) == 0:

                    st.warning("No impacts detected")
                    continue

                features = np.array([
                    extract_features(h, sr)
                    for h in hits
                ])

                features = scaler.transform(features)

                preds = model.predict(features)

                probs = model.predict_proba(features)[:,1]

                confidence = np.mean(probs)

                if confidence > 0.5:

                    st.error(
                        f"⚠️ LOOSE FLANGE DETECTED ({confidence*100:.1f}%)"
                    )

                else:

                    st.success(
                        f"✅ TIGHT FLANGE ({(1-confidence)*100:.1f}%)"
                    )

                draw_flange(confidence)

                fig, ax = plt.subplots(figsize=(10,2))

                ax.plot(signal)

                for s,e in boundaries:
                    ax.axvspan(s,e,alpha=0.3)

                st.pyplot(fig)
