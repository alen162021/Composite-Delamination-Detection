# Streamlit App for Composite Delamination Detection
import matplotlib.pyplot as plt

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Composite Delamination Detection", layout="wide")

st.title("🧪 Composite Delamination Detection System")

st.write("Upload vibration/acoustic signals to detect possible delamination in composite structures.")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_detection_model():
    model_path = "model.h5"  # adjust based on your repo
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

model = load_detection_model()

if model is None:
    st.warning("Model not found. Please ensure model.h5 exists in the repo root.")

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------

def extract_features(file):
    y, sr = librosa.load(file, sr=16000)

    # simple MFCC features (adjust if repo uses different pipeline)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    return mfcc.reshape(1, -1)

# -------------------------------
# UPLOAD FILE
# -------------------------------
uploaded_file = st.file_uploader("Upload vibration/audio file", type=["wav", "mp3"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    features = extract_features(uploaded_file)

    st.subheader("Extracted Features")
    st.write(features)

    if model is not None:
        prediction = model.predict(features)

        st.subheader("Prediction")

        if prediction[0][0] > 0.5:
            st.error("⚠️ Delamination Detected")
        else:
            st.success("✅ No Delamination Detected")

        st.write("Raw output:", prediction)

# -------------------------------
# VISUALIZATION
# -------------------------------
if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=16000)

    st.subheader("Waveform")
    fig, ax = plt.subplots()
    ax.plot(y)
    st.pyplot(fig)

    st.subheader("Spectrogram")
    X = librosa.stft(y)
    Xdb = librosa.amplitude_to_db(abs(X))

    fig2, ax2 = plt.subplots()
    img = librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    st.pyplot(fig2)
