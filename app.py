# app.py
import streamlit as st
import numpy as np
import librosa
import joblib
import speech_recognition as sr

MODEL_PATH = "models/genre_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

clf = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

st.title("🎵 Music Genre Classifier")
st.write("Upload a `.wav` file and let the model predict its genre!")

def extract_audio_features(file_path, n_mfcc=13):
    try:
        y, sr_rate = librosa.load(file_path, duration=30)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0).flatten()

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_rate)
        chroma_mean = np.mean(chroma.T, axis=0).flatten()

        # Tempo (scalar)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr_rate)
        tempo = float(tempo)  # ensure scalar

        # Concatenate all features as a 1D list of floats
        feature_vector = list(mfcc_mean) + list(chroma_mean) + [tempo]
        return [float(f) for f in feature_vector]

    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None


def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            return recognizer.recognize_google(audio_data)
    except Exception:
        return ""

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Save uploaded file to a temporary path
    tmp_path = "temp.wav"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(tmp_path, format="audio/wav")
    st.write("🔍 Extracting features...")

    features = extract_audio_features(tmp_path)
    transcript = transcribe_audio(tmp_path)

    # ---- Defensive checks to avoid the crash on reshape ----
    if features is None:
        st.stop()  # we already showed an error in extract_audio_features

    try:
        X = np.asarray(features, dtype=float).reshape(1, -1)
    except Exception as e:
        st.error(f"Failed to prepare features for prediction: {e}")
        st.stop()

    expected = getattr(clf, "n_features_in_", None)
    if expected is not None and X.shape[1] != expected:
        st.error(f"Feature length mismatch: model expects {expected}, got {X.shape[1]}.")
        st.stop()
    # --------------------------------------------------------

    pred = clf.predict(X)
    genre = encoder.inverse_transform(pred)[0]
    st.success(f"🎶 Predicted Genre: **{genre}**")

    if transcript:
        st.write("📝 Transcription (speech-to-text):")
        st.info(transcript)