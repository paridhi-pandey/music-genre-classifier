# train_model.py
import os
import zipfile
import numpy as np
import pandas as pd
import librosa
import speech_recognition as sr
from tqdm import tqdm

# Paths
DATASET_ZIP = r"C:\Users\hp2\Downloads\Data.zip"
EXTRACTED_PATH = r"C:\Users\hp2\Downloads\Data"
OUTPUT_CSV = "data/features.csv"

def extract_dataset():
    """Extracts dataset zip if not already extracted."""
    if not os.path.exists(EXTRACTED_PATH):
        print("Extracting dataset...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(EXTRACTED_PATH)
        print("Extraction complete!")
    else:
        print("Dataset already extracted.")

def extract_audio_features(file_path, n_mfcc=13):
    """Extract MFCC, chroma, and tempo features."""
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return list(mfcc_mean) + list(chroma_mean) + [tempo]
    except Exception as e:
        print(f"Skipping {file_path}: {e}")
        return None

def transcribe_audio(file_path):
    """Transcribe audio to text using SpeechRecognition."""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            return text
    except Exception:
        return ""  # return empty if transcription fails

def extract_features():
    """Extract all features from dataset."""
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    features_list = []
    labels = []
    transcripts = []

    GENRES_PATH = os.path.join(EXTRACTED_PATH, "genres_original")

    for genre in genres:
        genre_path = os.path.join(GENRES_PATH, genre)
        if not os.path.exists(genre_path):
            print(f"Warning: Missing genre folder -> {genre_path}")
            continue

        for filename in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            if filename.endswith(".wav"):
                file_path = os.path.join(genre_path, filename)

                # Extract numeric audio features
                audio_features = extract_audio_features(file_path)
                if audio_features is None:
                    continue  # Skip problematic files

                # Get transcription
                transcript = transcribe_audio(file_path)

                features_list.append(audio_features)
                labels.append(genre)
                transcripts.append(transcript)

    # ✅ Build dataframe with proper numeric columns
    features_df = pd.DataFrame(features_list)
    features_df.columns = [f"f{i}" for i in range(features_df.shape[1])]  # f0, f1, f2 ...
    features_df['label'] = labels
    features_df['transcript'] = transcripts

    return features_df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    extract_dataset()
    df = extract_features()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Feature extraction complete! Saved to {OUTPUT_CSV}")