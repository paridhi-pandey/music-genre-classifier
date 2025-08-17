# train_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Paths
FEATURES_PATH = "data/features.csv"
MODEL_PATH = "models/genre_classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

print(" Loading features...")
df = pd.read_csv(FEATURES_PATH)

# ✅ Ensure numeric features are floats, not strings
for col in df.columns:
    if col not in ["label", "transcript"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop transcript (text)
if "transcript" in df.columns:
    X = df.drop(["label", "transcript"], axis=1)
else:
    X = df.drop("label", axis=1)

y = df["label"]

# Encode labels to integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(" Training RandomForest model...")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluation
print("\n Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Save model and encoder
joblib.dump(clf, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

print(f"\n Model saved to {MODEL_PATH}")
print(f" Label encoder saved to {ENCODER_PATH}")