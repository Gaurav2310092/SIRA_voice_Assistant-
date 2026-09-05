"""
Trains the TF-IDF + Logistic Regression intent classifier and saves
the model + vectorizer to disk. Also produces evaluation artifacts
(classification report, confusion matrix) for the project report.

Run from project root:
    python -m nlp.training.train_intent_model
"""

import sys
from pathlib import Path

# allow running this file directly
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from nlp.preprocess_text import preprocess
from config.settings import DATASET_PATH, INTENT_MODEL_PATH, VECTORIZER_PATH, NLP_MODELS_DIR


def main():
    NLP_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    df["clean_text"] = df["text"].apply(preprocess)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["intent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000,C=10)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    report = classification_report(y_test, y_pred)
    print(report)

    # Save the report to a text file for the project write-up
    with open(NLP_MODELS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=model.classes_, yticklabels=model.classes_
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Intent Classifier Confusion Matrix")
    plt.tight_layout()
    plt.savefig(NLP_MODELS_DIR / "confusion_matrix.png")
    print(f"\nSaved confusion matrix to {NLP_MODELS_DIR / 'confusion_matrix.png'}")

    joblib.dump(model, INTENT_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model to {INTENT_MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()
