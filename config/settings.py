"""
Central configuration for SIRA.
All tunable constants and paths live here so nothing is hardcoded
in the middle of business logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
BASE_DIR = Path(__file__).resolve().parent.parent
NLP_MODELS_DIR = BASE_DIR / "nlp" / "models"
INTENT_MODEL_PATH = NLP_MODELS_DIR / "intent_classifier.pkl"
VECTORIZER_PATH = NLP_MODELS_DIR / "tfidf_vectorizer.pkl"
NER_MODEL_DIR = NLP_MODELS_DIR / "ner_model"
DATASET_PATH = BASE_DIR / "nlp" / "training" / "dataset.csv"

# ---- Audio ----
SAMPLE_RATE = 16000
CHANNELS = 1

# ---- Wake word ----
PORCUPINE_ACCESS_KEY = os.getenv("PORCUPINE_ACCESS_KEY", "")
WAKE_WORD = os.getenv("WAKE_WORD", "jarvis")

# ---- NLP ----
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", 0.35))
SEMANTIC_FALLBACK_THRESHOLD = 0.55  # cosine similarity cutoff for SBERT fallback

# ---- Email ----
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS", "")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD", "")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# ---- Known local apps (extend as needed) ----
APP_PATHS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "vs code": "code",
    "vscode": "code",
    "chrome": "chrome",
}
