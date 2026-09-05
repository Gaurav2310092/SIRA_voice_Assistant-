# SIRA — Speech-Interfaced Reactive Assistant

A lightweight, voice-activated desktop assistant that eliminates repetitive
manual desktop workflows through a hands-free voice loop, using an NLP-based
intent classification pipeline instead of rigid keyword matching.

## Architecture

```
Wake Word (Porcupine) → Audio Capture → DSP Preprocessing (Librosa + Noisereduce)
    → Speech-to-Text (SpeechRecognition) → Text Normalization
    → Intent Classification (TF-IDF + Logistic Regression)
        → confident: NER extraction (spaCy) → Action Dispatcher
        → low confidence: Semantic Fallback (SBERT) → Action Dispatcher
    → Action Handlers (Selenium / PyWhatKit / smtplib / subprocess)
    → Audible Feedback (pyttsx3)
    → loop back to Wake Word
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # then fill in your real credentials
```

## Train the intent classifier

```bash
python -m nlp.training.train_intent_model
```
This produces `nlp/models/intent_classifier.pkl`, `tfidf_vectorizer.pkl`,
and a confusion matrix + classification report in `nlp/models/`.

(Optional) Train the custom APP-entity NER model:
```bash
python -m nlp.training.train_ner_model
```

## Run

```bash
python main.py
```
Say the wake word (default: "jarvis"), then speak a command, e.g.:
- "search for python tutorials"
- "play despacito"
- "send an email to rahul"
- "open notepad"
- "quit sira"

## Run tests

```bash
python -m pytest tests/ -v
```

## Project structure

See `main.py` for the entry point and `actions/dispatcher.py` for how
intents route to automation handlers. The NLP pipeline lives under `nlp/`.

## Notes

- Contacts for the email action are stored in `actions/email_sender.py`'s
  `CONTACTS` dict — extend this or swap for a real contacts file/DB.
- App paths for the launcher live in `config/settings.py`'s `APP_PATHS`.
- Confidence threshold for intent fallback is tunable via `.env`
  (`INTENT_CONFIDENCE_THRESHOLD`).
