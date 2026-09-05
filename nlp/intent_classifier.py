"""
Loads the trained TF-IDF + Logistic Regression model and exposes a
single predict_intent() function used by the dispatcher.
"""

import joblib

from nlp.preprocess_text import preprocess
from config.settings import INTENT_MODEL_PATH, VECTORIZER_PATH, INTENT_CONFIDENCE_THRESHOLD


class IntentClassifier:
    def __init__(self):
        if not INTENT_MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            raise FileNotFoundError(
                "Trained model not found. Run: python -m nlp.training.train_intent_model"
            )
        self.model = joblib.load(INTENT_MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)

    def predict(self, text: str, threshold: float = INTENT_CONFIDENCE_THRESHOLD):
        """
        Returns (intent, confidence). If confidence is below threshold,
        returns ("unknown", confidence) so the dispatcher can fall back
        to the semantic matcher or ask for clarification.
        """
        clean_text = preprocess(text)
        if not clean_text:
            return "unknown", 0.0

        vec = self.vectorizer.transform([clean_text])
        probs = self.model.predict_proba(vec)[0]
        max_idx = probs.argmax()
        intent = self.model.classes_[max_idx]
        confidence = float(probs[max_idx])

        if confidence < threshold:
            return "unknown", confidence

        return intent, confidence


# Singleton instance — model is loaded once, reused across the app
_classifier_instance = None


def predict_intent(text: str):
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = IntentClassifier()
    return _classifier_instance.predict(text)
