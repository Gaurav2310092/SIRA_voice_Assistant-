"""
Basic sanity tests for the intent classifier.
Run: python -m pytest tests/test_intent_classifier.py -v

NOTE: requires the model to be trained first:
    python -m nlp.training.train_intent_model
"""

import pytest
from nlp.intent_classifier import predict_intent


@pytest.mark.parametrize("text,expected_intent", [
    ("search for python tutorials", "search"),
    ("play some music", "play"),
    ("send an email to rahul", "email"),
    ("open notepad", "open_app"),
    ("quit sira", "exit"),
])
def test_known_intents(text, expected_intent):
    intent, confidence = predict_intent(text)
    assert intent == expected_intent
    assert confidence > 0


def test_gibberish_returns_unknown():
    intent, confidence = predict_intent("asdkjfh qlwkejr")
    assert intent == "unknown"
