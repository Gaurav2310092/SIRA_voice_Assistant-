"""
Semantic similarity fallback used when the TF-IDF classifier's confidence
is below threshold. Embeds the user's command with Sentence-BERT and
compares it against embeddings of known example phrasings per intent —
catches paraphrases the classical model wasn't trained on
(e.g. "fire up notepad" instead of "open notepad").
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util

from config.settings import DATASET_PATH, SEMANTIC_FALLBACK_THRESHOLD

_model = None
_intent_examples = None
_example_embeddings = None


def _load():
    global _model, _intent_examples, _example_embeddings
    if _model is not None:
        return

    _model = SentenceTransformer("all-MiniLM-L6-v2")

    df = pd.read_csv(DATASET_PATH)
    df = df[df["intent"] != "unknown"]  # no useful "unknown" exemplars to match against

    _intent_examples = df["text"].tolist()
    intent_labels = df["intent"].tolist()
    _intent_examples = list(zip(_intent_examples, intent_labels))

    texts = [t for t, _ in _intent_examples]
    _example_embeddings = _model.encode(texts, convert_to_tensor=True)


def semantic_match(text: str, threshold: float = SEMANTIC_FALLBACK_THRESHOLD):
    """
    Returns (intent, similarity_score) for the closest known example,
    or ("unknown", score) if nothing clears the threshold.
    """
    _load()

    query_embedding = _model.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, _example_embeddings)[0]

    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])
    best_intent = _intent_examples[best_idx][1]

    if best_score < threshold:
        return "unknown", best_score

    return best_intent, best_score
