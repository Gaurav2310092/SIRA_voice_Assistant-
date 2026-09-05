"""
Extracts intent parameters (search query, contact name, app name) from
the transcribed command text using spaCy NER, with a regex fallback for
patterns spaCy's pretrained model won't catch (e.g. app names).
"""

import re
import spacy

from config.settings import NER_MODEL_DIR, APP_PATHS

_nlp = spacy.load("en_core_web_sm")

# Load the optional custom-trained APP entity model if it exists
_custom_ner = None
if NER_MODEL_DIR.exists() and any(NER_MODEL_DIR.iterdir()):
    _custom_ner = spacy.load(NER_MODEL_DIR)

# Trigger words to strip off before treating the remainder as a search query
_SEARCH_TRIGGERS = r"^(search for|search|look up|google|find|find me|find information about)\s+"
_PLAY_TRIGGERS = r"^(play|put on|start playing)\s+"
_EMAIL_TRIGGERS = r"^(send an email to|mail to|email to|send email to|mail|email)\s+"
_OPEN_TRIGGERS = r"^(open|launch|start)\s+"


def extract_person(text: str) -> str:
    """Pulls a PERSON entity out of the text using spaCy's pretrained NER."""
    doc = _nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return ""


def extract_app_name(text: str) -> str:
    """
    Tries the custom-trained APP NER model first (if available),
    falls back to matching against the known APP_PATHS dictionary.
    """
    if _custom_ner is not None:
        doc = _custom_ner(text)
        for ent in doc.ents:
            if ent.label_ == "APP":
                return ent.text.lower()

    text_lower = text.lower()
    for app_name in APP_PATHS:
        if app_name in text_lower:
            return app_name

    # last resort: strip the open/launch/start trigger and use the remainder
    remainder = re.sub(_OPEN_TRIGGERS, "", text_lower).strip()
    return remainder


def extract_search_query(text: str) -> str:
    clean = re.sub(_SEARCH_TRIGGERS, "", text.lower()).strip()
    return clean


def extract_song_query(text: str) -> str:
    clean = re.sub(_PLAY_TRIGGERS, "", text.lower()).strip()
    return clean


def extract_entities(intent: str, text: str) -> dict:
    """
    Main entry point — given the classified intent and raw text,
    returns a dict of extracted parameters relevant to that intent.
    """
    if intent == "search":
        return {"query": extract_search_query(text)}

    if intent == "play":
        return {"song": extract_song_query(text)}

    if intent == "email":
        person = extract_person(text)
        clean = re.sub(_EMAIL_TRIGGERS, "", text.lower()).strip()
        return {"contact_name": person or clean}

    if intent == "open_app":
        return {"app_name": extract_app_name(text)}

    return {}
