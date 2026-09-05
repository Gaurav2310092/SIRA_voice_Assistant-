"""
Tests for entity extraction. Run: python -m pytest tests/test_ner_extractor.py -v
"""

from nlp.ner_extractor import extract_entities


def test_search_query_extraction():
    entities = extract_entities("search", "search for python tutorials")
    assert entities["query"] == "python tutorials"


def test_song_extraction():
    entities = extract_entities("play", "play despacito")
    assert "despacito" in entities["song"]


def test_app_name_extraction():
    entities = extract_entities("open_app", "open notepad")
    assert entities["app_name"] == "notepad"
