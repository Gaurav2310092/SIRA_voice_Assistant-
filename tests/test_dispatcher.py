"""
Tests for the action dispatcher routing logic (mocks the actual
Selenium/email/subprocess calls so tests don't require a browser,
mic, or real credentials).
Run: python -m pytest tests/test_dispatcher.py -v
"""

from unittest.mock import patch
from actions.dispatcher import dispatch


def test_exit_returns_sentinel():
    result = dispatch("exit", {})
    assert result == "__EXIT__"


def test_unknown_intent_gives_fallback_message():
    result = dispatch("some_made_up_intent", {})
    assert "didn't understand" in result.lower()


@patch("actions.dispatcher.search_web")
def test_search_dispatch_calls_search_web(mock_search):
    mock_search.return_value = "Here are the search results."
    result = dispatch("search", {"query": "python"})
    mock_search.assert_called_once_with("python")
    assert result == "Here are the search results."


@patch("actions.dispatcher.open_app")
def test_open_app_dispatch(mock_open):
    mock_open.return_value = "Opening notepad."
    result = dispatch("open_app", {"app_name": "notepad"})
    mock_open.assert_called_once_with("notepad")
