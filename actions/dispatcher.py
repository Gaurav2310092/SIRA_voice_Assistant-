"""
Routes a classified (intent, entities) pair to the correct action
function. This is the single place that connects the NLP layer to
the automation layer.
"""

from actions.web_search import search_web
from actions.media_player import play_song
from actions.email_sender import send_email
from actions.app_launcher import open_app


def dispatch(intent: str, entities: dict) -> str:
    if intent == "search":
        return search_web(entities.get("query", ""))

    if intent == "play":
        return play_song(entities.get("song", ""))

    if intent == "email":
        return send_email(entities.get("contact_name", ""))

    if intent == "open_app":
        return open_app(entities.get("app_name", ""))

    if intent == "exit":
        return "__EXIT__"  # sentinel value main.py checks for to break the loop

    return "Sorry, I didn't understand that command."
