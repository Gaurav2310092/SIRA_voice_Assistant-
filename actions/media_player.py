"""Hands-free YouTube playback via PyWhatKit."""

import pywhatkit


def play_song(song: str):
    if not song:
        return "I didn't catch what you want to play."

    pywhatkit.playonyt(song)
    return f"Playing {song} on YouTube."
