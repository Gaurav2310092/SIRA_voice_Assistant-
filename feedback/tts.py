"""Offline audible feedback using pyttsx3."""

import pyttsx3


class TextToSpeech:
    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", rate)
        self.engine.setProperty("volume", volume)

    def speak(self, text: str):
        if not text:
            return
        print(f"[SIRA] {text}")
        self.engine.say(text)
        self.engine.runAndWait()
