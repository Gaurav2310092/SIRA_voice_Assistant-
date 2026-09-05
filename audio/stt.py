"""
Speech-to-text wrapper around SpeechRecognition.
Captures mic audio, runs it through DSP preprocessing, then transcribes.
"""

import io
import speech_recognition as sr

from audio.preprocessing import reduce_noise
from config.settings import SAMPLE_RATE


class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone(sample_rate=SAMPLE_RATE)

    def listen_and_transcribe(self, timeout: int = 5, phrase_time_limit: int = 8) -> str:
        """
        Records a single utterance from the mic, denoises it, and
        returns the transcribed text (empty string on failure).
        """
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("[STT] Listening...")
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
            except sr.WaitTimeoutError:
                print("[STT] No speech detected within timeout.")
                return ""

        raw_pcm = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=2)
        cleaned_pcm = reduce_noise(raw_pcm, sample_rate=SAMPLE_RATE)

        cleaned_audio = sr.AudioData(cleaned_pcm, SAMPLE_RATE, 2)

        try:
            text = self.recognizer.recognize_google(cleaned_audio)
            print(f"[STT] Transcribed: {text}")
            return text
        except sr.UnknownValueError:
            print("[STT] Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"[STT] API request failed: {e}")
            return ""
