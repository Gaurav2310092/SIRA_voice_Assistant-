"""
Offline wake word detection.

Uses Porcupine (pvporcupine) instead of Snowboy — Snowboy is
unmaintained/deprecated, Porcupine is its modern, actively supported
replacement with the same low-latency, low-CPU offline design.
"""

import struct
import pyaudio
import pvporcupine

from config.settings import PORCUPINE_ACCESS_KEY, WAKE_WORD


class WakeWordDetector:
    def __init__(self, access_key: str = PORCUPINE_ACCESS_KEY, keyword: str = WAKE_WORD):
        if not access_key:
            raise ValueError("PORCUPINE_ACCESS_KEY not set in .env")

        # Porcupine ships several built-in keywords (e.g. 'jarvis', 'computer').
        # For a fully custom wake word you'd train one at console.picovoice.ai
        # and pass its .ppn file path via keyword_paths= instead.
        self.porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=[keyword],
        )
        self.pa = pyaudio.PyAudio()
        self.audio_stream = None

    def start_stream(self):
        self.audio_stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )

    def listen_for_wake_word(self) -> bool:
        """
        Blocks until the wake word is detected. Returns True once triggered.
        Runs in a tight low-CPU loop reading small audio frames.
        """
        if self.audio_stream is None:
            self.start_stream()

        print(f"[WakeWord] Listening for '{WAKE_WORD}'...")
        while True:
            pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)

            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                print("[WakeWord] Wake word detected!")
                return True

    def close(self):
        if self.audio_stream is not None:
            self.audio_stream.close()
        self.porcupine.delete()
        self.pa.terminate()
