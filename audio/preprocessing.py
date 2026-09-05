"""
DSP preprocessing: reduces background noise from raw microphone audio
before it goes to the STT engine.

Pipeline: raw PCM -> float waveform -> spectral-gating noise reduction
(via Short-Time Fourier Transform under the hood in noisereduce) ->
cleaned waveform back to PCM int16 bytes.
"""

import io
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf

from config.settings import SAMPLE_RATE


def bytes_to_waveform(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Convert raw int16 PCM bytes into a normalized float32 waveform."""
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    audio_np /= np.iinfo(np.int16).max  # normalize to [-1, 1]
    return audio_np


def waveform_to_bytes(waveform: np.ndarray) -> bytes:
    """Convert a float32 waveform back into int16 PCM bytes."""
    waveform = np.clip(waveform, -1.0, 1.0)
    int_audio = (waveform * np.iinfo(np.int16).max).astype(np.int16)
    return int_audio.tobytes()


def reduce_noise(audio_bytes: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """
    Applies spectral gating noise reduction to raw mic audio.

    librosa is used for STFT-based feature access (and can be extended
    for spectral analysis / silence trimming); noisereduce performs the
    actual spectral gating using an STFT internally.
    """
    waveform = bytes_to_waveform(audio_bytes, sample_rate)

    # Trim leading/trailing silence — reduces false STT triggers on dead air
    trimmed, _ = librosa.effects.trim(waveform, top_db=25)

    # Spectral gating noise reduction
    cleaned = nr.reduce_noise(y=trimmed, sr=sample_rate, stationary=False)

    return waveform_to_bytes(cleaned)


def save_debug_wav(audio_bytes: bytes, path: str, sample_rate: int = SAMPLE_RATE):
    """Optional helper — dump cleaned audio to disk to sanity-check preprocessing."""
    waveform = bytes_to_waveform(audio_bytes, sample_rate)
    sf.write(path, waveform, sample_rate)
