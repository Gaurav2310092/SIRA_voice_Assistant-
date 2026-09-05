"""
Text normalization/sanitization used identically at training time
and inference time — keeping this single-sourced avoids train/serve skew.
"""

import re


def preprocess(text: str) -> str:
    """
    Lowercases and strips anything that isn't alphanumeric or whitespace.
    This also doubles as an injection-safety measure: stripping control
    characters and shell/HTML metacharacters before the text ever reaches
    Selenium, subprocess, or smtplib downstream.
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
