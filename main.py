"""
SIRA — Speech-Interfaced Reactive Assistant
Main voice loop: wake word -> STT -> NLP (intent + entities) -> action -> TTS feedback.

WAKE_WORD_MODE controls activation:
- "keypress": press Enter to activate (no Porcupine access key needed) — use this
  while you don't have a PORCUPINE_ACCESS_KEY set up yet.
- "voice": uses Porcupine offline wake word detection (requires the access key).
Switch back to "voice" once your Picovoice account is sorted out.
"""

WAKE_WORD_MODE = "keypress"  # "keypress" or "voice"

from audio.stt import SpeechToText
from nlp.intent_classifier import predict_intent
from nlp.semantic_fallback import semantic_match
from nlp.ner_extractor import extract_entities
from actions.dispatcher import dispatch
from feedback.tts import TextToSpeech
from utils.logger import logger


def process_command(text: str, tts: TextToSpeech) -> bool:
    """
    Runs one command through the full NLP -> action pipeline.
    Returns False if SIRA should shut down, True to keep looping.
    """
    if not text:
        return True

    intent, confidence = predict_intent(text)
    logger.info(f"Intent: {intent} (confidence={confidence:.2f}) for text: '{text}'")

    # Low-confidence classifier result -> try semantic fallback before giving up
    if intent == "unknown":
        fallback_intent, sim_score = semantic_match(text)
        logger.info(f"Semantic fallback: {fallback_intent} (similarity={sim_score:.2f})")
        if fallback_intent != "unknown":
            intent = fallback_intent
        else:
            tts.speak("I'm not sure what you meant. Could you rephrase that?")
            return True

    entities = extract_entities(intent, text)
    result = dispatch(intent, entities)

    if result == "__EXIT__":
        tts.speak("Goodbye!")
        return False

    tts.speak(result)
    return True


def main():
    tts = TextToSpeech()
    stt = SpeechToText()

    wake_detector = None
    if WAKE_WORD_MODE == "voice":
        from audio.wake_word import WakeWordDetector
        wake_detector = WakeWordDetector()

    tts.speak("SIRA is online.")

    running = True
    try:
        while running:
            if WAKE_WORD_MODE == "keypress":
                input("\nPress Enter to activate SIRA (Ctrl+C to quit)...")
            else:
                wake_detector.listen_for_wake_word()

            tts.speak("Yes?")

            text = stt.listen_and_transcribe()
            running = process_command(text, tts)
    except KeyboardInterrupt:
        logger.info("Shutting down (keyboard interrupt).")
    finally:
        if wake_detector is not None:
            wake_detector.close()


if __name__ == "__main__":
    main()
