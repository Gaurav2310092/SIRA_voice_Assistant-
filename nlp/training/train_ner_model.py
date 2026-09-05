"""
OPTIONAL: Trains a custom spaCy NER component that recognizes an "APP"
entity type (e.g. "VS Code", "Notepad", "Calculator") — something the
pretrained en_core_web_sm model doesn't know about out of the box.

This is an enhancement, not required for the MVP — the base pipeline
in ner_extractor.py works fine with just the pretrained model + regex
fallback. Do this only if you have time and want a stronger "trained a
custom NER model" point for your report.

Run from project root:
    python -m nlp.training.train_ner_model
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import random
import spacy
from spacy.training import Example

from config.settings import NER_MODEL_DIR

# Minimal labeled examples: (text, {"entities": [(start_char, end_char, label)]})
TRAIN_DATA = [
    ("open notepad", {"entities": [(5, 12, "APP")]}),
    ("launch vs code", {"entities": [(7, 14, "APP")]}),
    ("start the calculator", {"entities": [(10, 20, "APP")]}),
    ("open chrome please", {"entities": [(5, 11, "APP")]}),
    ("launch visual studio code", {"entities": [(7, 26, "APP")]}),
    ("open ms paint", {"entities": [(5, 13, "APP")]}),
    ("start notepad for me", {"entities": [(6, 13, "APP")]}),
    ("can you open calculator", {"entities": [(13, 23, "APP")]}),
]


def main():
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    ner.add_label("APP")

    optimizer = nlp.begin_training()

    for epoch in range(30):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, losses: {losses}")

    NER_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(NER_MODEL_DIR)
    print(f"Saved custom NER model to {NER_MODEL_DIR}")


if __name__ == "__main__":
    main()
