# file: 5_spacy_ner_pos.py
import spacy
from pprint import pprint

nlp = spacy.load("en_core_web_sm")

text = ("Apple is opening a new office in Bengaluru next quarter. "
        "Tim Cook met Karnataka officials on September 3, 2025 to discuss expansion.")

doc = nlp(text)

print("\nNamed Entities (text, label):")
for ent in doc.ents:
    print(f"{ent.text:<25}  -> {ent.label_}")

print("\nPart-of-Speech & Lemmas:")
for token in doc:
    if not token.is_space:
        print(f"{token.text:<15} POS={token.pos_:<5}  Lemma={token.lemma_}")

print("\nNoun chunks (base NPs):")
pprint([chunk.text for chunk in doc.noun_chunks])