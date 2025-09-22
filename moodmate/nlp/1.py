# file: 1_preprocess.py
import re
from typing import List
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy

nlp = spacy.load("en_core_web_sm")  # small, fast English model

def basic_clean(text: str) -> str:
    # lower, strip urls/emails/@mentions/hashtags, keep letters/numbers/space/apostrophe
    text = text.lower()
    text = re.sub(r"(http\S+|www\.\S+)", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_stop_lemma(text: str) -> List[str]:
    doc = nlp(text)
    out = []
    for tok in doc:
        if tok.is_space or tok.is_punct:
            continue
        lemma = tok.lemma_.lower().strip()
        if len(lemma) < 3:           # drop very short tokens
            continue
        if lemma in ENGLISH_STOP_WORDS:  # sklearn's built-in stoplist
            continue
        out.append(lemma)
    return out

def preprocess(text: str) -> List[str]:
    return tokenize_stop_lemma(basic_clean(text))

if __name__ == "__main__":
    s = "Emails like help@site.com are filtered. I’m LOVING NLP!!! Visit https://x.y."
    print(preprocess(s))