# file: 7_summarize_extractive.py
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def sent_split(text: str):
    # lightweight splitter; for production use nltk or spacy
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sents if s]

def summarize(text: str, max_sentences=3):
    sents = sent_split(text)
    words = re.findall(r"[a-zA-Z']+", text.lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    freqs = Counter(words)
    # normalize frequencies
    if not freqs: return " ".join(sents[:max_sentences])
    maxf = max(freqs.values())
    for w in freqs: freqs[w] /= maxf

    # sentence scores = sum of word scores
    scored = []
    for i, s in enumerate(sents):
        ws = re.findall(r"[a-zA-Z']+", s.lower())
        score = sum(freqs.get(w, 0.0) for w in ws) / (len(ws) + 1e-9)
        scored.append((score, i, s))

    # keep top sentences in original order
    top = sorted(sorted(scored, key=lambda x: -x[0])[:max_sentences], key=lambda x: x[1])
    return " ".join(s for _, _, s in top)

if __name__ == "__main__":
    text = (
        "Transformers have revolutionized natural language processing. "
        "By leveraging self-attention, they capture long-range dependencies effectively. "
        "Pretraining on large corpora enables strong performance on many tasks. "
        "However, transformers can be computationally expensive. "
        "Researchers explore efficient architectures and distillation to reduce cost."
    )
    print(summarize(text, max_sentences=2))