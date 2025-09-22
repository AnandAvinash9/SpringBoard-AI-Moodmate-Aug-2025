# file: 6_semantic_search.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

corpus = [
    "deep learning methods for image classification",
    "convolutional neural networks for vision",
    "natural language processing with transformers",
    "classical machine learning with SVM and logistic regression",
    "transfer learning for NLP tasks"
]

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
X = vectorizer.fit_transform(corpus)

def search(query: str, topk=3):
    q = vectorizer.transform([query])
    sims = cosine_similarity(q, X).ravel()
    top_idx = sims.argsort()[::-1][:topk]
    results = [(corpus[i], float(sims[i])) for i in top_idx]
    return results

if __name__ == "__main__":
    for item, score in search("best models for text classification", topk=3):
        print(f"{score:.3f}  {item}")