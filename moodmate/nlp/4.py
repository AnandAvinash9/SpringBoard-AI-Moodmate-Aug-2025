# file: 4_topic_lda.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

docs = [
    "cats purr and sleep on the sofa",
    "dogs bark and love to play fetch",
    "kittens and puppies are adorable",
    "stocks rallied as the market rose",
    "investors expect inflation to ease",
    "central bank raised interest rates"
]

# bag-of-words for LDA
cv = CountVectorizer(stop_words="english")
X = cv.fit_transform(docs)

lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

words = cv.get_feature_names_out()

def show_topics(model, feature_names, topn=6):
    for i, comp in enumerate(model.components_):
        terms = comp.argsort()[::-1][:topn]
        print(f"Topic {i}:"," ".join(feature_names[t] for t in terms))

show_topics(lda, words)

# infer topics for a new doc
import numpy as np
new = cv.transform(["the puppy sleeps while the dog plays"])
print("Topic distribution:", np.round(lda.transform(new), 3))