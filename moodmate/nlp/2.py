# file: 2_classify_tfidf.py
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# tiny demo dataset (positive/negative sentiment)
texts = [
    "I loved this movie, fantastic acting and great story",
    "This film was terrible and boring",
    "Absolutely wonderful experience, highly recommend",
    "Worst acting ever, do not watch",
    "It was okay, some parts were fun",
    "I hated the plot, very disappointing",
    "Brilliant direction and superb cast",
    "Not good, waste of time",
    "Enjoyable and engaging from start to finish",
    "Awful soundtrack and weak story"
]
labels = np.array([1,0,1,0,1,0,1,0,1,0])  # 1=pos, 0=neg

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=1)),
    ("clf",   LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("Classification report:\n", classification_report(y_test, y_pred, digits=4))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# try the model
samples = ["pretty good but slow in places", "utterly awful, I want my time back"]
print("Predictions:", pipe.predict(samples))
print("Class probabilities:", pipe.predict_proba(samples))