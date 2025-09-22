import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.text_cleaner import TextCleaner  # Make sure this exists

# Load your song data
songs_df = pd.read_parquet("models/songs_clean.parquet")  # or CSV if that's what you have

# Use the column with meaningful text for search
text_data = songs_df["search_query"]  # or another column if appropriate

# Define and fit pipeline
pipe = Pipeline([
    ("cleaner", TextCleaner()),
    ("vectorizer", TfidfVectorizer())
])

pipe.fit(text_data)

# Save it
os.makedirs("models", exist_ok=True)
joblib.dump(pipe, "models/tfidf_vectorizer.joblib")
print("✅ Vectorizer saved to models/tfidf_vectorizer.joblib")
