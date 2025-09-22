import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("C:/Users/avins/OneDrive/Desktop/moodmate/data/music/Music.csv")
# -----------------------------
# Create mood labels from valence + energy
# -----------------------------
def assign_mood(valence, energy):
    if valence > 0.5 and energy > 0.5:
        return "Happy"
    elif valence <= 0.5 and energy > 0.5:
        return "Energetic"
    elif valence <= 0.5 and energy <= 0.5:
        return "Sad"
    else:
        return "Calm"

df["mood"] = df.apply(lambda x: assign_mood(x["valence"], x["energy"]), axis=1)

# -----------------------------
# Features and target
# -----------------------------
X = df[["danceability","energy","loudness","speechiness",
        "acousticness","instrumentalness","liveness","valence"]]
y = df["mood"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model as .joblib
joblib.dump(model, "music_mood_model.joblib")
print("✅ Model trained and saved as music_mood_model.joblib")
