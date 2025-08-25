
# ===============================
# predict.py
# ===============================

import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

# -------------------------------
# Load model & vectorizer
# -------------------------------
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Preprocessing
# -------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# -------------------------------
# User input for prediction
# -------------------------------
user_input = input("Enter a movie review: ")
cleaned_input = clean_text(user_input)
vectorized_input = tfidf.transform([cleaned_input])
prediction = model.predict(vectorized_input)

print("Predicted sentiment:", prediction[0])
