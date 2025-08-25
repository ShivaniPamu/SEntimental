
# ===============================
# train.py
# ==============================

import pandas as pd
import re
import nltk
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
dataset_path = "dataset/train_data.csv"
df = pd.read_csv(dataset_path)
print("Dataset loaded. Shape:", df.shape)

# Data cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

text_col = "review"
label_col = "sentiment"

# Text preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df[text_col] = df[text_col].astype(str).apply(clean_text)

# Vectorization and balancing
X = df[text_col]
y = df[label_col]

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_tfidf, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)
print("Model training completed.")

# Save model & vectorizer
os.makedirs("models", exist_ok=True)
with open("models/logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved in 'models/' folder.")
