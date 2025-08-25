# ===============================
# train.py
# ===============================

import pandas as pd
import re
import nltk
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
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

# Model training with loss tracking
model = LogisticRegression(max_iter=1, warm_start=True)  # warm_start=True allows incremental training
epochs = 50
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.fit(X_train, y_train)
    
    y_train_prob = model.predict_proba(X_train)
    y_test_prob = model.predict_proba(X_test)
    
    train_loss = log_loss(y_train, y_train_prob)
    test_loss = log_loss(y_test, y_test_prob)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    print(f"Epoch {epoch+1}/{epochs} — Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Save model & vectorizer
os.makedirs("models", exist_ok=True)
with open("models/logreg_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# Save loss CSV
loss_df = pd.DataFrame({
    "epoch": list(range(1, epochs+1)),
    "train_loss": train_losses,
    "test_loss": test_losses
})
loss_df.to_csv("loss.csv", index=False)

# Plot loss graph
plt.figure(figsize=(8,6))
plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss")
plt.plot(loss_df["epoch"], loss_df["test_loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training and Testing Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")

print("Model, vectorizer, loss CSV, and loss plot saved in 'models/' folder.")
