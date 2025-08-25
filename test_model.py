# ===============================
# test_model.py
# ===============================

import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
dataset_path = "dataset/test_data.csv"
df = pd.read_csv(dataset_path)

# Preprocessing
nltk.download("stopwords")
nltk.download("wordnet")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
text_col = "review"
label_col = "sentiment"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df[text_col] = df[text_col].astype(str).apply(clean_text)
X = df[text_col]
y_true = df[label_col]

# Load model and vectorizer
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# Vectorize and predict
X_tfidf = tfidf.transform(X)
y_pred = model.predict(X_tfidf)

# Print classification report and confusion matrix
print("\nClassification Report:\n", classification_report(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))

# Save confusion matrix as CSV
cm = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, columns=[f"Pred{i}" for i in range(cm.shape[1])],
                     index=[f"True{i}" for i in range(cm.shape[0])])
cm_df.to_csv("confusion_matrix.csv", index=True)

# Save predictions
pred_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
pred_df.to_csv("predictions.csv", index=False)

# Save confusion matrix as Markdown for README
cm_md = cm_df.to_markdown()
with open("confusion_matrix.md", "w") as f:
    f.write("## Confusion Matrix\n\n")
    f.write(cm_md)

# Save confusion matrix plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

print("Confusion matrix, predictions, and plot saved successfully!")
