# ===============================
# test_model.py
# ===============================

import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Load dataset
# -------------------------------
dataset_path = "dataset/test_data.csv"
df = pd.read_csv(dataset_path)

# -------------------------------
# Preprocessing
# -------------------------------
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

# -------------------------------
# Load model and vectorizer
# -------------------------------
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Vectorize and predict
# -------------------------------
X_tfidf = tfidf.transform(X)
y_pred = model.predict(X_tfidf)

# -------------------------------
# Metrics
# -------------------------------
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label='positive')
rec = recall_score(y_true, y_pred, pos_label='positive')
f1 = f1_score(y_true, y_pred, pos_label='positive')

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nClassification Report:\n", classification_report(y_true, y_pred))
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

# -------------------------------
# Save confusion matrix as CSV
# -------------------------------
cm_df = pd.DataFrame(cm, columns=[f"Pred{i}" for i in range(cm.shape[1])],
                     index=[f"True{i}" for i in range(cm.shape[0])])
cm_df.to_csv("confusion_matrix.csv", index=True)

# -------------------------------
# Save predictions
# -------------------------------
pred_df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
pred_df.to_csv("predictions.csv", index=False)

# -------------------------------
# Save confusion matrix as Markdown for README
# -------------------------------
cm_md = cm_df.to_markdown()
with open("confusion_matrix.md", "w") as f:
    f.write("## Confusion Matrix\n\n")
    f.write(cm_md)

f_metrics = f"## Metrics\n\n- Accuracy: {acc:.4f}\n- Precision: {prec:.4f}\n- Recall: {rec:.4f}\n- F1-score: {f1:.4f}\n"
with open("metrics.md", "w") as f:
    f.write(f_metrics)

# -------------------------------
# Save confusion matrix plot
# -------------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")

# -------------------------------
# Plot training/testing loss if available
# -------------------------------
history_path = "models/train_history.pkl"
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history = pickle.load(f)
    
    plt.figure(figsize=(8,6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.title("Training and Testing Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")

print("Confusion matrix, metrics, predictions, and plots saved successfully!")
