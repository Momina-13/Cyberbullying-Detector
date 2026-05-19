"""
Member 2 — Step 1: Train Logistic Regression with TF-IDF
Run: python train_lr.py
"""

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tfidf import TFIDFVectorizer
from logistic_regression import LogisticRegression
from evaluate import compute_metrics, print_metrics
import pickle

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv('../dataset/train.csv')
val   = pd.read_csv('../dataset/val.csv')
test  = pd.read_csv('../dataset/test.csv')

X_train, y_train = train['text'].tolist(), train['label'].values.astype(float)
X_val,   y_val   = val['text'].tolist(),   val['label'].values.astype(float)
X_test,  y_test  = test['text'].tolist(),  test['label'].values.astype(float)

# ── TF-IDF ────────────────────────────────────────────────────────────────────
print("\nBuilding TF-IDF features...")
vectorizer = TFIDFVectorizer(max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf   = vectorizer.transform(X_val)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# ── Train Logistic Regression ─────────────────────────────────────────────────
print("\nTraining Logistic Regression...")
model = LogisticRegression(lr=0.1, epochs=200)
model.fit(X_train_tfidf, y_train)

# ── Evaluate on validation set ────────────────────────────────────────────────
val_preds = model.predict(X_val_tfidf)
val_metrics = compute_metrics(y_val, val_preds)
print_metrics("Logistic Regression — Validation", val_metrics)

# ── Evaluate on test set ──────────────────────────────────────────────────────
test_preds = model.predict(X_test_tfidf)
test_metrics = compute_metrics(y_test, test_preds)
print_metrics("Logistic Regression — Test", test_metrics)

# ── Save model ────────────────────────────────────────────────────────────────
os.makedirs('../saved_models', exist_ok=True)
with open('../saved_models/lr_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'vectorizer': vectorizer}, f)
print("\nModel saved to saved_models/lr_model.pkl")

# Save results for comparison
results = {
    'Model': 'Logistic Regression',
    'Accuracy':  test_metrics['Accuracy'],
    'Precision': test_metrics['Precision'],
    'Recall':    test_metrics['Recall'],
    'F1-Score':  test_metrics['F1-Score']
}
pd.DataFrame([results]).to_csv('../saved_models/lr_results.csv', index=False)
print("Results saved to saved_models/lr_results.csv")
