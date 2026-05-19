"""
Member 2 — Step 2: Train LSTM from scratch
Run: python train_lstm.py
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lstm_model import LSTMFromScratch, build_vocab, encode
from evaluate import compute_metrics, print_metrics

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_DIM  = 64
HIDDEN_DIM = 128
MAX_LEN    = 50
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 0.001
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv('../dataset/train.csv')
val   = pd.read_csv('../dataset/val.csv')
test  = pd.read_csv('../dataset/test.csv')

# ── Build vocabulary ──────────────────────────────────────────────────────────
print("Building vocabulary...")
vocab = build_vocab(train['text'].tolist(), max_vocab=5000)
VOCAB_SIZE = len(vocab)
print(f"Vocabulary size: {VOCAB_SIZE}")

# ── Encode texts ──────────────────────────────────────────────────────────────
def make_tensors(df):
    X = torch.tensor([encode(t, vocab, MAX_LEN) for t in df['text']], dtype=torch.long)
    y = torch.tensor(df['label'].values, dtype=torch.float)
    return X, y

X_train, y_train = make_tensors(train)
X_val,   y_val   = make_tensors(val)
X_test,  y_test  = make_tensors(test)

# ── DataLoader ────────────────────────────────────────────────────────────────
from torch.utils.data import TensorDataset, DataLoader

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
model = LSTMFromScratch(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=1
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nTraining LSTM...")
best_val_f1 = 0
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(DEVICE)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            val_preds.extend(preds)
            val_true.extend(y_batch.numpy().astype(int))

    m = compute_metrics(val_true, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val F1: {m['F1-Score']}%")

    if m['F1-Score'] > best_val_f1:
        best_val_f1 = m['F1-Score']
        torch.save({'model_state': model.state_dict(), 'vocab': vocab},
                   '../saved_models/lstm_model.pt')

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\nLoading best model for test evaluation...")
checkpoint = torch.load('../saved_models/lstm_model.pt', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
        test_preds.extend(preds)
        test_true.extend(y_batch.numpy().astype(int))

test_metrics = compute_metrics(test_true, test_preds)
print_metrics("LSTM — Test", test_metrics)

results = {
    'Model': 'LSTM',
    'Accuracy':  test_metrics['Accuracy'],
    'Precision': test_metrics['Precision'],
    'Recall':    test_metrics['Recall'],
    'F1-Score':  test_metrics['F1-Score']
}
pd.DataFrame([results]).to_csv('../saved_models/lstm_results.csv', index=False)
print("Results saved to saved_models/lstm_results.csv")
