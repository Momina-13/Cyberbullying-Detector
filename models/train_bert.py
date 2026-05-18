"""
Member 3 — Train BERT with custom classification head
Run: python train_bert.py
NOTE: First run: pip install transformers torch pandas numpy
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import BertTokenizer
from torch.utils.data import DataLoader
from bert_model import BertClassifier, CyberbullyingDataset
from evaluate import compute_metrics, print_metrics

# ── Config ────────────────────────────────────────────────────────────────────
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv('../dataset/train.csv')
val   = pd.read_csv('../dataset/val.csv')
test  = pd.read_csv('../dataset/test.csv')

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# ── Datasets and DataLoaders ──────────────────────────────────────────────────
train_dataset = CyberbullyingDataset(train['text'].tolist(), train['label'].tolist(), tokenizer, MAX_LEN)
val_dataset   = CyberbullyingDataset(val['text'].tolist(),   val['label'].tolist(),   tokenizer, MAX_LEN)
test_dataset  = CyberbullyingDataset(test['text'].tolist(),  test['label'].tolist(),  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading BERT model with custom classification head...")
model = BertClassifier('bert-base-uncased').to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nFine-tuning BERT...")
os.makedirs('../saved_models', exist_ok=True)
best_val_f1 = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Validation
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label']
            logits = model(input_ids, attention_mask)
            preds  = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            val_preds.extend(preds)
            val_true.extend(labels.numpy().astype(int))

    m = compute_metrics(val_true, val_preds)
    print(f"\nEpoch {epoch+1}/{EPOCHS} | Avg Loss: {total_loss/len(train_loader):.4f} | Val F1: {m['F1-Score']}%\n")

    if m['F1-Score'] > best_val_f1:
        best_val_f1 = m['F1-Score']
        torch.save(model.state_dict(), '../saved_models/bert_model.pt')
        print(f"  --> Best model saved (Val F1: {best_val_f1}%)")

# ── Test evaluation ───────────────────────────────────────────────────────────
print("\nLoading best model for test evaluation...")
model.load_state_dict(torch.load('../saved_models/bert_model.pt', map_location=DEVICE))
model.eval()

test_preds, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label']
        logits = model(input_ids, attention_mask)
        preds  = (

