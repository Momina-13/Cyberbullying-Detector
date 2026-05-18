import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ── Custom Dataset ────────────────────────────────────────────────────────────
class CyberbullyingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts    = texts
        self.labels   = labels
        self.tokenizer = tokenizer
        self.max_len  = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.float)
        }


# ── Custom Classification Head ────────────────────────────────────────────────
class BertClassifier(nn.Module):
    """
    Uses pre-trained BERT weights (allowed).
    Classification head is our own nn.Module — NOT BertForSequenceClassification.
    """

    def __init__(self, bert_model_name='bert-base-uncased', dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Our custom classification head — written from scratch
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_output)
        return logits.squeeze(1)
