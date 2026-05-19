import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# ── Vocabulary builder ────────────────────────────────────────────────────────
def build_vocab(texts, max_vocab=5000):
    freq = {}
    for text in texts:
        for token in str(text).lower().split():
            freq[token] = freq.get(token, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in sorted_words[:max_vocab - 2]:
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=50):
    tokens = str(text).lower().split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    # Pad or truncate to max_len
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


# ── LSTM from scratch ─────────────────────────────────────────────────────────
class LSTMFromScratch(nn.Module):
    """
    LSTM implemented from scratch using nn.Parameter.
    No nn.LSTM used anywhere — all gate matrices defined manually.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, pad_idx=0):
        super().__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.hidden_dim = hidden_dim

        # ── Input gate weights ────────────────────────────────────────────────
        self.W_ii = nn.Parameter(torch.randn(embed_dim,  hidden_dim) * 0.01)
        self.W_hi = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_i  = nn.Parameter(torch.zeros(hidden_dim))

        # ── Forget gate weights ───────────────────────────────────────────────
        self.W_if = nn.Parameter(torch.randn(embed_dim,  hidden_dim) * 0.01)
        self.W_hf = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_f  = nn.Parameter(torch.ones(hidden_dim))   # init forget bias to 1

        # ── Cell gate weights ─────────────────────────────────────────────────
        self.W_ig = nn.Parameter(torch.randn(embed_dim,  hidden_dim) * 0.01)
        self.W_hg = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_g  = nn.Parameter(torch.zeros(hidden_dim))

        # ── Output gate weights ───────────────────────────────────────────────
        self.W_io = nn.Parameter(torch.randn(embed_dim,  hidden_dim) * 0.01)
        self.W_ho = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_o  = nn.Parameter(torch.zeros(hidden_dim))

        # ── Final classifier ──────────────────────────────────────────────────
        self.fc = nn.Linear(hidden_dim, output_dim)

    def lstm_step(self, x_t, h_prev, c_prev):
        """One LSTM time step — manually computing all 4 gates."""
        i = torch.sigmoid(x_t @ self.W_ii + h_prev @ self.W_hi + self.b_i)
        f = torch.sigmoid(x_t @ self.W_if + h_prev @ self.W_hf + self.b_f)
        g = torch.tanh   (x_t @ self.W_ig + h_prev @ self.W_hg + self.b_g)
        o = torch.sigmoid(x_t @ self.W_io + h_prev @ self.W_ho + self.b_o)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)   # (batch, seq_len, embed_dim)
        batch_size, seq_len, _ = embedded.shape

        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(seq_len):
            x_t = embedded[:, t, :]   # (batch, embed_dim)
            h, c = self.lstm_step(x_t, h, c)

        out = self.fc(h)               # (batch, 1)
        return out.squeeze(1)
