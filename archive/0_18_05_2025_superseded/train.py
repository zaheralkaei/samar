# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:29:54 2025

@author: zaher
"""
# === File: train.py ===

import torch
from torch.utils.data import DataLoader
from model import SamarTransformer
from tokenizer import load_vocab
from collate import pad_collate
from dataset import SamarDataset

# === Load vocab ===
token2idx, _, _ = load_vocab("data/samar_vocab.json")
vocab_size = len(token2idx)
pad_idx = token2idx["<PAD>"]

# === Dataset and Dataloader ===
dataset = SamarDataset("data/tokenized", max_len=256)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda b: pad_collate(b, pad_idx))

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SamarTransformer(vocab_size, pad_idx=pad_idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

# === Train Loop ===
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# === Save Model ===
torch.save(model.state_dict(), "checkpoints/samar_transformer.pt")
print("✅ Model saved to checkpoints/samar_transformer.pt")
