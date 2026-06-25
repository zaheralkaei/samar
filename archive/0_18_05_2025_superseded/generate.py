# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:02:53 2025

@author: zaher
"""

# === File: generate.py ===

# === File: generate.py ===

import torch
from model import SamarTransformer
from tokenizer import load_vocab
from reconstructor import reconstruct_musicxml_from_events

# === Load model and vocab ===
token2idx, idx2token, _ = load_vocab("data/samar_vocab.json")
vocab_size = len(token2idx)
pad_idx = token2idx["<PAD>"]
bos_idx = token2idx["<BOS>"]
eos_idx = token2idx["<EOS>"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamarTransformer(vocab_size, pad_idx=pad_idx).to(device)
model.load_state_dict(torch.load("checkpoints/samar_transformer.pt", map_location=device))
model.eval()

# === Top-1 Sampling ===
def sample(logits):
    probs = torch.softmax(logits, dim=0)
    return torch.argmax(probs).item()

# === Generate tokens ===
generated = [bos_idx]
for _ in range(512):
    x = torch.tensor([generated], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)[0, -1]
    next_token_id = sample(logits)
    if next_token_id == eos_idx:
        break
    generated.append(next_token_id)

# === Convert IDs back to tokens ===
tokens = [idx2token[str(t)] for t in generated if t not in [bos_idx, pad_idx]]

# === Save and reconstruct ===
with open("generated.remi.txt", "w", encoding="utf-8") as f:
    for t in tokens:
        f.write(t + "\n")

print("✅ Saved to generated.remi.txt")
reconstruct_musicxml_from_events(tokens, "generated_output.xml")
print("🎼 Reconstructed MusicXML saved to generated_output.xml")

