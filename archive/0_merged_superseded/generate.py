# -*- coding: utf-8 -*-
"""
Created on Sat May 10 03:39:58 2025

@author: zaher
"""

# generate_remi_events.py
# Generate and display REMI+ events from the trained Transformer model

import torch
import json
from models.samar_transformer import SamarTransformer
from vocab_and_tokenizer import SamarTokenizer

# === Load model config and weights ===
with open("checkpoints/samar_transformer_config.json", "r") as f:
    config = json.load(f)

model = SamarTransformer(**config)
model.load_state_dict(torch.load("checkpoints/samar_transformer.pt", map_location="cpu"))
model.eval()

# === Load tokenizer ===
tokenizer = SamarTokenizer.load("samar_vocab.pkl")

# === Prepare prompt ===
prompt = [
    "TimeSignature_4/4", "Tempo_10", "KeySignature_0",
    "Bar_1", "NoteDensity_5", "MeanPitch_18", "MeanVelocity_10",
    "Position_0", "Pitch_48", "Velocity_16", "Duration_10",
    "Position_8", "Pitch_50", "Velocity_16", "Duration_10",
    "Bar_2"
]

input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)  # [1, T]

# === Generation Parameters ===
max_length = 128
temperature = 1.0  # set < 1.0 for more conservative outputs
pad_id = tokenizer.get_vocab().pad_id

# === Generate ===
with torch.no_grad():
    model.eval()
    generated = input_ids
    for _ in range(max_length - input_ids.size(1)):
        logits = model(generated)  # [seq_len, batch, dim]
        logits = logits.permute(1, 0, 2)  # [batch, seq_len, dim]
        next_logits = logits[:, -1, :]  # [batch, dim]

        probs = torch.softmax(next_logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # sample

        generated = torch.cat([generated, next_token], dim=1)

        if next_token.item() == pad_id:
            break

# === Decode ===
generated_events = tokenizer.decode(generated[0].tolist())

# === Print filtered result ===
seen_bars = set()
print("\nGenerated REMI+ Events:\n")
for event in generated_events:
    if event.startswith("Bar_"):
        if event in seen_bars:
            break  # stop if repeating the same bar endlessly
        seen_bars.add(event)
    print(event)