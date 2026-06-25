# -*- coding: utf-8 -*-
"""
Created on Mon May 19 20:20:15 2025

@author: zaher
"""

# smart_generate.py

import torch
from model import SamarTransformer
from tokenizer import load_vocab
from reconstructor import reconstruct_musicxml_from_events

# === Load vocab & model
token2idx, idx2token, _ = load_vocab("data/samar_vocab.json")
vocab_size = len(token2idx)
pad_idx = token2idx["<PAD>"]
bos_idx = token2idx["<BOS>"]
eos_idx = token2idx["<EOS>"]
unk_idx = token2idx["<UNK>"]

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamarTransformer(vocab_size, pad_idx=pad_idx).to(device)
model.load_state_dict(torch.load("checkpoints/samar_transformer.pt", map_location=device))
model.eval()

# === Sampling functions
def sample_top_k(probs, k=8):
    top_probs, top_idxs = torch.topk(probs, k)
    top_probs = top_probs / top_probs.sum()
    return top_idxs[torch.multinomial(top_probs, 1)].item()

# === Description conditioning (optional)
description_tokens = [
    "Tempo_80",
    "KeySignature_0",
    "TimeSignature_4/4",
    "Instrument_Violin",
    "AveragePitch_127"
]

input_ids = [bos_idx] + [token2idx.get(t, unk_idx) for t in description_tokens]

# === Generation loop
max_len = 512
max_bars = 32
bar_count = 0
generated = input_ids[:]
note_buffer = {}
valid_sequence = []

for _ in range(max_len):
    x = torch.tensor([generated], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
    probs = torch.softmax(logits[0, -1], dim=0)

    next_token_id = sample_top_k(probs, k=8)
    next_token = idx2token[str(next_token_id)]

    if next_token == "<EOS>":
        break

    # Count bars
    if next_token.startswith("Bar_"):
        bar_count += 1
        if bar_count > max_bars:
            break
        valid_sequence.append(next_token)

    # Handle valid note structure
    elif next_token.startswith("Position_"):
        note_buffer = {"Position": next_token}

    elif next_token.startswith("Pitch_"):
        note_buffer["Pitch"] = next_token

    elif next_token.startswith("Velocity_"):
        note_buffer["Velocity"] = next_token

    elif next_token.startswith("Duration_"):
        note_buffer["Duration"] = next_token

    elif next_token.startswith("Instrument_"):
        note_buffer["Instrument"] = next_token

        if all(k in note_buffer for k in ["Position", "Pitch", "Velocity", "Duration", "Instrument"]):
            valid_sequence.extend([
                note_buffer["Position"],
                note_buffer["Pitch"],
                note_buffer["Velocity"],
                note_buffer["Duration"],
                note_buffer["Instrument"]
            ])
            note_buffer = {}

    # Add description tokens only once
    elif next_token.startswith("Tempo_") or next_token.startswith("KeySignature_") or next_token.startswith("TimeSignature_"):
        if next_token not in valid_sequence:
            valid_sequence.append(next_token)

    # Track full sequence for context
    generated.append(next_token_id)

# === Save .txt
with open("generated_smart.remi.txt", "w", encoding="utf-8") as f:
    for t in valid_sequence:
        f.write(t + "\n")

print("✅ Smart REMI+24 sequence saved to: generated_smart.remi.txt")

# === Reconstruct
reconstruct_musicxml_from_events(valid_sequence, "generated_smart_output.xml")
print("🎼 Reconstructed MusicXML saved to: generated_smart_output.xml")
