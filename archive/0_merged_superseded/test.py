# -*- coding: utf-8 -*-
"""
Created on Sat May 10 03:10:46 2025

@author: zaher
"""

# test_all.py
# generate_remi_events.py
# Generate and display REMI+ events from the trained Transformer model without metadata description

# test_samar_inspect.py
# Deep inspection tests for SAMAR system to investigate empty-bar generation behavior

import os
import torch
import json
from core import SAMARInputRepresentation
from data import SAMARDataset, SamarLatentDataset, precompute_samar_latents
from models.samar_vae import SamarVQVAE
from models.samar_transformer import SamarTransformer
from vocab_and_tokenizer import SamarTokenizer

# === Setup ===
xml_dir = "xml_data"
vae_ckpt_path = "checkpoints/samar_vae.pt"
transformer_ckpt_path = "checkpoints/samar_transformer.pt"
transformer_config_path = "checkpoints/samar_transformer_config.json"
latent_path = "latents/test_latents.pt"
tokenizer_path = "samar_vocab.pkl"

print("[0] Loading tokenizer and model configs...")
tokenizer = SamarTokenizer.load(tokenizer_path)
with open(transformer_config_path, "r") as f:
    transformer_config = json.load(f)

print("\n[1] Inspecting Input Representation for one MusicXML file")
xml_file = next((f for f in os.listdir(xml_dir) if f.endswith(".xml")), None)
if xml_file:
    full_path = os.path.join(xml_dir, xml_file)
    rep = SAMARInputRepresentation(full_path)
    events = rep.get_event_sequence()
    print(f"File: {xml_file}")
    print("Description Tokens:", rep.get_description_tokens())
    print("First 20 Event Tokens:", events[:20])
    print("Total Events:", len(events))
else:
    print("⚠️ No XML files found.")

print("\n[2] Inspecting Encoded Tokens")
encoded = tokenizer.encode(events)
decoded = tokenizer.decode(encoded)
print("Encoded (first 20):", encoded[:20])
print("Decoded Check (first 20):", decoded[:20])

print("\n[3] VAE Round Trip Test")
vae_ckpt = torch.load(vae_ckpt_path, map_location="cpu")
vae = SamarVQVAE(**vae_ckpt["config"])
vae.load_state_dict(vae_ckpt["model_state_dict"])
vae.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)
vae.device = device 

input_ids = torch.tensor(encoded[:128]).unsqueeze(0)  # 1 x T
with torch.no_grad():
    z_q = vae.encode_latent(input_ids)
    logits = vae.decode(z_q)
    predicted_ids = logits.argmax(dim=-1)
    print("Original:", tokenizer.decode(input_ids[0].tolist()))
    print("Reconstructed:", tokenizer.decode(predicted_ids[0].tolist()))

print("\n[4] Inspecting latent sample")
precompute_samar_latents(xml_dir, vae, latent_path, batch_size=2, context_size=128)
latent_data = torch.load(latent_path)
print("Loaded", len(latent_data), "latent samples.")
sample = latent_data[0]
print("Sample keys:", sample.keys())
print("Token length:", len(sample['tokens']))
print("Token values:", sample['tokens'][:20])
print("Latent shape:", sample['latent'].shape)

print("\n[5] Transformer Round Trip")
transformer = SamarTransformer(**transformer_config)
transformer.load_state_dict(torch.load(transformer_ckpt_path, map_location="cpu"))
transformer.eval()

input_ids = torch.tensor(sample['tokens'][:64]).unsqueeze(0)
latent = sample['latent'].unsqueeze(0)

with torch.no_grad():
    logits = transformer(input_ids, tgt=latent)
    predicted_ids = logits.permute(1, 0, 2).argmax(dim=-1)
    print("Input IDs:", input_ids.tolist()[0])
    print("Predicted IDs:", predicted_ids[0].tolist())
    print("Decoded:", tokenizer.decode(predicted_ids[0].tolist()))

print("\n✅ Deep inspection complete.")

################################
# inspect_vocab_and_vae.py
# Inspects vocabulary content, input/output coverage, and VAE reconstruction performance

import os
import pickle
import torch
from collections import Counter
from core import SAMARInputRepresentation
from vocab_and_tokenizer import Vocab, SamarTokenizer
from models.samar_vae import SamarVQVAE

# === Load vocab ===
vocab_path = "samar_vocab.pkl"
print("[1] Loading vocabulary from:", vocab_path)
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

print("✅ Vocab size:", len(vocab))
print("🔠 First 20 tokens:")
for tok, idx in list(vocab.stoi.items())[:20]:
    print(f"{idx:4} | {tok}")

# === Build tokenizer ===
tokenizer = SamarTokenizer(vocab)

# === Load one sample ===
print("\n[2] Loading and encoding a MusicXML sample...")
xml_file = next((f for f in os.listdir("xml_data") if f.endswith(".xml")), None)
if not xml_file:
    print("❌ No XML files found.")
    exit()

path = os.path.join("xml_data", xml_file)
rep = SAMARInputRepresentation(path)
events = rep.get_event_sequence()
desc = rep.get_description_tokens()
all_tokens = events + desc

print(f"File: {xml_file} | Total tokens: {len(all_tokens)}")
encoded = tokenizer.encode(all_tokens)
decoded = tokenizer.decode(encoded)

# === Report UNKs ===
unk_count = sum(1 for t in decoded if t == "<unk>")
print(f"\n[3] Token encoding stats:")
print(f"Total: {len(encoded)} | UNKs: {unk_count} | Known: {len(encoded) - unk_count}")

# Show side-by-side
print("\nFirst 30 tokens (original vs decoded):")
for i, (orig, rec) in enumerate(zip(all_tokens, decoded)):
    print(f"{i:02}: {orig:35} → {rec}")
    if i >= 29:
        break

# === VAE test ===
print("\n[4] Loading VAE and running round-trip test...")
vae_ckpt = torch.load("checkpoints/samar_vae.pt", map_location="cpu")
vae = SamarVQVAE(**vae_ckpt["config"])
vae.load_state_dict(vae_ckpt["model_state_dict"])
vae.eval()
vae.device = torch.device("cpu")

input_ids = torch.tensor(encoded[:128]).unsqueeze(0)
with torch.no_grad():
    z_q = vae.encode_latent(input_ids)
    logits = vae.decode(z_q)
    pred_ids = logits.argmax(dim=-1)
    rec_tokens = tokenizer.decode(pred_ids[0].tolist())

print("\n[5] VAE Reconstruction (first 30 tokens):")
for i, (orig, rec) in enumerate(zip(decoded, rec_tokens)):
    print(f"{i:02}: {orig:35} → {rec}")
    if i >= 29:
        break

print("\n✅ Inspection complete.")

