# -*- coding: utf-8 -*-
"""
Created on Fri May  9 03:56:25 2025

@author: zaher
"""
# === File: generating.py ===
# Inference: sample with VAE+Transformer 

import torch
import json
import os
from .models.samar_vae import SamarVQVAE
from .models.samar_transformer import SamarTransformer
from .tokenizer import SamarTokenizer
from .reconstructor import reconstruct_musicxml_from_events


# === Load tokenizer and vocab ===
tokenizer = SamarTokenizer.load(os.path.join(os.path.dirname(__file__), "samar_vocab.pkl"))
vocab = tokenizer.get_vocab()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Load VAE ===
vae_ckpt = "checkpoints/samar_vae.pt"
vae_checkpoint = torch.load(vae_ckpt, map_location=device)
vae = SamarVQVAE(**vae_checkpoint["config"])
vae.load_state_dict(vae_checkpoint["model_state_dict"])
vae.to(device).eval()

# === Load Transformer ===
lm_ckpt = "checkpoints/samar_transformer.pt"
config_path = "checkpoints/samar_transformer_config.json"

with open(config_path, 'r') as f:
    lm_config = json.load(f)

lm = SamarTransformer(**lm_config)
# Use from_pretrained: handles checkpoint/model mismatches (e.g. description_embedding
# and pos_embedding were added after the checkpoint was saved) by loading with
# strict=False and warm-starting missing embeddings from existing weights.
lm, _load_report = SamarTransformer.from_pretrained(lm_ckpt, config=lm_config, device=str(device))
if _load_report["missing"]:
    print(f"[load] warm-started missing layers: {_load_report['missing']}")
if _load_report["unexpected"]:
    print(f"[load] ignored unexpected keys: {_load_report['unexpected']}")
lm.to(device).eval()

# === Load latent from dataset ===
latent_data = torch.load("latents/latents.pt")
sample = latent_data[15]  # Pick an example with useful structure
seed_latent = sample["latent"].unsqueeze(0).to(device)
# Model expects [B, T, latent_dim]; a single latent vector broadcasts across T
if seed_latent.dim() == 2:
    seed_latent = seed_latent.unsqueeze(1)
print(f"Using latent from file: {sample['file']}")

# === Sample token sequence from Transformer ===
start_bar_token = torch.tensor([[vocab.to_i("Bar_0")]], device=device)

with torch.no_grad():
    gen_token_ids = lm.sample(
        start_tokens=start_bar_token,
        latent=seed_latent,
        max_length=512,
        pad_id=getattr(vocab, "pad_id", None)
    )

# === Decode generated tokens into REMI+ events ===
gen_token_list = gen_token_ids.squeeze().cpu().tolist()
gen_events = vocab.decode(gen_token_list)

# === Print the REMI+ output ===
print("\n=== Generated REMI+ Event Tokens ===")
for i, event in enumerate(gen_events):
    print(f"{i+1:03d}: {event}")

# === Check for unknown tokens ===
if any(e == "<unk>" for e in gen_events):
    print("Warning: Generated sequence contains unknown tokens.")

# === Save REMI+ event tokens to file
with open("generated_events.txt", "w", encoding="utf-8") as f:
    for event in gen_events:
        f.write(event + "\n")
print("Saved generated event sequence to 'generated_events.txt'")

# === Convert REMI+ tokens to MusicXML and save
output_xml_path = "generated.xml"
reconstruct_musicxml_from_events(gen_events, "generated.xml")
print(f"Saved generated MusicXML to: {output_xml_path}")