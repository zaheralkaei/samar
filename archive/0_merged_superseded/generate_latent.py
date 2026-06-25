# -*- coding: utf-8 -*-
"""
Created on Sat May 10 03:30:50 2025

@author: zaher
"""

# clean_and_rebuild_latents.py
# Deletes outdated latent file and regenerates it using current vocab and VAE

import os
import torch
from data import precompute_samar_latents
from models.samar_vae import SamarVQVAE
from vocab_and_tokenizer import SamarTokenizer

# === CONFIG ===
xml_dir = "xml_data"
vae_ckpt_path = "checkpoints/samar_vae.pt"
vocab_path = "samar_vocab.pkl"
output_path = "latents/test_latents.pt"

# === Step 1: Delete existing latents ===
if os.path.exists(output_path):
    os.remove(output_path)
    print(f"🗑️ Deleted old latent file: {output_path}")
else:
    print("ℹ️ No previous latent file found.")

# === Step 2: Load VAE ===
print("📦 Loading VAE model from:", vae_ckpt_path)
checkpoint = torch.load(vae_ckpt_path, map_location="cpu")
vae = SamarVQVAE(**checkpoint["config"])
vae.load_state_dict(checkpoint["model_state_dict"])
vae.eval()
vae.device = torch.device("cpu")
print("✅ VAE loaded.")

# === Step 3: Regenerate latents ===
print("🔁 Regenerating latents from:", xml_dir)
precompute_samar_latents(xml_dir, vae, output_path, batch_size=2, context_size=128)
print("✅ New latents saved to:", output_path)
