# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:21:24 2025

@author: zaher
"""

# === File: precompute_samar_latents.py ===

import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from .models.samar_vae import SamarVQVAE
from .dataset import SAMARDataset
from .tokenizer import SamarTokenizer, DescriptionTokenizer
tokenizer = SamarTokenizer.load(os.path.join(os.path.dirname(__file__), "samar_vocab.pkl"))
desc_tokenizer = DescriptionTokenizer()

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
CONTEXT_SIZE = 256  # Token sequence length
BATCH_SIZE = 2  # Batch size for DataLoader
XML_DATA_DIR = "./data/xml"  # Path to MusicXML files (relative to project root)
CHECKPOINT_PATH = "./checkpoints/samar_vae.pt"  # Path to pre-trained VAE model
LATENT_SAVE_PATH = "./latents/latents.pt"  # Output file for saved latent representations

# === Custom collate function ===
# Pads input sequences to the same length and returns a batch dictionary
def pad_collate(batch):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
    files = [item["file"] for item in batch]

    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids_padded,
        "labels": labels_padded,
        "file": files
    }

# === Load VAE ===
# Load the pretrained SAMAR VQ-VAE model and move it to the selected device
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
vae = SamarVQVAE(**ckpt["config"])
vae.load_state_dict(ckpt["model_state_dict"])
vae.to(DEVICE).eval()

# === Dataset and Loader ===
print("Loading XML files from:", XML_DATA_DIR)
dataset = SAMARDataset(XML_DATA_DIR, context_size=CONTEXT_SIZE, tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

print(f"Found {len(dataset.files)} MusicXML files")
print(f"Total token chunks prepared: {len(dataset)}")

# === Encode and save ===
print("Encoding and storing latents...")
all_latents = []  # List to store encoded latent vectors with tokens and file names

# Disable gradient computation for inference
with torch.no_grad():
    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(DEVICE)
        latents = vae.encode_latent(input_ids)  # Encode input sequences into latent vectors

        # Store tokens, latents, and per-bar descriptions per example.
        # Round-5: descriptions are needed for description-conditional
        # generation; encode via DescriptionTokenizer against
        # DescriptionVocab (matches FIGARO's separate-vocab split).
        for i in range(input_ids.size(0)):
            desc_tokens = dataset.examples[len(all_latents)]["description"]
            desc_ids = desc_tokenizer.encode(desc_tokens)
            all_latents.append({
                "tokens": input_ids[i].cpu().tolist(),    # Event token IDs (SamarVocab)
                "latent": latents[i].cpu(),                # VAE latent vector
                "file": batch["file"][i],                  # Source filename
                "description": desc_ids,                   # Description token IDs (DescriptionVocab)
            })

# === Save as a single .pt file ===
# Ensure output directory exists and save all latents
os.makedirs(os.path.dirname(LATENT_SAVE_PATH), exist_ok=True)
torch.save(all_latents, LATENT_SAVE_PATH)
print("Latent vectors saved to:", LATENT_SAVE_PATH)
