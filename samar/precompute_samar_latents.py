# -*- coding: utf-8 -*-
"""
Precompute training data: events + descriptions + structural IDs.

Round 18: VQ-VAE removed. This now just parses XML, tokenizes, computes
bar_ids/position_ids, and saves to latents.pt for the trainer.
"""

import os
import torch
from tqdm import tqdm

from .dataset import SAMARDataset, compute_structural_ids, compute_desc_bar_ids
from .tokenizer import SamarTokenizer, DescriptionTokenizer
from .constants import BOS_TOKEN, EOS_TOKEN

tokenizer = SamarTokenizer.load(
    os.path.join(os.path.dirname(__file__), "samar_vocab.pkl")
)
desc_tokenizer = DescriptionTokenizer()

# === Config ===
CONTEXT_SIZE = 256
XML_DATA_DIR = "./data/xml"
LATENT_SAVE_PATH = "./latents/latents.pt"
MIN_CHUNK_LEN = 8

# === Dataset ===
print("Loading XML files from:", XML_DATA_DIR)
dataset = SAMARDataset(
    XML_DATA_DIR,
    context_size=CONTEXT_SIZE,
    tokenizer=tokenizer,
    min_chunk_len=MIN_CHUNK_LEN,
)

print(f"Found {len(dataset.files)} MusicXML files")
print(f"Total token chunks prepared: {len(dataset)}")

# === Save ===
print("Encoding and storing data...")
all_samples = []

for i in tqdm(range(len(dataset))):
    example = dataset.examples[i]
    all_samples.append({
        "tokens": example["tokens"],
        "bar_ids": example["bar_ids"],
        "position_ids": example["position_ids"],
        "description": example["description"],
        "desc_bar_ids": example["desc_bar_ids"],
        "file": os.path.basename(example["file"]),
    })

os.makedirs(os.path.dirname(LATENT_SAVE_PATH), exist_ok=True)
torch.save(all_samples, LATENT_SAVE_PATH)
print(f"Saved {len(all_samples)} samples to: {LATENT_SAVE_PATH}")

# Summary stats
token_lens = [len(s["tokens"]) for s in all_samples]
print(f"Token lengths: min={min(token_lens)}, max={max(token_lens)}, "
      f"mean={sum(token_lens)/len(token_lens):.0f}")
