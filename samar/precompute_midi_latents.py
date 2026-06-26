# -*- coding: utf-8 -*-
"""
Round-10: Precompute latents from MIDI dataset.

Mirrors samar/precompute_samar_latents.py but reads .mid files
instead of .xml/.mxl files. Saves the same latents.pt format so
the existing transformer training pipeline can use it unchanged.
"""

import argparse
import os
import torch
import numpy as np
from tqdm import tqdm

from .midi_loader import load_midi_as_samar
from .models.samar_vae import SamarVQVAE
from .tokenizer import SamarTokenizer
from .dataset import SAMARDataset


def collect_midi_samples(dataset_dir, context_size=256):
    """Walk dataset_dir, parse all .mid files, return token-id samples.

    Each sample is a dict with keys: tokens (list[int]), file (str),
    description (list[int] or None). Mirrors precompute_samar_latents.py.

    Returns: list of dicts, each up to context_size tokens.
    """
    tokenizer = SamarTokenizer.load(os.path.join("samar", "samar_vocab.pkl"))
    vocab = tokenizer.get_vocab()

    all_samples = []
    midi_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for fname in sorted(files):
            if fname.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, fname))

    print(f"Found {len(midi_files)} MIDI files in {dataset_dir}")

    for midi_path in tqdm(midi_files, desc="Parsing MIDI"):
        samar_ir = load_midi_as_samar(midi_path)
        if samar_ir is None or len(samar_ir.notes) == 0:
            continue
        events = samar_ir.events
        if len(events) < 10:
            continue

        # Convert events to token IDs
        token_ids = [vocab.to_i(e) for e in events if e in vocab.stoi]

        # Chunk into context_size windows
        for start in range(0, len(token_ids), context_size):
            chunk = token_ids[start:start + context_size]
            if len(chunk) >= 32:  # skip tiny chunks
                all_samples.append({
                    "tokens": chunk,
                    "file": os.path.basename(midi_path),
                    "description": None,
                })

    print(f"Total samples: {len(all_samples)}")
    return all_samples


def compute_and_save_latents(dataset_dir, output_path, context_size=256):
    """Compute VAE latents for each MIDI sample and save to output_path."""
    samples = collect_midi_samples(dataset_dir, context_size)
    if len(samples) == 0:
        raise RuntimeError(f"No usable samples found in {dataset_dir}")

    tokenizer = SamarTokenizer.load(os.path.join("samar", "samar_vocab.pkl"))
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = SamarVQVAE(vocab_size=vocab_size, d_model=128)

    # Try to load pre-trained VAE weights
    vae_ckpt = os.path.join("checkpoints", "samar_vae.pt")
    if os.path.exists(vae_ckpt):
        ckpt = torch.load(vae_ckpt, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            vae.load_state_dict(ckpt["model_state_dict"])
        else:
            vae.load_state_dict(ckpt)
        print(f"Loaded VAE from {vae_ckpt}")
    else:
        print(f"WARNING: no VAE checkpoint at {vae_ckpt}, using random init")

    vae.to(device).eval()

    # Encode each sample
    all_latents = []
    with torch.no_grad():
        for sample in tqdm(samples, desc="Encoding VAE latents"):
            tokens = sample["tokens"][:context_size]
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            latent = vae.encode_latent(x)  # [1, T, latent_dim]
            all_latents.append({
                "tokens": tokens,
                "latent": latent.squeeze(0).cpu(),
                "file": sample["file"],
                "description": sample["description"],
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(all_latents, output_path)
    print(f"Saved {len(all_latents)} samples to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/midi_dataset",
                        help="Directory containing MIDI files (recursive)")
    parser.add_argument("--output", default="latents/midi_latents.pt",
                        help="Output latents.pt path")
    parser.add_argument("--context-size", type=int, default=256)
    args = parser.parse_args()
    compute_and_save_latents(args.dataset_dir, args.output, args.context_size)