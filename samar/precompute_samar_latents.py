# -*- coding: utf-8 -*-
"""
Precompute training data: events + descriptions + structural IDs.

Round 18: VQ-VAE removed. This now just parses XML, tokenizes, computes
bar_ids/position_ids, and saves to latents.pt for the trainer.
"""

import argparse
import os
import torch
from tqdm import tqdm

from .dataset import SAMARDataset, compute_structural_ids, compute_desc_bar_ids
from .tokenizer import SamarTokenizer, DescriptionTokenizer
from .constants import BOS_TOKEN, EOS_TOKEN


def _build_default_tokenizer():
    return SamarTokenizer.load(
        os.path.join(os.path.dirname(__file__), "samar_vocab.pkl")
    )


def precompute(data_dir, save_path, context_size=256, min_chunk_len=8,
               tokenizer=None):
    """Walk data_dir for MusicXML files, compute samples, save to save_path.

    Returns: list of saved samples (each is a dict with tokens/bar_ids/...).
    """
    if tokenizer is None:
        tokenizer = _build_default_tokenizer()
    desc_tokenizer = DescriptionTokenizer()

    print(f"Loading XML files from: {data_dir}")
    dataset = SAMARDataset(
        data_dir,
        context_size=context_size,
        tokenizer=tokenizer,
        min_chunk_len=min_chunk_len,
    )

    print(f"Found {len(dataset.files)} MusicXML files")
    print(f"Total token chunks prepared: {len(dataset)}")
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

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(all_samples, save_path)
    print(f"Saved {len(all_samples)} samples to: {save_path}")

    token_lens = [len(s["tokens"]) for s in all_samples]
    print(f"Token lengths: min={min(token_lens)}, max={max(token_lens)}, "
          f"mean={sum(token_lens)/len(token_lens):.0f}")

    return all_samples


def main():
    parser = argparse.ArgumentParser(
        description="Precompute r18-schema latents for SAMAR training.",
    )
    parser.add_argument("--data-dir", default="./data/xml",
                        help="Directory of .xml/.mxl files (recursive).")
    parser.add_argument("--output", default="./latents/latents.pt",
                        help="Output latents.pt path.")
    parser.add_argument("--context-size", type=int, default=256,
                        help="Context size for chunking.")
    parser.add_argument("--min-chunk-len", type=int, default=8,
                        help="Minimum samples length to keep.")
    args = parser.parse_args()

    precompute(
        data_dir=args.data_dir,
        save_path=args.output,
        context_size=args.context_size,
        min_chunk_len=args.min_chunk_len,
    )


if __name__ == "__main__":
    main()
