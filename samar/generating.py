# -*- coding: utf-8 -*-
"""
Inference: sample from a trained VAE + Transformer.

Three modes:

1. ``python -m samar.generating`` (default)
   Uses sample[15] from ``latents/latents.pt`` as the seed latent AND its
   description. Reconstructs to ``generated.xml``.

2. ``python -m samar.generating --description-source path/to/piece.xml``
   Uses the given XML's description tokens for conditioning. Useful for
   \"generate a piece in the same maqam/style as this template\".

3. ``python -m samar.generating --latent-index 42``
   Pick a different latent from ``latents/latents.pt`` (default 15).

Architecture follows FIGARO's two-stream design: the transformer gets a
(latent, description, start_tokens) triple and autoregressively generates
event tokens.
"""

import argparse
import json
import os
import sys

import torch

from .models.samar_vae import SamarVQVAE
from .models.samar_transformer import SamarTransformer
from .tokenizer import SamarTokenizer, DescriptionTokenizer
from .reconstructor import reconstruct_musicxml_from_events
from .input_representation import SAMARInputRepresentation

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(p):
    """Resolve a path relative to the repo root if not absolute."""
    if os.path.isabs(p):
        return p
    return os.path.join(REPO_ROOT, p)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MusicXML from a trained VAE + Transformer.",
    )
    parser.add_argument(
        "--latent-index", type=int, default=15,
        help="Index into latents/latents.pt to use as the seed latent. Default: 15.",
    )
    parser.add_argument(
        "--description-source",
        help="Path to an XML file whose description tokens should be used. "
             "If omitted, uses the description of latent[--latent-index].",
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Max generation length. Default: 256 (matches training context_size).",
    )
    parser.add_argument(
        "--output-xml", default="generated.xml",
        help="Where to write the generated MusicXML. Default: generated.xml.",
    )
    parser.add_argument(
        "--output-events", default="generated_events.txt",
        help="Where to write the generated event tokens. Default: generated_events.txt.",
    )
    args = parser.parse_args()

    # === Load tokenizer + vocab ===
    tokenizer = SamarTokenizer.load(
        os.path.join(os.path.dirname(__file__), "samar_vocab.pkl"),
    )
    vocab = tokenizer.get_vocab()
    desc_tokenizer = DescriptionTokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load VAE ===
    vae_ckpt = _resolve_path("checkpoints/samar_vae.pt")
    vae_checkpoint = torch.load(vae_ckpt, map_location=device)
    vae = SamarVQVAE(**vae_checkpoint["config"])
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae.to(device).eval()

    # === Load Transformer (warm-starts missing layers automatically) ===
    lm_ckpt = _resolve_path("checkpoints/samar_transformer.pt")
    config_path = _resolve_path("checkpoints/samar_transformer_config.json")
    with open(config_path) as f:
        lm_config = json.load(f)

    lm, load_report = SamarTransformer.from_pretrained(
        lm_ckpt, config=lm_config, device=str(device),
    )
    if load_report["missing"]:
        print(f"[load] warm-started missing layers: {load_report['missing']}")
    if load_report["unexpected"]:
        print(f"[load] ignored unexpected keys: {load_report['unexpected']}")
    lm.to(device).eval()

    # === Load latent + description ===
    latent_path = _resolve_path("latents/latents.pt")
    latent_data = torch.load(latent_path, map_location=device, weights_only=False)
    if not latent_data:
        print("ERROR: latents.pt is empty. Run "
              "`python -m samar.precompute_samar_latents` first.")
        sys.exit(1)
    if args.latent_index >= len(latent_data):
        print(f"ERROR: latent_index {args.latent_index} out of range "
              f"(have {len(latent_data)} samples).")
        sys.exit(1)

    sample = latent_data[args.latent_index]
    seed_latent = sample["latent"].to(device)
    if seed_latent.dim() == 2:
        # [T, latent_dim] -> [1, 1, latent_dim] so the model can broadcast
        seed_latent = seed_latent.unsqueeze(0).unsqueeze(1)
    elif seed_latent.dim() == 1:
        seed_latent = seed_latent.unsqueeze(0).unsqueeze(0)
    print(f"Using latent from file: {sample['file']} (index {args.latent_index})")

    # === Description: either from XML file or from the latent sample ===
    if args.description_source:
        xml_path = _resolve_path(args.description_source)
        print(f"Loading description from: {xml_path}")
        ir = SAMARInputRepresentation(xml_path)
        desc_tokens = ir.get_description_tokens()
    else:
        # Use the description stored in the latent sample (round-5)
        desc_tokens = sample.get("description")
        if desc_tokens is None:
            print("WARNING: no description stored in this latent sample, "
                  "falling back to empty description.")
            desc_tokens = []
    desc_ids = desc_tokenizer.encode(desc_tokens)
    desc_tensor = torch.tensor([desc_ids], dtype=torch.long, device=device)
    print(f"Description: {len(desc_tokens)} tokens ({len(desc_ids)} IDs)")

    # === Sample event tokens from Transformer ===
    start_bar_token = torch.tensor([[vocab.to_i("Bar_0")]], device=device)

    with torch.no_grad():
        gen_token_ids = lm.sample(
            start_tokens=start_bar_token,
            latent=seed_latent,
            description=desc_tensor,
            max_length=args.max_length,
            pad_id=getattr(vocab, "pad_id", None),
        )

    # === Decode generated tokens into REMI+ events ===
    gen_token_list = gen_token_ids.squeeze().cpu().tolist()
    gen_events = vocab.decode(gen_token_list)

    print("\n=== Generated REMI+ Event Tokens ===")
    for i, event in enumerate(gen_events):
        print(f"{i+1:03d}: {event}")

    if any(e == "<unk>" for e in gen_events):
        print("Warning: Generated sequence contains unknown tokens.")

    # === Save outputs ===
    with open(args.output_events, "w", encoding="utf-8") as f:
        for event in gen_events:
            f.write(event + "\n")
    print(f"Saved generated event sequence to: {args.output_events}")

    reconstruct_musicxml_from_events(gen_events, args.output_xml)
    print(f"Saved generated MusicXML to: {args.output_xml}")


if __name__ == "__main__":
    main()