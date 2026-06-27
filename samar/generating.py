# -*- coding: utf-8 -*-
"""
Inference: sample from a trained Transformer (round 18, no VQ-VAE).

Usage:
  python -m samar.generating
  python -m samar.generating --description-source path/to/piece.xml
  python -m samar.generating --temperature 0.8 --top-k 50
"""

import argparse
import json
import os
import sys

import torch

from .models.samar_transformer import SamarTransformer
from .tokenizer import SamarTokenizer, DescriptionTokenizer
from .reconstructor import reconstruct_musicxml_from_events
from .input_representation import SAMARInputRepresentation
from .dataset import compute_desc_bar_ids
from .constants import BOS_TOKEN, BAR_KEY

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_path(p):
    if os.path.isabs(p):
        return p
    return os.path.join(REPO_ROOT, p)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MusicXML from a trained Transformer.",
    )
    parser.add_argument(
        "--description-source",
        help="Path to an XML file whose description tokens should be used.",
    )
    parser.add_argument(
        "--description-index", type=int, default=0,
        help="Index into latents.pt for description. Default: 0.",
    )
    parser.add_argument(
        "--max-length", type=int, default=256,
        help="Max generation length. Default: 256.",
    )
    parser.add_argument(
        "--max-bars", type=int, default=-1,
        help="Stop after this many bars. -1 = no limit.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature. Default: 0.8.",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-k sampling. Default: 50.",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.0,
        help="Nucleus sampling threshold. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--output-xml", default="generated.xml",
        help="Output MusicXML path. Default: generated.xml.",
    )
    parser.add_argument(
        "--output-events", default="generated_events.txt",
        help="Output event list path. Default: generated_events.txt.",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/samar_transformer.pt",
        help="Transformer checkpoint path.",
    )
    parser.add_argument(
        "--latent-path", default="latents/latents.pt",
        help="Path to precomputed data (for description fallback).",
    )
    args = parser.parse_args()

    # === Load tokenizer + vocab ===
    tokenizer = SamarTokenizer.load(
        os.path.join(os.path.dirname(__file__), "samar_vocab.pkl"),
    )
    vocab = tokenizer.get_vocab()
    desc_tokenizer = DescriptionTokenizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load Transformer ===
    lm_ckpt = _resolve_path(args.checkpoint)
    config_path = _resolve_path(args.checkpoint.replace(".pt", "_config.json"))
    if not os.path.exists(config_path):
        config_path = _resolve_path("checkpoints/samar_transformer_config.json")
    with open(config_path) as f:
        lm_config = json.load(f)

    lm, load_report = SamarTransformer.from_pretrained(
        lm_ckpt, config=lm_config, device=str(device),
    )
    if load_report["missing"]:
        print(f"[load] missing layers: {load_report['missing']}")
    if load_report["unexpected"]:
        print(f"[load] unexpected keys: {load_report['unexpected']}")
    lm.to(device).eval()

    # === Get description ===
    if args.description_source:
        xml_path = _resolve_path(args.description_source)
        print(f"Loading description from: {xml_path}")
        ir = SAMARInputRepresentation(xml_path)
        desc_tokens = ir.get_description_tokens()
        desc_ids = desc_tokenizer.encode(desc_tokens)
        desc_bar_ids = compute_desc_bar_ids(desc_tokens)
    else:
        latent_path = _resolve_path(args.latent_path)
        latent_data = torch.load(latent_path, map_location=device, weights_only=False)
        if not latent_data:
            print("ERROR: latents.pt is empty. Run "
                  "`python -m samar.precompute_samar_latents` first.")
            sys.exit(1)
        idx = min(args.description_index, len(latent_data) - 1)
        sample = latent_data[idx]
        desc_ids = sample["description"]
        if isinstance(desc_ids, torch.Tensor):
            desc_ids = desc_ids.tolist()
        desc_bar_ids = sample.get("desc_bar_ids", [0] * len(desc_ids))
        if isinstance(desc_bar_ids, torch.Tensor):
            desc_bar_ids = desc_bar_ids.tolist()
        print(f"Using description from: {sample.get('file', '?')} (index {idx})")

    # Truncate description to model's max_bars
    max_bars = lm_config.get("max_bars", 512)
    if len(desc_ids) > max_bars:
        print(f"Truncating description {len(desc_ids)} -> {max_bars}")
        desc_ids = desc_ids[:max_bars]
        desc_bar_ids = desc_bar_ids[:max_bars]

    desc_tensor = torch.tensor([desc_ids], dtype=torch.long, device=device)
    desc_bar_tensor = torch.tensor([desc_bar_ids], dtype=torch.long, device=device)
    print(f"Description: {len(desc_ids)} tokens")

    # === Generate ===
    bos_id = vocab.to_i(BOS_TOKEN)
    start_tokens = torch.tensor([[bos_id]], device=device)

    with torch.no_grad():
        gen_token_ids = lm.sample(
            start_tokens=start_tokens,
            description=desc_tensor,
            desc_bar_ids=desc_bar_tensor,
            max_length=args.max_length,
            max_bars=args.max_bars,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            vocab=vocab,
        )

    # === Decode ===
    gen_token_list = gen_token_ids.squeeze().cpu().tolist()
    gen_events = vocab.decode(gen_token_list)

    # Strip BOS/EOS for display and reconstruction
    gen_events = [e for e in gen_events if e not in (BOS_TOKEN, '<eos>', '<pad>')]

    print(f"\n=== Generated REMI+ Event Tokens ({len(gen_events)}) ===")
    for i, event in enumerate(gen_events):
        print(f"{i+1:03d}: {event}")

    if any(e == "<unk>" for e in gen_events):
        print("Warning: Generated sequence contains unknown tokens.")

    # === Save ===
    with open(args.output_events, "w", encoding="utf-8") as f:
        for event in gen_events:
            f.write(event + "\n")
    print(f"Saved event sequence to: {args.output_events}")

    reconstruct_musicxml_from_events(gen_events, args.output_xml)
    print(f"Saved MusicXML to: {args.output_xml}")


if __name__ == "__main__":
    main()
