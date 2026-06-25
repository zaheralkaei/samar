"""Description utility for the SAMAR project.

This script is the user-facing entry point for generating and inspecting
description tokens for MusicXML pieces. It supports three operations:

  1. EXTRACT: read MusicXML files and emit per-bar description strings
  2. ENCODE: convert description strings into description-vocab IDs
     (this is what the transformer consumes)
  3. SHOW STATS: per-corpus distribution of description tokens (useful
     for understanding what the model can condition on)

Run examples (from project root):

  # Extract descriptions for every file in data/xml/
  python -m samar.describe extract data/xml/

  # Extract descriptions for a single file
  python -m samar.describe extract data/xml/fairuz_ya_bia_alkhwatem_ajam_1964.xml

  # Encode a description into the model's token IDs
  python -m samar.describe encode "Bar_1,TimeSignature_4/4,NoteDensity_3,MeanVelocity_16"

  # Show description-token distribution across the whole corpus
  python -m samar.describe stats data/xml/

This script does NOT modify any model or checkpoint. It only reads.
"""

import argparse
import json
import os
import sys
from collections import Counter

# Allow running from project root without installing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samar.input_representation import SAMARInputRepresentation
from samar.tokenizer import DescriptionTokenizer
from samar.vocab import DescriptionVocab


def _iter_xml_files(path):
    """Yield (label, filepath) tuples from a file or directory.

    If ``path`` is a directory, walks it recursively and yields every
    ``*.xml`` AND ``*.mxl`` file (compressed MusicXML 4.0). If it's
    a file, yields it once.
    """
    valid_exts = (".xml", ".mxl")
    if os.path.isfile(path) and path.lower().endswith(valid_exts):
        yield os.path.basename(path), path
        return
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for fn in sorted(files):
                if fn.lower().endswith(valid_exts):
                    yield fn, os.path.join(root, fn)


def cmd_extract(args):
    """Extract per-bar description tokens for one or more XML files."""
    pieces = []
    for label, path in _iter_xml_files(args.path):
        try:
            ir = SAMARInputRepresentation(path)
            desc = ir.get_description_tokens()
        except Exception as e:
            print(f"  [skip] {label}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        pieces.append({"file": label, "description": desc})

    if args.format == "json":
        print(json.dumps(pieces, indent=2))
    else:
        # Human-readable: one block per piece
        for piece in pieces:
            print(f"=== {piece['file']} ({len(piece['description'])} desc tokens) ===")
            # Print first 24 tokens, then truncate
            tokens = piece["description"]
            shown = tokens[: args.show]
            print("  " + " ".join(shown))
            if len(tokens) > args.show:
                print(f"  ... ({len(tokens) - args.show} more)")

    print(f"\nTotal pieces: {len(pieces)}", file=sys.stderr)


def cmd_encode(args):
    """Encode a description string into description-vocab IDs.

    Accepts a comma-separated list of tokens (e.g.
    ``"Bar_1,TimeSignature_4/4,NoteDensity_3"``) on the command line,
    or a JSON file containing a list of description-token lists.
    """
    dt = DescriptionTokenizer()
    vocab = dt.get_vocab()

    if args.file:
            # JSON file: list of {"file": x, "description": [...]} OR
            # dict {file: [...]} OR dict {file: {"description": [...]}}.
            with open(args.file) as f:
                data = json.load(f)
            results = {}
            if isinstance(data, list):
                # List format (from `extract --format json`)
                for entry in data:
                    file_label = entry["file"]
                    tokens = entry["description"]
                    results[file_label] = dt.encode(tokens)
            else:
                # Dict format
                for file_label, payload in data.items():
                    tokens = payload["description"] if isinstance(payload, dict) else payload
                    results[file_label] = dt.encode(tokens)
            print(json.dumps(results, indent=2))
            return

    # Inline comma-separated tokens
    tokens = [t.strip() for t in args.tokens.split(",") if t.strip()]
    if not tokens:
        print("ERROR: no tokens provided", file=sys.stderr)
        sys.exit(1)

    ids = dt.encode(tokens)
    print(f"Tokens ({len(tokens)}):")
    for t, i in zip(tokens, ids):
        marker = " " if i != 1 else " <-- <unk>!"
        print(f"  {t:30s} -> {i}{marker}")
    print(f"\nEncoded IDs: {ids}")


def cmd_stats(args):
    """Show per-corpus distribution of description tokens."""
    dt = DescriptionTokenizer()
    dv = dt.get_vocab()

    # Per-category subcounters (must be defined before category_files
    # because the dict-comprehension references it).
    sub_counters = {
        "Bar": Counter(),
        "TimeSignature": Counter(),
        "NoteDensity": Counter(),
        "MeanVelocity": Counter(),
        "MeanPitch": Counter(),
        "MeanDuration": Counter(),
        "Instrument": Counter(),
    }

    # Counter per token category
    category_counter = Counter()
    category_files = {k: set() for k in sub_counters}
    category_files["other"] = set()
    pieces_seen = 0
    pieces_failed = 0
    total_desc_tokens = 0

    for label, path in _iter_xml_files(args.path):
        try:
            ir = SAMARInputRepresentation(path)
            desc = ir.get_description_tokens()
        except Exception:
            pieces_failed += 1
            continue

        pieces_seen += 1
        total_desc_tokens += len(desc)

        for tok in desc:
            # Split on '_' but keep multi-word numbers together
            matched = False
            for prefix in sub_counters:
                if tok.startswith(prefix + "_"):
                    category_counter[prefix] += 1
                    category_files[prefix].add(label)
                    sub_counters[prefix][tok] += 1
                    matched = True
                    break
            if not matched:
                category_counter["other"] += 1
                category_files["other"].add(label)

    print(f"Pieces parsed: {pieces_seen} (failed: {pieces_failed})")
    print(f"Total description tokens: {total_desc_tokens}")
    if pieces_seen > 0:
        print(f"Avg description tokens / piece: "
              f"{total_desc_tokens / pieces_seen:.1f}")

    print(f"\nDescription token categories:")
    print(f"  {'category':<20s} {'tokens':>8s} {'files':>8s} {'vocab_size':>12s}")
    for cat in sorted(category_counter, key=lambda c: -category_counter[c]):
            n_unique = len(sub_counters.get(cat, {}))
            n_files = len(category_files.get(cat, set()))
            print(f"  {cat:<20s} {category_counter[cat]:>8d} "
                  f"{n_files:>8d} {n_unique:>12d}")

    # Top-K per category
    print(f"\nTop values per category:")
    for cat, counter in sub_counters.items():
        if not counter:
            continue
        top = counter.most_common(5)
        print(f"  {cat}:")
        for tok, n in top:
            print(f"    {tok:<30s} {n:>5d}")

    # File-extension breakdown (round-4: supports both .xml and .mxl)
    print(f"\nFile format breakdown:")
    ext_counter = Counter()
    for label in (n for n, _ in _iter_xml_files(args.path)):
        if "." in label:
            ext_counter[label.rsplit(".", 1)[-1].lower()] += 1
    for ext, n in ext_counter.most_common():
        print(f"  .{ext}: {n} files")

    # Maqam + singer distribution (best-effort from filenames)
    print(f"\nMaqam tags (best-effort from filename):")
    maqam_keywords = [
        "bayat", "rast", "saba", "hijaz", "kurd", "ajam", "nahawand",
        "sikah", "maqam", "saz", "huzam", "awj", "iraq",
    ]
    maqam_counter = Counter()
    for label in (n for n, _ in _iter_xml_files(args.path)):
        lower = label.lower()
        for m in maqam_keywords:
            if m in lower:
                maqam_counter[m] += 1
    for m, n in maqam_counter.most_common():
        print(f"  {m}: {n} files")

    print(f"\nSinger distribution (top 10):")
    singer_counter = Counter()
    for label in (n for n, _ in _iter_xml_files(args.path)):
        singer = label.split("_")[0]
        singer_counter[singer] += 1
    for s, n in singer_counter.most_common(10):
        print(f"  {s}: {n} files")


def main():
    parser = argparse.ArgumentParser(
        description="Description-token utilities for SAMAR pieces.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # extract
    p_extract = sub.add_parser(
        "extract",
        help="Extract per-bar description tokens from MusicXML files.",
    )
    p_extract.add_argument(
        "path",
        help="Either a single .xml file or a directory (walks recursively).",
    )
    p_extract.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format. Default: text.",
    )
    p_extract.add_argument(
        "--show", type=int, default=60,
        help="How many tokens to show per piece in text mode. Default: 60.",
    )
    p_extract.set_defaults(func=cmd_extract)

    # encode
    p_encode = sub.add_parser(
        "encode",
        help="Encode description tokens into description-vocab IDs.",
    )
    p_encode.add_argument(
        "tokens", nargs="?",
        help="Comma-separated tokens (e.g. 'Bar_1,TimeSignature_4/4,NoteDensity_3').",
    )
    p_encode.add_argument(
        "--file",
        help="JSON file produced by `describe extract --format json`.",
    )
    p_encode.set_defaults(func=cmd_encode)

    # stats
    p_stats = sub.add_parser(
        "stats",
        help="Show description-token distribution across a corpus.",
    )
    p_stats.add_argument(
        "path",
        help="Either a single .xml file or a directory (walks recursively).",
    )
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()