# -*- coding: utf-8 -*-
"""
PyTorch Datasets/DataLoaders for SAMAR.

Round 18: Added BOS/EOS wrapping, bar_ids, position_ids. Dropped VQ-VAE
latent support. Aligned with FIGARO's dataset.py patterns.
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .input_representation import SAMARInputRepresentation
from .tokenizer import SamarTokenizer, DescriptionTokenizer
from .constants import (
    UNK_TOKEN, BAR_KEY, POSITION_KEY, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
)

_TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "samar_vocab.pkl"
)


def _load_default_tokenizer():
    import sys as _sys
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in _sys.path:
        _sys.path.insert(0, pkg_dir)
    from . import vocab as _vocab_mod
    _sys.modules.setdefault("vocab", _vocab_mod)
    return SamarTokenizer.load(_TOKENIZER_PATH)


_tokenizer_singleton = None

def get_tokenizer():
    global _tokenizer_singleton
    if _tokenizer_singleton is None:
        _tokenizer_singleton = _load_default_tokenizer()
    return _tokenizer_singleton


def compute_structural_ids(events):
    """Compute bar_ids and position_ids from a list of event strings.

    Returns two lists of ints the same length as `events`.
    - bar_ids: cumulative bar counter (0 for BOS/EOS, increments on Bar_ tokens)
    - position_ids: within-bar position (from Position_ tokens, carried forward)

    Shared between dataset, precompute, and generation to avoid divergence.
    """
    bar_ids = []
    position_ids = []
    cur_bar = 0
    cur_pos = 0
    for ev in events:
        if ev == BOS_TOKEN:
            cur_bar = 0
            cur_pos = 0
        elif ev == EOS_TOKEN:
            pass  # keep current bar/pos
        elif ev.startswith(f"{BAR_KEY}_"):
            cur_bar += 1
            cur_pos = 0
        elif ev.startswith(f"{POSITION_KEY}_"):
            try:
                cur_pos = int(ev.split("_")[-1])
            except ValueError:
                pass
        bar_ids.append(cur_bar)
        position_ids.append(cur_pos)
    return bar_ids, position_ids


def compute_desc_bar_ids(desc_tokens):
    """Compute bar_ids for description tokens.

    Each Bar_N token increments the counter. Non-bar tokens inherit
    the current bar. Used for the encoder's bar_embedding.
    """
    bar_ids = []
    cur_bar = 0
    for tok in desc_tokens:
        if tok.startswith(f"{BAR_KEY}_"):
            cur_bar += 1
        bar_ids.append(cur_bar)
    return bar_ids


class SAMARDataset(Dataset):
    """Dataset that parses MusicXML files on the fly.

    Round 18: wraps event sequences with BOS/EOS, computes bar_ids and
    position_ids per token.
    """

    def __init__(self, data_dir, context_size=256, max_files=-1,
                 min_chunk_len=8, tokenizer=None):
        self.data_dir = data_dir
        self.context_size = context_size
        self.min_chunk_len = min_chunk_len
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.desc_tokenizer = DescriptionTokenizer()
        self.vocab = self.tokenizer.get_vocab()

        print(f"Loading MusicXML files from: {data_dir}")
        xml_files = sorted(glob.glob(os.path.join(data_dir, "**/*.xml"), recursive=True))
        mxl_files = sorted(glob.glob(os.path.join(data_dir, "**/*.mxl"), recursive=True))
        self.files = xml_files + mxl_files
        if max_files > 0:
            self.files = self.files[:max_files]

        print(f"Found {len(self.files)} MusicXML files "
              f"({len(xml_files)} .xml + {len(mxl_files)} .mxl)")
        self.examples = []
        for file in self.files:
            try:
                ir = SAMARInputRepresentation(file)
                raw_events = ir.events
                description = ir.get_description_tokens()

                # Wrap with BOS/EOS
                wrapped = [BOS_TOKEN] + raw_events + [EOS_TOKEN]
                event_ids = self.tokenizer.encode(wrapped)
                bar_ids, position_ids = compute_structural_ids(wrapped)
                desc_ids = self.desc_tokenizer.encode(description)
                desc_bar_ids = compute_desc_bar_ids(description)

                bos_id = self.vocab.to_i(BOS_TOKEN)
                eos_id = self.vocab.to_i(EOS_TOKEN)

                # Chunk on bar boundaries (FIGARO pattern).
                inner_ids = event_ids[1:-1]
                inner_bar = bar_ids[1:-1]
                inner_pos = position_ids[1:-1]

                bar_prefix = f"{BAR_KEY}_"
                bar_starts = [i for i, ev in enumerate(raw_events)
                              if ev.startswith(bar_prefix)]
                bar_starts.append(len(inner_ids))

                chunk_start = 0
                for bi in range(len(bar_starts)):
                    end = bar_starts[bi]
                    chunk_len = end - chunk_start
                    is_last = (bi == len(bar_starts) - 1)

                    if chunk_len >= context_size - 2 or is_last:
                        if is_last:
                            end = len(inner_ids)
                        c_ids = inner_ids[chunk_start:end][:context_size - 2]
                        c_bar = inner_bar[chunk_start:end][:context_size - 2]
                        c_pos = inner_pos[chunk_start:end][:context_size - 2]
                        chunk_start = end

                        if len(c_ids) < self.min_chunk_len:
                            continue
                        c_ids = [bos_id] + c_ids + [eos_id]
                        c_bar = [0] + c_bar + [0]
                        c_pos = [0] + c_pos + [0]

                        self.examples.append({
                            "tokens": c_ids,
                            "bar_ids": c_bar,
                            "position_ids": c_pos,
                            "description": desc_ids,
                            "desc_bar_ids": desc_bar_ids,
                            "file": file,
                        })
            except Exception as e:
                print(f"Failed to process {file}: {e}")

        print(f"Total token chunks prepared: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        tokens = item["tokens"]
        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long),
            "bar_ids": torch.tensor(item["bar_ids"][:-1], dtype=torch.long),
            "position_ids": torch.tensor(item["position_ids"][:-1], dtype=torch.long),
            "description": torch.tensor(item["description"], dtype=torch.long),
            "desc_bar_ids": torch.tensor(item["desc_bar_ids"], dtype=torch.long),
            "file": os.path.basename(item["file"]),
        }


class SamarLatentDataset(Dataset):
    """Reads precomputed data (round-18 format: no VQ-VAE latent).

    Each sample has tokens, bar_ids, position_ids, description,
    desc_bar_ids.
    """

    def __init__(self, latent_path, context_size=256, tokenizer=None,
                 min_keep_len=None):
        self.context_size = context_size
        self.samples = torch.load(latent_path, weights_only=False)
        self.tokenizer = tokenizer or SamarTokenizer.load(_TOKENIZER_PATH)
        self.pad_id = self.tokenizer.get_vocab().pad_id

        if min_keep_len is None:
            min_keep_len = 16
        n_before = len(self.samples)
        self.samples = [s for s in self.samples if len(s["tokens"]) >= min_keep_len]
        n_after = len(self.samples)
        if n_before != n_after:
            print(f"[SamarLatentDataset] Filtered {n_before - n_after} samples "
                  f"shorter than {min_keep_len} tokens "
                  f"({n_before} -> {n_after} samples)")

        # Warn about unk ratio
        unk_id = self.tokenizer.get_vocab().to_i(UNK_TOKEN)
        total_tokens = sum(len(s["tokens"]) for s in self.samples)
        unk_tokens = sum(1 for s in self.samples
                         for t in s["tokens"] if t == unk_id)
        if total_tokens > 0:
            unk_ratio = unk_tokens / total_tokens
            if unk_ratio > 0.05:
                print(f"[SamarLatentDataset] WARNING: {unk_ratio:.1%} <unk> tokens "
                      f"in {latent_path}. Regenerate with "
                      f"`python -m samar.precompute_samar_latents`.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = list(item["tokens"][:self.context_size])
        bar_ids = list(item["bar_ids"][:self.context_size])
        position_ids = list(item["position_ids"][:self.context_size])

        # Pad to context_size
        pad_len = self.context_size - len(tokens)
        if pad_len > 0:
            tokens = tokens + [self.pad_id] * pad_len
            bar_ids = bar_ids + [0] * pad_len
            position_ids = position_ids + [0] * pad_len

        result = {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),
            "labels": torch.tensor(tokens[1:], dtype=torch.long),
            "bar_ids": torch.tensor(bar_ids[:-1], dtype=torch.long),
            "position_ids": torch.tensor(position_ids[:-1], dtype=torch.long),
        }

        # Cap description to 256 tokens to prevent OOM in encoder attention
        max_desc_len = self.context_size

        if "description" in item:
            desc = item["description"]
            if isinstance(desc, torch.Tensor):
                result["description"] = desc[:max_desc_len].long()
            elif isinstance(desc, list):
                result["description"] = torch.tensor(desc[:max_desc_len], dtype=torch.long)
            else:
                result["description"] = torch.tensor(desc, dtype=torch.long)[:max_desc_len]

        if "desc_bar_ids" in item:
            dbi = item["desc_bar_ids"]
            if isinstance(dbi, torch.Tensor):
                result["desc_bar_ids"] = dbi[:max_desc_len].long()
            elif isinstance(dbi, list):
                result["desc_bar_ids"] = torch.tensor(dbi[:max_desc_len], dtype=torch.long)
            else:
                result["desc_bar_ids"] = torch.tensor(dbi, dtype=torch.long)[:max_desc_len]

        return result


_DESCRIPTION_TOKENIZER = None

def _get_desc_tokenizer():
    global _DESCRIPTION_TOKENIZER
    if _DESCRIPTION_TOKENIZER is None:
        _DESCRIPTION_TOKENIZER = DescriptionTokenizer()
    return _DESCRIPTION_TOKENIZER


def samar_collate_fn(batch):
    """Collate function: pads input_ids, labels, bar_ids, position_ids,
    description, desc_bar_ids."""

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True, padding_value=0
    )
    labels = pad_sequence(
        [item["labels"] for item in batch],
        batch_first=True, padding_value=0
    )
    bar_ids = pad_sequence(
        [item["bar_ids"] for item in batch],
        batch_first=True, padding_value=0
    )
    position_ids = pad_sequence(
        [item["position_ids"] for item in batch],
        batch_first=True, padding_value=0
    )

    batch_dict = {
        "input_ids": input_ids,
        "labels": labels,
        "bar_ids": bar_ids,
        "position_ids": position_ids,
    }

    if "description" in batch[0]:
        desc_tok = _get_desc_tokenizer()
        desc_ids = []
        for item in batch:
            d = item["description"]
            if isinstance(d, torch.Tensor):
                desc_ids.append(d.long())
            elif isinstance(d, (list, tuple)) and len(d) > 0 and isinstance(d[0], int):
                desc_ids.append(torch.tensor(d, dtype=torch.long))
            else:
                desc_ids.append(
                    torch.tensor(desc_tok.encode(d), dtype=torch.long)
                )
        batch_dict["description"] = pad_sequence(
            desc_ids, batch_first=True,
            padding_value=desc_tok.get_vocab().pad_id,
        )

    if "desc_bar_ids" in batch[0]:
        dbi_list = []
        for item in batch:
            dbi = item["desc_bar_ids"]
            if isinstance(dbi, torch.Tensor):
                dbi_list.append(dbi.long())
            else:
                dbi_list.append(torch.tensor(dbi, dtype=torch.long))
        batch_dict["desc_bar_ids"] = pad_sequence(
            dbi_list, batch_first=True, padding_value=0,
        )

    if "file" in batch[0]:
        batch_dict["file"] = [item.get("file", "") for item in batch]

    return batch_dict


def get_samar_dataloader(data_dir, batch_size=16, context_size=256,
                         max_files=-1, num_workers=0, min_chunk_len=8,
                         drop_last=False):
    dataset = SAMARDataset(
        data_dir=data_dir,
        context_size=context_size,
        max_files=max_files,
        min_chunk_len=min_chunk_len,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=samar_collate_fn,
    )
