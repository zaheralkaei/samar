# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:54:36 2025

@author: zaher
"""

# === File: samar/dataset.py ===
# PyTorch Datasets/DataLoaders for raw & latent sequences

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from .input_representation import SAMARInputRepresentation
from .tokenizer import SamarTokenizer
# Load the default vocab pickle sitting next to this package's __init__.py.
# NOTE: the original pickle was built when the codebase was flat (vocab.py at
# the project root) and is not yet regenerated. Loading it at import time
# would re-trigger a stale relative import. We expose a lazy ``tokenizer``
# attribute that loads on first access and caches the result.
import os as _os, pickle as _pickle
_TOKENIZER_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "samar_vocab.pkl")

def _load_default_tokenizer():
    # Load the pickle in a controlled namespace so its stale ``vocab`` module
    # reference resolves to the current samar.vocab class.
    import sys as _sys, types as _types
    pkg_dir = _os.path.dirname(_os.path.abspath(__file__))
    if pkg_dir not in _sys.path:
        _sys.path.insert(0, pkg_dir)
    # Pre-import and alias so pickle finds the class under its old module name
    from . import vocab as _vocab_mod
    _sys.modules.setdefault("vocab", _vocab_mod)
    return SamarTokenizer.load(_TOKENIZER_PATH)

tokenizer = None  # populated on first access via ``get_tokenizer()``

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = _load_default_tokenizer()
    return tokenizer

# Dataset class for processing SAMAR MusicXML files into token sequences
class SAMARDataset(Dataset):
    def __init__(self, data_dir, context_size=256, max_files=-1, min_chunk_len=8, tokenizer=None):
        self.data_dir = data_dir
        self.context_size = context_size
        self.min_chunk_len = min_chunk_len
        # If a tokenizer is provided use it directly; otherwise fall back to the
        # lazy ``get_tokenizer()`` helper which loads the default vocab pickle
        # (legacy samar_vocab.pkl living next to this module).
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

        # Load all .xml files recursively from the given data directory
        print(f"Loading XML files from: {data_dir}")
        self.files = sorted(glob.glob(os.path.join(data_dir, "**/*.xml"), recursive=True))
        if max_files > 0:
            self.files = self.files[:max_files]

        print(f"Found {len(self.files)} MusicXML files")
        self.examples = []
        for file in self.files:
            try:
                # Convert XML file into a sequence of events, then tokenize
                ir = SAMARInputRepresentation(file)
                events = ir.get_event_sequence()
                description = ir.get_description_tokens()  # extract description
                token_ids = self.tokenizer.encode(events)
                self.examples.extend([{
                    "tokens": token_ids[i:i + context_size],
                    "file": file,
                    "description": description
                    } for i in range(0, len(token_ids), context_size) if len(token_ids[i:i + context_size]) >= self.min_chunk_len])
            except Exception as e:
                # Log failed file processing
                print(f"Failed to process {file}: {e}")

        print(f"Total token chunks prepared: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        return {
        'input_ids': torch.tensor(item["tokens"][:-1], dtype=torch.long),
        'labels': torch.tensor(item["tokens"][1:], dtype=torch.long),
        'file': os.path.basename(item["file"]),
        'description': item["description"]
        }

# Helper function to return a DataLoader from SAMARDataset
def get_samar_dataloader(data_dir, batch_size=16, context_size=256, max_files=-1, num_workers=0, min_chunk_len=8, drop_last=False):
    dataset = SAMARDataset(
        data_dir=data_dir,
        context_size=context_size,
        max_files=max_files,
        min_chunk_len=min_chunk_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=samar_collate_fn  # custom collation for batching
    )

# Dataset class for working with precomputed latent representations
class SamarLatentDataset(Dataset):
    def __init__(self, latent_path, context_size=256, tokenizer=None):
        self.context_size = context_size
        self.samples = torch.load(latent_path)
        self.tokenizer = tokenizer or SamarTokenizer.load(_TOKENIZER_PATH)
        # Keep only samples that meet the minimum length requirement
        self.samples = [
            s for s in self.samples if len(s["tokens"]) >= context_size
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = item["tokens"][:self.context_size]
        latent = item["latent"]
        return {
            "input_ids": torch.tensor(tokens[:-1], dtype=torch.long),  # input sequence
            "labels": torch.tensor(tokens[1:], dtype=torch.long),      # target sequence
            "latent": torch.tensor(latent, dtype=torch.float)          # latent vector
        }

from torch.nn.utils.rnn import pad_sequence

# Collate function to pad sequences in a batch for training
def samar_collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad input and label sequences to the longest sequence in the batch
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)

    batch_dict = {
        'input_ids': input_ids_padded,
        'labels': labels_padded,
    }

    # If latent vectors are included in the dataset, pad and include them
    if 'latent' in batch[0]:
        latent = [item['latent'] for item in batch]
        latent_padded = pad_sequence(latent, batch_first=True, padding_value=0)
        batch_dict['latent'] = latent_padded

    
    if 'description' in batch[0]:
        desc_lists = [item['description'] for item in batch]
        desc_ids = [torch.tensor(get_tokenizer().encode(desc), dtype=torch.long) for desc in desc_lists]
        desc_padded = pad_sequence(desc_ids, batch_first=True, padding_value=get_tokenizer().get_vocab().pad_id)
        batch_dict['description'] = desc_padded
    return batch_dict