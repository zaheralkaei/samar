# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:22:38 2025

@author: zaher
"""

# samar/vocab.py

import os
import json
from collections import Counter, defaultdict

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<MASK>"]

class VocabBuilder:
    def __init__(self):
        self.counter = Counter()
        self.token2idx = {}
        self.idx2token = {}
        self.token_groups = defaultdict(set)

    def add_tokens_from_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    self.counter[token] += 1
                    group = token.split("_")[0]
                    self.token_groups[group].add(token)

    def build_vocab(self, min_freq=1):
        idx = 0
        for tok in SPECIAL_TOKENS:
            self.token2idx[tok] = idx
            idx += 1
        for token, count in self.counter.items():
            if count >= min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                idx += 1
        self.idx2token = {i: t for t, i in self.token2idx.items()}

    def save(self, path):
        out = {
            "token2idx": self.token2idx,
            "idx2token": self.idx2token,
            "token_groups": {k: list(v) for k, v in self.token_groups.items()}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"✅ Saved vocab with {len(self.token2idx)} tokens to {path}")
