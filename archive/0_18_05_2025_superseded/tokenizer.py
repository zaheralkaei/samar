# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:23:11 2025

@author: zaher
"""

# samar/tokenizer.py

import json

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["token2idx"], data["idx2token"], data["token_groups"]

def tokenize_remi_file(file_path, token2idx):
    with open(file_path, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return [token2idx.get("<BOS>", 1)] + [token2idx.get(t, token2idx["<UNK>"]) for t in tokens] + [token2idx.get("<EOS>", 2)]
