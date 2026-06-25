# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:23:44 2025

@author: zaher
"""

# build_vocab_and_tokenize.py

import os
from vocab import VocabBuilder
from tokenizer import load_vocab, tokenize_remi_file

def build_vocab(remi_folder, vocab_path):
    vb = VocabBuilder()
    for fname in os.listdir(remi_folder):
        vb.add_tokens_from_file(os.path.join(remi_folder, fname))
    vb.build_vocab()
    vb.save(vocab_path)

def tokenize_all(remi_folder, vocab_path, out_folder):
    token2idx, _, _ = load_vocab(vocab_path)
    os.makedirs(out_folder, exist_ok=True)

    for fname in os.listdir(remi_folder):
        if fname.endswith(".txt"):
            ids = tokenize_remi_file(os.path.join(remi_folder, fname), token2idx)
            out_file = os.path.join(out_folder, fname.replace(".txt", ".ids"))
            with open(out_file, "w") as f:
                f.write(" ".join(map(str, ids)))
            print(f"✅ Tokenized {fname} → {out_file}")

if __name__ == "__main__":
    build_vocab("data/remi_txt/", "data/samar_vocab.json")
    tokenize_all("data/remi_txt/", "data/samar_vocab.json", "data/tokenized/")
