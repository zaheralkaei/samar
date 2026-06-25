# -*- coding: utf-8 -*-
"""
Created on Sat May 10 03:56:55 2025

@author: zaher
"""

# build_vocab.py
# Dynamically builds a vocabulary from all tokens found in your MusicXML dataset

import os
import pickle
from collections import Counter
from core import SAMARInputRepresentation
from vocab_and_tokenizer import Vocab

xml_dir = "xml_data"
out_path = "samar_vocab.pkl"

print("🔍 Scanning MusicXML files to build vocabulary...")
all_tokens = []

for fname in os.listdir(xml_dir):
    if not fname.endswith(".xml"):
        continue
    path = os.path.join(xml_dir, fname)
    try:
        rep = SAMARInputRepresentation(path)
        events = rep.get_event_sequence()
        desc = rep.get_description_tokens()
        all_tokens.extend(events + desc)
    except Exception as e:
        print(f"❗ Failed to process {fname}: {e}")

print(f"✅ Collected {len(all_tokens)} total tokens from {xml_dir}.")

# Build vocabulary
counter = Counter(all_tokens)
vocab = Vocab(counter)

# Save
with open(out_path, "wb") as f:
    pickle.dump(vocab, f)

print(f"📦 Vocabulary saved to: {out_path}")
print(f"🔠 Vocab size: {len(vocab)}")
print("Example tokens:", list(vocab.stoi.items())[:10])
