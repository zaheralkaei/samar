# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:56:39 2025

@author: zaher
"""

# === File: samar/tokenizer.py ===
# Wraps Vocab for encode/decode

import pickle
from .vocab import SamarVocab

class SamarTokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or SamarVocab()

    def encode(self, events):
        return self.vocab.encode(events)

    def decode(self, indices):
        return self.vocab.decode(indices)

    def save(self, path="samar_vocab.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.vocab, f)

    @staticmethod
    def load(path="samar_vocab.pkl"):
        # Older pickles reference the pre-reorg ``vocab`` module by name. Make
        # sure the current ``samar.vocab`` is also findable under that name
        # before unpickling so legacy artifacts still load.
        import sys as _sys
        from . import vocab as _vocab_mod
        _sys.modules.setdefault("vocab", _vocab_mod)
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        return SamarTokenizer(vocab)

    def get_vocab(self):
        return self.vocab
