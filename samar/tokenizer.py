# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:56:39 2025

@author: zaher
"""

# === File: samar/tokenizer.py ===
# Wraps Vocab classes for encode/decode. Two tokenizer types match FIGARO's
# split between RemiVocab and DescriptionVocab (see figaro/src/vocab.py):
#
#   * SamarTokenizer    -- encodes the event stream (Pitch_24EDO_N,
#                          Bar_N, Position_N, Velocity_N, ...). Backed by
#                          a pickled SamarVocab.
#   * DescriptionTokenizer -- encodes the per-piece description tokens
#                          (composer credits, Mean Pitch, Note Density, ...).
#                          Rebuilt from scratch each session because the
#                          description vocab is large (~800 tokens) and
#                          deterministic.

import pickle
from .vocab import SamarVocab, DescriptionVocab


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


class DescriptionTokenizer:
    """Encode description tokens (composer credits, mean pitch, ...) by ID.

    This is a thin wrapper around ``DescriptionVocab`` -- it has no pickle
    to load because the description vocab is fully deterministic from the
    constants in :mod:`samar.constants` and recreating it is cheaper than
    parsing a ~100KB pickle.
    """

    def __init__(self, vocab=None):
        self.vocab = vocab or DescriptionVocab()

    def encode(self, description_tokens):
        return self.vocab.encode(description_tokens)

    def decode(self, indices):
        return self.vocab.decode(indices)

    def get_vocab(self):
        return self.vocab