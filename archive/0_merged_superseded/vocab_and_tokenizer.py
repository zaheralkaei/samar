# -*- coding: utf-8 -*-
"""
Created on Sat May 10 03:06:08 2025

@author: zaher
"""

# vocab_and_tokenizer.py
# Combines vocab.py and tokenizer.py

import pickle
from collections import Counter
from constants import (
    DEFAULT_VELOCITY_BINS,
    DEFAULT_DURATION_BINS,
    DEFAULT_TEMPO_BINS,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_NOTE_DENSITY_BINS,
    DEFAULT_MEAN_VELOCITY_BINS,
    DEFAULT_MEAN_PITCH_BINS,
    DEFAULT_MEAN_DURATION_BINS,
    MAX_BAR_LENGTH,
    MAX_N_BARS,
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    MASK_TOKEN,
    TIME_SIGNATURE_KEY,
    BAR_KEY,
    POSITION_KEY,
    INSTRUMENT_KEY,
    PITCH_KEY,
    VELOCITY_KEY,
    DURATION_KEY,
    TEMPO_KEY,
    CHORD_KEY,
    NOTE_DENSITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_DURATION_KEY,
)

class Tokens:
    def get_instrument_tokens(key=INSTRUMENT_KEY):
        common_instrs = ["Violin", "Violoncello", "Flute", "Clarinet", "Trumpet", "Oud", "Qanun", "Nay", "Percussion", "Voice", "Piano"]
        return [f'{key}_{name}' for name in common_instrs] + [f'{key}_Unknown']

    def get_chord_tokens(key=CHORD_KEY, qualities=None):
        if qualities is None:
            qualities = ['maj', 'min', 'dim', 'aug', 'dom7', 'maj7', 'min7', 'None']
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chords = [f'{root}:{quality}' for root in pitch_classes for quality in qualities]
        chords.append('N:N')
        return [f'{key}_{chord}' for chord in chords]

    def get_time_signature_tokens(key=TIME_SIGNATURE_KEY):
        denominators = [2, 4, 8, 16]
        time_sigs = [f'{p}/{q}' for q in denominators for p in range(1, MAX_BAR_LENGTH * q + 1)]
        return [f'{key}_{sig}' for sig in time_sigs]

    def get_midi_tokens_samar(
        instrument_key=INSTRUMENT_KEY,
        time_signature_key=TIME_SIGNATURE_KEY,
        pitch_key=PITCH_KEY,
        velocity_key=VELOCITY_KEY,
        duration_key=DURATION_KEY,
        tempo_key=TEMPO_KEY,
        bar_key=BAR_KEY,
        position_key=POSITION_KEY
    ):
        instrument_tokens = Tokens.get_instrument_tokens(instrument_key)
        pitch_tokens = [f'{pitch_key}_{i}' for i in range(24 * 6)] + [f'{pitch_key}_Rest']
        velocity_tokens = [f'{velocity_key}_{i}' for i in range(len(DEFAULT_VELOCITY_BINS))]
        duration_tokens = [f'{duration_key}_{i}' for i in range(len(DEFAULT_DURATION_BINS))]
        tempo_tokens = [f'{tempo_key}_{i}' for i in range(len(DEFAULT_TEMPO_BINS))]
        bar_tokens = [f'{bar_key}_{i}' for i in range(MAX_N_BARS)]
        position_tokens = [f'{position_key}_{i}' for i in range(MAX_BAR_LENGTH * 4 * DEFAULT_POS_PER_QUARTER)]
        time_sig_tokens = Tokens.get_time_signature_tokens(time_signature_key)

        return (
            time_sig_tokens + tempo_tokens + instrument_tokens +
            pitch_tokens + velocity_tokens + duration_tokens +
            bar_tokens + position_tokens
        )

class Vocab:
    def __init__(self, counter, specials=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN], unk_token=UNK_TOKEN):
        self.unk_token = unk_token
        self.specials = specials
        all_tokens = list(specials) + sorted(counter.keys())
        self.stoi = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.itos = {idx: tok for tok, idx in self.stoi.items()}
        self.pad_token = PAD_TOKEN
        self.pad_id = self.stoi[self.pad_token]

    def to_i(self, token):
        return self.stoi.get(token, self.stoi[self.unk_token])

    def to_s(self, idx):
        return self.itos.get(idx, self.unk_token)

    def __len__(self):
        return len(self.stoi)

    def encode(self, seq):
        return [self.to_i(tok) for tok in seq]

    def decode(self, seq):
        return [self.to_s(idx) for idx in seq]

class DescriptionVocab(Vocab):
    def __init__(self):
        time_sig_tokens = Tokens.get_time_signature_tokens()
        instrument_tokens = Tokens.get_instrument_tokens()
        chord_tokens = Tokens.get_chord_tokens()
        bar_tokens = [f'Bar_{i}' for i in range(MAX_N_BARS)]
        density_tokens = [f'{NOTE_DENSITY_KEY}_{i}' for i in range(len(DEFAULT_NOTE_DENSITY_BINS))]
        velocity_tokens = [f'{MEAN_VELOCITY_KEY}_{i}' for i in range(len(DEFAULT_MEAN_VELOCITY_BINS))]
        pitch_tokens = [f'{MEAN_PITCH_KEY}_{i}' for i in range(len(DEFAULT_MEAN_PITCH_BINS))]
        duration_tokens = [f'{MEAN_DURATION_KEY}_{i}' for i in range(len(DEFAULT_MEAN_DURATION_BINS))]

        self.tokens = (
            time_sig_tokens + instrument_tokens + chord_tokens +
            density_tokens + velocity_tokens + pitch_tokens + duration_tokens + bar_tokens
        )
        counter = Counter(self.tokens)
        super().__init__(counter)

class SamarVocab(Vocab):
    def __init__(self):
        midi_tokens = Tokens.get_midi_tokens_samar()
        chord_tokens = Tokens.get_chord_tokens()
        description_vocab = DescriptionVocab()
        self.tokens = midi_tokens + chord_tokens + description_vocab.tokens
        counter = Counter(self.tokens)
        super().__init__(counter)

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
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        return SamarTokenizer(vocab)

    def get_vocab(self):
        return self.vocab
