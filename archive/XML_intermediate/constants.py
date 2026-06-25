# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 20:11:50 2025

@author: zaher
"""

# === File: samar/constants.py ===
# Global constants & bin definitions

import numpy as np

# === Default Parameters for Input Representation ===
DEFAULT_POS_PER_QUARTER = 12
DEFAULT_RESOLUTION = 480

DEFAULT_DURATION_BINS = np.sort(np.concatenate([
    np.arange(1, 13),  # smallest possible units up to 1 quarter
    np.arange(12, 24, 3)[1:],  # 16th notes up to 1 bar
    np.arange(13, 24, 4)[1:],  # triplets up to 1 bar
    np.arange(24, 48, 6),      # 8th notes up to 2 bars
    np.arange(48, 4*48, 12),   # quarter notes up to 8 bars
    np.arange(4*48, 16*48+1, 24)  # half notes up to 16 bars
]))

DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 33, dtype=int)
DEFAULT_TEMPO_BINS = np.linspace(30, 240, 33, dtype=int)
DEFAULT_NOTE_DENSITY_BINS = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0,
    1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0
])
DEFAULT_MEAN_VELOCITY_BINS = np.linspace(0, 128, 33)
DEFAULT_MEAN_PITCH_BINS = np.linspace(0, 144, 33)
DEFAULT_MEAN_DURATION_BINS = np.logspace(0, 7, 33, base=2)
DEFAULT_VOLUME_BINS = np.linspace(0, 127, 64)

# === Output & Sequence Parameters ===
MAX_BAR_LENGTH = 3
MAX_N_BARS = 512

# === Special Tokens ===
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
MASK_TOKEN = '<mask>'

# === Vocabulary/Event Types ===
TIME_SIGNATURE_KEY = 'TimeSignature'
KEY_SIGNATURE_KEY = 'KeySignature'
BAR_KEY = 'Bar'
POSITION_KEY = 'Position'
INSTRUMENT_KEY = 'Instrument'
PITCH_KEY = 'Pitch_24EDO'
VELOCITY_KEY = 'Velocity'
DURATION_KEY = 'Duration'
TEMPO_KEY = 'Tempo'
CHORD_KEY = 'Chord'
VOLUME_KEY = 'Volume'

NOTE_DENSITY_KEY = 'NoteDensity'
MEAN_PITCH_KEY = 'MeanPitch'
MEAN_VELOCITY_KEY = 'MeanVelocity'
MEAN_DURATION_KEY = 'MeanDuration'

REST_TOKEN = 'Rest'
