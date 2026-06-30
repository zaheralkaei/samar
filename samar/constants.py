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

NOTE_DENSITY_KEY = 'NoteDensity'
MEAN_PITCH_KEY = 'MeanPitch'
MEAN_VELOCITY_KEY = 'MeanVelocity'
MEAN_DURATION_KEY = 'MeanDuration'

# Round-20: tokens for ties, dotted notes, tuplets, chords. These are
# the structural MusicXML features the round-19 audit found missing.
TIE_KEY = 'Tie'             # Tie_Start, Tie_Stop
DOT_KEY = 'Dot'              # Dot_0 (plain), Dot_1 (dotted), Dot_2 (double-dotted)
TUPLET_KEY = 'Tuplet'        # Tuplet_3, Tuplet_5, Tuplet_7 (normal-notes always 2)
CHORD_KEY = 'Chord'          # Chord_On (non-first member of a chord group)
# Round-23: Staff_1 (treble, pitch >= middle C) and Staff_2 (bass, pitch < middle C)
# for piano two-staff output. The model learns which hand plays each note.
STAFF_KEY = 'Staff'

# Round-23: pitch threshold (MIDI) separating treble (right hand) from
# bass (left hand). Middle C = 60. Notes with pitch >= 60 go on staff 1
# (treble clef); notes with pitch < 60 go on staff 2 (bass clef).
# Chord members inherit the staff of the first note in the chord.
STAFF_TREBLE_PITCH_THRESHOLD = 60

# Discrete values for the new tokens
DOT_PLAIN = 0
DOT_SINGLE = 1
DOT_DOUBLE = 2

TUPLET_VALUES = [3, 5, 6, 7, 12]  # actual-notes count; normal-notes always 2

# === Misc ===
# Default instrument label when a ``<part>`` doesn't carry a ``<part-name>``.
# Maps to the existing ``Instrument_Voice`` token in
# ``SamarVocab`` / ``DescriptionVocab`` (no new vocab entries needed).
# Audit round-2 finding A4.
DEFAULT_INSTRUMENT = 'Voice'

# === Trainer audit version ===
# Bump when ``train_samar_transformer.py`` or ``train_samar_vae.py``
# changes its training-loop semantics (loss, optimizer, schedule,
# gradient handling). Set/checkpoint saved alongside the weights
# allows future audits to detect silent training-loop drift without
# diffing the source.
AUDIT_TRAINER_VERSION = 'round-3'
