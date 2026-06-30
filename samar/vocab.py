# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:56:16 2025

@author: zaher
"""

# === File: samar/vocab.py ===
# Token definitions & Vocab classes

from collections import Counter
from .constants import (
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
    # Round-20: structural music tokens
    TIE_KEY,
    DOT_KEY,
    TUPLET_KEY,
    TUPLET_VALUES,
    # Round-23: staff tokens for two-staff piano output
    STAFF_KEY,
)

class Tokens:
    def get_instrument_tokens(key=INSTRUMENT_KEY):
        common_instrs = [
            "Violin", "Violoncello", "Flute", "Clarinet", "Trumpet",
            "Oud", "Qanun", "Nay", "Percussion", "Voice", "Piano",
            # Added round-4 audit after expanding data/xml to 78 files:
            "Harp", "Drumset", "Acoustic Guitar", "Classical Guitar",
            "Classical Guitar (Tablature)",
        ]
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
        time_sigs = [f'{p}/{q}' for q in denominators for p in range(1, MAX_BAR_LENGTH*q + 1)]
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
            # Pitch tokens cover 24-EDO MIDI 0..127 (24 * 11 = 264 quarter-tones,
            # since 24-EDO doubles MIDI pitch). The previous ``range(24 * 6)`` =
            # 144 covered only MIDI 0..71 and silently dropped higher notes to
            # ``<unk>`` -- which is most Arabic vocal music (which lives in
            # MIDI 60..96). See audit round-2 finding A3.
            #
            # FIGARO uses ``range(128)`` MIDI pitches directly (no 24-EDO
            # doubling). SAMAR's doubling is the project's deliberate divergence
            # to support quarter-tones.
            pitch_tokens = [f'{pitch_key}_{i}' for i in range(24 * 11)] + [f'{pitch_key}_Rest']
            velocity_tokens = [f'{velocity_key}_{i}' for i in range(len(DEFAULT_VELOCITY_BINS))]
            duration_tokens = [f'{duration_key}_{i}' for i in range(len(DEFAULT_DURATION_BINS))]
            tempo_tokens = [f'{tempo_key}_{i}' for i in range(len(DEFAULT_TEMPO_BINS))]
            bar_tokens = [f'{bar_key}_{i}' for i in range(MAX_N_BARS)]
            position_tokens = [f'{position_key}_{i}' for i in range(MAX_BAR_LENGTH * 4 * DEFAULT_POS_PER_QUARTER)]
            time_sig_tokens = Tokens.get_time_signature_tokens(time_signature_key)
            # Round-20: structural music tokens (ties, dots, tuplets, chord).
            # These carry the MusicXML-level structure that the round-19
            # audit found was being silently dropped by the tokenizer.
            tie_tokens = [f'{TIE_KEY}_Start', f'{TIE_KEY}_Stop']
            dot_tokens = [f'{DOT_KEY}_0', f'{DOT_KEY}_1', f'{DOT_KEY}_2']
            tuplet_tokens = [f'{TUPLET_KEY}_{n}' for n in TUPLET_VALUES]
            chord_tokens = [f'{CHORD_KEY}_On']
            # Round-23: Staff_1 (treble, right hand) and Staff_2 (bass,
            # left hand) tokens for two-staff piano output.
            staff_tokens = [f'{STAFF_KEY}_1', f'{STAFF_KEY}_2']

            return (
                time_sig_tokens +
                tempo_tokens +
                instrument_tokens +
                pitch_tokens +
                velocity_tokens +
                duration_tokens +
                bar_tokens +
                position_tokens +
                tie_tokens +
                dot_tokens +
                tuplet_tokens +
                chord_tokens +
                staff_tokens
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

class SamarVocab(Vocab):
    """Event vocabulary for MusicXML events (pitch, bar, position, ...).

    Matches ``figaro/src/vocab.py:RemiVocab`` -- deliberately excludes
    description tokens (``Mean Pitch``, ``Mean Velocity``, ``Description_*``).
    Descriptions are encoded by :class:`DescriptionTokenizer` against
    :class:`DescriptionVocab`. Earlier versions concatenated the two, which
    made every description token silently map to ``<unk>`` after the
    vocabulary and training data drifted apart.
    """
    def __init__(self):
        midi_tokens = Tokens.get_midi_tokens_samar()
        chord_tokens = Tokens.get_chord_tokens()
        self.tokens = midi_tokens + chord_tokens
        counter = Counter(self.tokens)
        super().__init__(counter)

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
            time_sig_tokens +
            instrument_tokens +
            chord_tokens +
            density_tokens +
            velocity_tokens +
            pitch_tokens +
            duration_tokens +
            bar_tokens
        )
        counter = Counter(self.tokens)
        super().__init__(counter)

        # Round-12 audit: cross-vocab ID consistency check.
        #
        # ``DescriptionVocab`` and ``SamarVocab`` (event vocab) overlap on
        # 716 strings (Bar_*, Chord_*, TimeSignature_*, Instrument_*) but
        # assign DIFFERENT IDs because they sort differently. This is a
        # latent landmine: if any code path encodes a description-side
        # token with the event tokenizer (or vice versa), the wrong token
        # comes out -- silently.
        #
        # The intended pattern (FIGARO two-stream) is to ALWAYS use the
        # right tokenizer for the right stream. We warn at module-import
        # time if any non-special token has a different ID in the two
        # vocabs. (In practice the IDs always differ for the overlapping
        # strings; the warning is the affordance -- if you see it, you
        # know which tokens are at risk of cross-tokenizer confusion.)
        try:
            _event_vocab = SamarVocab()
            _special_tokens = {PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, MASK_TOKEN}
            mismatches = []
            for tok in self.tokens:
                if tok in _special_tokens:
                    continue
                if tok in _event_vocab.stoi and _event_vocab.stoi[tok] != self.stoi[tok]:
                    mismatches.append(
                        (tok, self.stoi[tok], _event_vocab.stoi[tok])
                    )
            if mismatches:
                # Only show the first few to avoid log spam.
                sample = mismatches[:5]
                msg = (
                    f"[DescriptionVocab] {len(mismatches)} overlapping strings "
                    f"have different IDs in SamarVocab vs DescriptionVocab "
                    f"(e.g. {sample}). Cross-tokenizer use will mis-encode. "
                    f"This is by-design per FIGARO two-stream split, but be "
                    f"careful: always use SamarTokenizer for events and "
                    f"DescriptionTokenizer for descriptions."
                )
                import warnings
                warnings.warn(msg, stacklevel=2)
        except Exception:
            # Don't fail vocab construction if cross-check can't run.
            pass
