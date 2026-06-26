# -*- coding: utf-8 -*-
"""
Round-10: MIDI parser for SAMAR training data.

Parses .mid files (Western classical) into SamarNote-compatible dicts
so the existing SAMARInputRepresentation event builder can use MIDI
input the same way it uses MusicXML input.
"""

import numpy as np
import pretty_midi
from .constants import DEFAULT_RESOLUTION


def midi_to_samar_notes(midi_path):
    """Parse a .mid file into a list of SamarNote-compatible dicts.

    Uses pretty_midi to read the file. For each instrument, converts
    each note to a dict with keys:
        - start_tick: int (units of DEFAULT_RESOLUTION = 480 ticks/quarter)
        - step: 'C'..'B'
        - alter: float (0.0 or 1.0 for 12-EDO MIDI; 24-EDO can have +/- 0.5)
        - octave: int
        - duration: int (ticks)
        - instrument: str (General MIDI name)
        - velocity: int (1-127)
        - is_rest: bool

    Returns: list of SamarNote-compatible dicts, sorted by start_tick.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"  [midi] failed to load {midi_path}: {e}")
        return []

    # Use tempo map to compute ticks. Get the first non-zero tempo (BPM)
    # and use it globally. This is approximate for pieces with tempo
    # changes, but good enough for chunked training (each chunk is
    # independent of others).
    # NOTE: pretty_midi.get_tempo_changes() returns (times, tempos).
    times, tempos = midi.get_tempo_changes()
    tempo_bpm = 120.0  # default
    for t in tempos:
        if t > 1.0:
            tempo_bpm = float(t)
            break

    # ticks = (seconds * tempo_bpm / 60) * DEFAULT_RESOLUTION
    sec_per_tick = 60.0 / tempo_bpm / DEFAULT_RESOLUTION

    notes_out = []

    for inst_idx, inst in enumerate(midi.instruments):
        if inst.is_drum:
            continue  # skip drums
        if not inst.notes:
            continue

        inst_name = _gm_program_to_name(inst.program)

        for note in inst.notes:
            start_tick = int(round(note.start / sec_per_tick))
            end_tick = int(round(note.end / sec_per_tick))
            duration = end_tick - start_tick
            if duration <= 0:
                continue

            step, alter, octave = _midi_to_step_alter_octave(note.pitch)

            notes_out.append({
                'start_tick': start_tick,
                'step': step,
                'alter': alter,
                'octave': octave,
                'duration': duration,
                'instrument': inst_name,
                'velocity': int(note.velocity),
                'is_rest': False,
            })

    notes_out.sort(key=lambda n: n['start_tick'])
    return notes_out


def _midi_to_step_alter_octave(midi_pitch):
    """Convert MIDI pitch (0-127) to (step, alter, octave).

    Standard MIDI is 12-EDO so alter is always 0.0 or 1.0.
    """
    octave = midi_pitch // 12 - 1
    semitone_in_octave = midi_pitch % 12
    step_map = {0: 'C', 1: 'C', 2: 'D', 3: 'D', 4: 'E', 5: 'F',
                6: 'F', 7: 'G', 8: 'G', 9: 'A', 10: 'A', 11: 'B'}
    alter_map = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0,
                 6: 1, 7: 0, 8: 1, 9: 0, 10: 1, 11: 0}
    return step_map[semitone_in_octave], float(alter_map[semitone_in_octave]), octave


def _gm_program_to_name(program):
    """Map General MIDI program number to instrument name."""
    gm_names = [
        "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
        "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
        "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
        "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ",
        "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
        "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)",
        "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
        "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar",
        "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)",
        "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2",
        "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass",
        "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
        "String Ensemble 1", "String Ensemble 2", "Synth Strings 1",
        "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Choir",
        "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
        "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
        "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
        "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder",
        "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
        "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
        "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
        "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)",
        "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)",
        "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)",
        "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
        "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
        "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle",
        "Shanai", "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock",
        "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
        "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
        "Telephone Ring", "Helicopter", "Applause", "Gunshot"
    ]
    if 0 <= program < len(gm_names):
        return gm_names[program]
    return "Piano"