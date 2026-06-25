# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:54:36 2025

@author: zaher
"""
# # === File: samar/input_representation.py === 
# Builds REMI+ event sequences from notes + metadata
from .constants import (
    BAR_KEY,
    TIME_SIGNATURE_KEY,
    POSITION_KEY,
    PITCH_KEY,
    DURATION_KEY,
    TEMPO_KEY,
    KEY_SIGNATURE_KEY,
    VELOCITY_KEY,
    INSTRUMENT_KEY,
    NOTE_DENSITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_DURATION_KEY,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_DURATION_BINS,
    DEFAULT_TEMPO_BINS,
    DEFAULT_NOTE_DENSITY_BINS,
    DEFAULT_MEAN_PITCH_BINS,
    DEFAULT_MEAN_VELOCITY_BINS,
    DEFAULT_MEAN_DURATION_BINS,
    DEFAULT_VELOCITY_BINS,
    DEFAULT_RESOLUTION,
)
from .parser import SamarNote, MusicXMLParser
from .metadata_extractor import extract_metadata
import numpy as np

# Event class representing a symbolic music event with metadata
class Event:
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return f"Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})"

# Converts MusicXML input into SAMAR event representation
class SAMARInputRepresentation:
    def __init__(self, xml_file):
        self.parser = MusicXMLParser(xml_file)
        self.notes = sorted(self.parser.parse_notes(), key=lambda n: n.start_tick)
        self.metadata = extract_metadata(xml_file)
        self.time_sig = self.parser.parse_time_signature()
        self.description_tokens = self._build_description_tokens()
        self.events = self._build_remi_events()

    # Build high-level description tokens based on extracted metadata
    def _build_description_tokens(self):
        return [
        f"{k}_{v}" for k, v in self.metadata.items()
        if isinstance(v, (int, float, str)) and v is not None
    ]

    # Accessor for description tokens
    def get_description_tokens(self):
        return self.description_tokens

    # Accessor for event sequence
    def get_event_sequence(self):
    # Combine metadata and musical events
        return self.get_description_tokens() + self.events

    # Core logic to convert notes and metadata into REMI+ style event tokens
    def _build_remi_events(self):
        events = []
        if not self.notes:
            return events

        # Calculate bar structure from time signature
        beats, beat_type = self.time_sig
        quarters_per_bar = 4 * (beats / beat_type)
        time_sig_parts = self.metadata.get('TimeSignature', '4/4').split('/')
        numerator = int(time_sig_parts[0]) if len(time_sig_parts) > 0 else 4
        denominator = int(time_sig_parts[1]) if len(time_sig_parts) > 1 else 4
        quarters_per_bar = 4 * numerator / denominator
        ticks_per_bar = int(DEFAULT_RESOLUTION * quarters_per_bar)
        positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)

        # Group notes by bar
        notes_by_bar = {}
        for note in self.notes:
            bar_num = note.start_tick // ticks_per_bar + 1
            notes_by_bar.setdefault(bar_num, []).append(note)

        # Add header events (time signature, key, tempo)
        time_sig = f"{beats}/{beat_type}"
        if time_sig:
            events.append(Event(TIME_SIGNATURE_KEY, None, time_sig, time_sig))
        if self.metadata.get("KeySignature"):
            events.append(Event(KEY_SIGNATURE_KEY, None, self.metadata["KeySignature"], str(self.metadata["KeySignature"])))
        if self.metadata.get("Tempo"):
            tempo = int(self.metadata["Tempo"])
            tempo_idx = np.argmin(np.abs(DEFAULT_TEMPO_BINS - tempo))
            events.append(Event(TEMPO_KEY, None, tempo_idx, str(tempo)))

        # Process each bar's notes and compute per-bar statistics
        for bar_num in sorted(notes_by_bar.keys()):
            bar_notes = notes_by_bar[bar_num]
            events.append(Event(BAR_KEY, None, bar_num, str(bar_num)))

            # Compute bar-level statistics
            note_density = len(bar_notes) / positions_per_bar
            avg_velocity = np.mean([n.velocity for n in bar_notes if n.velocity is not None])
            avg_pitch = np.mean([n.to_24edo_pitch() for n in bar_notes if not n.is_rest and n.to_24edo_pitch() is not None])
            avg_duration = np.mean([n.duration for n in bar_notes])

            # Quantize and encode bar-level statistics
            d_idx = np.argmin(np.abs(DEFAULT_NOTE_DENSITY_BINS - note_density))
            v_idx = np.argmin(np.abs(DEFAULT_MEAN_VELOCITY_BINS - avg_velocity))
            p_idx = np.argmin(np.abs(DEFAULT_MEAN_PITCH_BINS - avg_pitch))
            dur_idx = np.argmin(np.abs(DEFAULT_MEAN_DURATION_BINS - avg_duration))

            events.append(Event(NOTE_DENSITY_KEY, None, d_idx, str(note_density)))
            events.append(Event(MEAN_VELOCITY_KEY, None, v_idx, str(avg_velocity)))
            events.append(Event(MEAN_PITCH_KEY, None, p_idx, str(avg_pitch)))
            events.append(Event(MEAN_DURATION_KEY, None, dur_idx, str(avg_duration)))

            # Process individual notes in the bar
            for note in bar_notes:
                rel_tick = note.start_tick % ticks_per_bar
                position = int(rel_tick / ticks_per_bar * positions_per_bar)
                events.append(Event(POSITION_KEY, note.start_tick, position, str(position)))
                
                if note.is_rest:
                    # Rest note: encode as "Rest" with velocity 0
                    events.append(Event(PITCH_KEY, note.start_tick, "Rest", "Rest"))
                    events.append(Event(VELOCITY_KEY, note.start_tick, 0, "0"))
                else:
                    # Non-rest note: encode pitch and velocity
                    pitch_val = note.to_24edo_pitch()
                    events.append(Event(PITCH_KEY, note.start_tick, pitch_val, str(pitch_val)))
                    if note.velocity is not None:
                        vel_idx = np.argmin(np.abs(DEFAULT_VELOCITY_BINS - note.velocity))
                        events.append(Event(VELOCITY_KEY, note.start_tick, vel_idx, str(note.velocity)))

                # Encode duration using bins
                duration_pos = int(note.duration / DEFAULT_RESOLUTION * DEFAULT_POS_PER_QUARTER)
                duration_idx = np.argmin(np.abs(DEFAULT_DURATION_BINS - duration_pos))
                events.append(Event(DURATION_KEY, note.start_tick, duration_idx, str(duration_pos)))
                
                # Encode instrument information if present
                if note.instrument:
                    events.append(Event(INSTRUMENT_KEY, note.start_tick, note.instrument, note.instrument))

        # Convert list of Event objects into list of string tokens
        return [f"{e.name}_{e.value}" for e in events]
