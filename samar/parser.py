# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 20:12:24 2025

@author: zaher
"""

# === File: samar/parser.py ===
# MusicXMLParser & SamarNote for note extraction


import xml.etree.ElementTree as ET
from .constants import DEFAULT_RESOLUTION

# Class representing a single note in the SAMAR format
class SamarNote:
    def __init__(self, start_tick, step, alter, octave, duration, instrument, velocity=64, is_rest=False):
        self.start_tick = int(start_tick)  # start time in ticks
        self.step = step  # note step (e.g., C, D, E)
        self.alter = float(alter) if alter is not None else 0.0  # microtonal alteration (e.g., quarter-tone)
        self.octave = int(octave)  # octave number
        self.duration = int(duration)  # note duration in ticks
        self.instrument = instrument  # instrument name or part
        self.velocity = int(velocity)  # dynamic value (MIDI velocity)
        self.is_rest = is_rest  # whether the note is a rest

    # Convert to 24EDO pitch representation (doubling MIDI scale)
    def to_24edo_pitch(self):
        if self.is_rest:
            return None
        step_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        base_pitch = step_map[self.step] + self.alter
        midi_pitch = 12 * (self.octave + 1) + base_pitch
        return int(round(midi_pitch * 2))

# Parser class to extract notes and metadata from a MusicXML file
class MusicXMLParser:
    def __init__(self, path):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.notes = self.parse_notes()
        self.measures = self._group_notes_by_measure()

    # Parse all notes from the MusicXML structure
    def parse_notes(self):
        notes = []

        # Read <divisions> to scale note durations to ticks
        divisions = 1  # fallback value
        first_divisions = self.root.find(".//divisions")
        if first_divisions is not None and first_divisions.text.isdigit():
            divisions = int(first_divisions.text)

        ticks_per_division = DEFAULT_RESOLUTION / divisions

        # Map instrument part IDs to instrument names
        part_names = {}
        for part in self.root.findall(".//score-part"):
            pid = part.attrib.get("id")
            name = part.findtext("part-name", default="Instrument")
            part_names[pid] = name

        # Iterate over each part and extract notes
        for part in self.root.findall(".//part"):
            part_id = part.attrib.get("id")
            instrument = part_names.get(part_id, "Unknown")
            current_tick = 0

            # Read dynamics if available from <sound dynamics>
            dynamic_val = 64
            sound_tag = part.find(".//sound[@dynamics]")
            if sound_tag is not None:
                dynamic_val = int(float(sound_tag.attrib.get("dynamics", 64)))

            # Process each measure and note
            for measure in part.findall("measure"):
                for note in measure.findall("note"):
                    rest = note.find("rest") is not None
                    pitch = note.find("pitch")
                    duration_divs = int(note.findtext("duration", default="1"))
                    tick_duration = duration_divs * ticks_per_division

                    if rest:
                        notes.append(SamarNote(current_tick, "C", 0, 4, tick_duration, instrument, velocity=dynamic_val, is_rest=True))
                    elif pitch is not None:
                        step = pitch.findtext("step", "C")
                        alter = pitch.findtext("alter", "0")
                        octave = pitch.findtext("octave", "4")
                        notes.append(SamarNote(current_tick, step, alter, octave, tick_duration, instrument, velocity=dynamic_val))

                    current_tick += tick_duration
        return notes

    # Optional: Group notes by estimated bar index (for inspection/debugging)
    def _group_notes_by_measure(self):
        grouped = {}
        for note in self.notes:
            bar_idx = note.start_tick // (DEFAULT_RESOLUTION * 4)  # crude bar estimate
            grouped.setdefault(bar_idx, []).append(note)
        return grouped

    def parse_time_signature(self):
        for ts in self.root.findall(".//time"):
            beats = ts.findtext("beats")
            beat_type = ts.findtext("beat-type")
            if beats and beat_type:
                return int(beats), int(beat_type)
        return 4, 4

    def parse_tempo(self):
        sound = self.root.find(".//sound[@tempo]")
        if sound is not None:
            tempo = sound.get("tempo")
            if tempo:
                return int(float(tempo))
        metronome = self.root.find(".//metronome/per-minute")
        if metronome is not None:
            return int(float(metronome.text.strip()))
        return None

    def parse_key_signature(self):
        for key in self.root.findall(".//key"):
            fifths = key.findtext("fifths")
            if fifths:
                return f"{fifths}"
        return None

