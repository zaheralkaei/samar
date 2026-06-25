# -*- coding: utf-8 -*-
"""
SAMAR canonical MusicXML input representation.

This module is the single source of truth for MusicXML -> REMI+ conversion
(``SamarNote``, ``MusicXMLParser``, ``extract_metadata``, ``Event``,
``SAMARInputRepresentation``). Older codebases had two parallel
implementations (one in :mod:`samar.parser` and one in
:mod:`samar.input_representation`); they now re-export from here.

See ``docs/audit-2026-06-25.md`` finding #1 for the rationale.
"""

import xml.etree.ElementTree as ET
import numpy as np
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


class SamarNote:
    """A single note (or rest) in the SAMAR format. Supports 24-EDO alters."""

    def __init__(self, start_tick, step, alter, octave, duration, instrument, velocity=64, is_rest=False):
        self.start_tick = int(start_tick)
        self.step = step
        self.alter = float(alter) if alter is not None else 0.0
        self.octave = int(octave)
        self.duration = int(duration)
        self.instrument = instrument
        self.velocity = int(velocity)
        self.is_rest = is_rest

    def to_24edo_pitch(self):
        """Convert to 24-EDO pitch space (MIDI pitch * 2).

        Returns ``None`` for rests. Quarter-tone alters are preserved
        via the fractional ``alter`` field; e.g. ``alter=-0.5`` lowers the
        note by a quarter-tone before doubling to 24-EDO.
        """
        if self.is_rest:
            return None
        step_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        base_pitch = step_map[self.step] + self.alter
        midi_pitch = 12 * (self.octave + 1) + base_pitch
        return int(round(midi_pitch * 2))


class MusicXMLParser:
    """Parse a MusicXML file into ``SamarNote`` objects.

    Honors per-bar time-signature changes (``<time>`` elements inside any
    ``<measure>``) when computing bar lengths, so a piece that switches
    2/4 -> 4/4 mid-score is grouped correctly.
    """

    def __init__(self, path):
        self.tree = ET.parse(path)
        self.root = self.tree.getroot()
        self.time_signatures = self._parse_time_signatures_by_bar()
        self.notes = self.parse_notes()
        self.measures = self._group_notes_by_measure()

    def _parse_time_signatures_by_bar(self):
        """Map ``bar_number -> (beats, beat_type)`` for every bar that defines one.

        Bars without an explicit ``<time>`` element inherit the most recent
        time signature (handled in :meth:`parse_notes`).
        """
        bar_time_sigs = {}
        for part in self.root.findall(".//part"):
            for measure in part.findall("measure"):
                bar_number = int(measure.attrib.get("number", 1))
                time_elem = measure.find("attributes/time")
                if time_elem is not None:
                    beats = time_elem.findtext("beats")
                    beat_type = time_elem.findtext("beat-type")
                    if beats and beat_type:
                        bar_time_sigs[bar_number] = (int(beats), int(beat_type))
        return bar_time_sigs

    def parse_notes(self):
        notes = []
        divisions = 1
        first_divisions = self.root.find(".//divisions")
        if first_divisions is not None and first_divisions.text.isdigit():
            divisions = int(first_divisions.text)
        ticks_per_division = DEFAULT_RESOLUTION / divisions

        part_names = {}
        for part in self.root.findall(".//score-part"):
            pid = part.attrib.get("id")
            name = part.findtext("part-name", default="Instrument")
            part_names[pid] = name

        # Walk parts in score order; tick counter is shared so multi-part
        # scores stay time-aligned.
        for part in self.root.findall(".//part"):
            part_id = part.attrib.get("id")
            instrument = part_names.get(part_id, "Unknown")
            current_tick = 0
            for measure in part.findall("measure"):
                bar_number = int(measure.attrib.get("number", 1))
                # Use this bar's time signature, falling back to 4/4 if
                # the score never declared one (very rare in real XML).
                beats, beat_type = self.time_signatures.get(bar_number, (4, 4))
                ticks_per_bar = int(DEFAULT_RESOLUTION * (4 * beats / beat_type))
                for note in measure.findall("note"):
                    rest = note.find("rest") is not None
                    pitch = note.find("pitch")
                    duration_divs = int(note.findtext("duration", default="1"))
                    tick_duration = duration_divs * ticks_per_division

                    if rest:
                        notes.append(SamarNote(current_tick, "C", 0, 4, tick_duration, instrument, velocity=64, is_rest=True))
                    elif pitch is not None:
                        step = pitch.findtext("step", "C")
                        alter = pitch.findtext("alter", "0")
                        octave = pitch.findtext("octave", "4")
                        notes.append(SamarNote(current_tick, step, alter, octave, tick_duration, instrument, velocity=64))

                    current_tick += tick_duration
        return notes

    def _group_notes_by_measure(self):
        grouped = {}
        for note in self.notes:
            bar_idx = note.start_tick // (DEFAULT_RESOLUTION * 4)
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
        """First ``<sound tempo="...">`` in the score, or None."""
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
        """Number of fifths as a string, or None if missing."""
        for key in self.root.findall(".//key"):
            fifths = key.findtext("fifths")
            if fifths:
                return fifths
        return None


def extract_metadata(xml_path):
    """Pull composer/lyricist credits, time signature, key, tempo out of an XML file.

    The returned dict's keys are split into two groups:
      * ``Description_*`` -- meant for the description vocabulary, see
        ``figaro/src/vocab.py:DescriptionVocab``.
      * ``TimeSignature``, ``KeySignature``, ``Tempo``, ``Instruments`` --
        consumed by ``SAMARInputRepresentation._build_remi_events``.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    metadata = {}
    for credit in root.findall("credit"):
        credit_type = credit.find("credit-type")
        credit_words = credit.find("credit-words")
        if credit_type is not None and credit_words is not None:
            key = f"Description_{credit_type.text.capitalize()}"
            metadata[key] = credit_words.text.strip()
    creator = root.find(".//creator[@type='composer']")
    if creator is not None:
        metadata["Description_Composer"] = creator.text.strip()
    for score_part in root.findall(".//score-part"):
        part_name = score_part.find("part-name")
        if part_name is not None:
            metadata.setdefault("Instruments", []).append(part_name.text.strip())
    first_measure = root.find(".//part/measure")
    if first_measure is not None:
        time = first_measure.find("attributes/time")
        if time is not None:
            beats = time.find("beats")
            beat_type = time.find("beat-type")
            if beats is not None and beat_type is not None:
                metadata["TimeSignature"] = f"{beats.text}/{beat_type.text}"
        key = first_measure.find("attributes/key/fifths")
        if key is not None:
            metadata["KeySignature"] = int(key.text)
        sound = first_measure.find(".//sound")
        if sound is not None:
            tempo = sound.attrib.get("tempo")
            if tempo is not None:
                metadata["Tempo"] = int(float(tempo))
    return metadata


class Event:
    """A single symbolic-music event with metadata (kept for backwards compat)."""

    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return f"Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})"


class SAMARInputRepresentation:
    """Convert MusicXML -> a sequence of REMI+ tokens.

    Splits into two streams (matching FIGARO's design, see
    ``figaro/src/input_representation.py``):

      * ``description_tokens`` -- time-signature / mean-pitch /
        note-density / etc. tokens that condition the generation.
      * ``events`` -- the per-note REMI+ event sequence (Bar, Position,
        Pitch_24EDO, Velocity, Duration, Instrument).
    """

    def __init__(self, xml_file):
        self.parser = MusicXMLParser(xml_file)
        self.notes = sorted(self.parser.notes, key=lambda n: n.start_tick)
        self.metadata = extract_metadata(xml_file)
        self.time_sig = self.parser.parse_time_signature()
        self.description_tokens = self._build_description_tokens()
        self.events = self._build_remi_events()

    @staticmethod
    def _safe_mean(values):
        """``np.mean(values)`` or 0 if the input is empty.

        Avoids ``RuntimeWarning: Mean of empty slice`` when a bar contains
        only rests or the file has no pitched notes.
        """
        if not values:
            return 0
        return np.mean(values)

    def _build_description_tokens(self):
        """Build description tokens following the FIGARO naming convention.

        Token keys match ``MEAN_PITCH_KEY`` (``"MeanPitch"``),
        ``MEAN_VELOCITY_KEY`` (``"MeanVelocity"``),
        ``MEAN_DURATION_KEY`` (``"MeanDuration"``) and
        ``NOTE_DENSITY_KEY`` (``"NoteDensity"``) -- these are exactly the
        keys produced by ``figaro/src/vocab.py:DescriptionVocab``. Earlier
        versions used ``AveragePitch`` / ``AverageVelocity`` /
        ``AverageDuration``, which silently mapped to ``<unk>`` because the
        description vocab never contained them.

        ``Description_*`` keys (composer credits, lyricist credits),
        ``KeySignature_*`` and ``Tempo_*`` are deliberately excluded:
        they're free-text / unstructured metadata that FIGARO never encodes
        in the description stream (``figaro/src/vocab.py:DescriptionVocab``
        has no slots for them). Tempo and key signature instead appear in
        the event stream via :meth:`_build_remi_events`. We still extract
        them in :func:`extract_metadata` for downstream filtering or
        human-readable labels, but they don't enter the description tokens.

        All four statistics are quantized to their bin indices (matching
        how :meth:`_build_remi_events` emits per-bar statistics). Without
        this step the raw float values would not match any vocab token.
        """
        tokens = [
            f"{k}_{v}" for k, v in self.metadata.items()
            if isinstance(v, (int, float, str))
            and v is not None
            and not k.startswith("Description_")
            and not k.startswith("KeySignature")
            and not k.startswith("Tempo")
        ]

        if self.notes:
            velocities = [n.velocity for n in self.notes if not n.is_rest]
            durations = [n.duration for n in self.notes if not n.is_rest]
            pitches = [n.to_24edo_pitch() for n in self.notes
                       if not n.is_rest and n.to_24edo_pitch() is not None]

            avg_vel = int(self._safe_mean(velocities))
            avg_dur = int(self._safe_mean(durations))
            avg_pitch = int(self._safe_mean(pitches))

            beats, beat_type = self.time_sig
            quarters_per_bar = 4 * beats / beat_type
            ticks_per_bar = int(DEFAULT_RESOLUTION * quarters_per_bar)
            positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)
            total_bars = max((n.start_tick // ticks_per_bar + 1) for n in self.notes)

            note_density = round(len(self.notes) / (positions_per_bar * total_bars), 3)

            # Quantize each statistic to its nearest bin so the emitted
            # token (``MeanPitch_{N}`` etc.) actually exists in the vocab.
            # ``np.argmin(abs(bins - x))`` returns the closest bin index,
            # which matches how :meth:`_build_remi_events` does it for the
            # per-bar statistics.
            p_idx = int(np.argmin(np.abs(DEFAULT_MEAN_PITCH_BINS - avg_pitch)))
            v_idx = int(np.argmin(np.abs(DEFAULT_MEAN_VELOCITY_BINS - avg_vel)))
            d_idx = int(np.argmin(np.abs(DEFAULT_MEAN_DURATION_BINS - avg_dur)))
            nd_idx = int(np.argmin(np.abs(DEFAULT_NOTE_DENSITY_BINS - note_density)))

            tokens += [
                f"{MEAN_PITCH_KEY}_{p_idx}",
                f"{MEAN_VELOCITY_KEY}_{v_idx}",
                f"{MEAN_DURATION_KEY}_{d_idx}",
                f"{NOTE_DENSITY_KEY}_{nd_idx}",
            ]

        return tokens

    def get_description_tokens(self):
        return self.description_tokens

    def get_event_sequence(self):
        return self.get_description_tokens() + self.events

    def _build_remi_events(self):
        events = []
        if not self.notes:
            return events

        beats, beat_type = self.time_sig
        time_sig_parts = self.metadata.get('TimeSignature', '4/4').split('/')
        numerator = int(time_sig_parts[0]) if len(time_sig_parts) > 0 else 4
        denominator = int(time_sig_parts[1]) if len(time_sig_parts) > 1 else 4
        quarters_per_bar = 4 * numerator / denominator
        ticks_per_bar = int(DEFAULT_RESOLUTION * quarters_per_bar)
        positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)

        notes_by_bar = {}
        for note in self.notes:
            bar_num = note.start_tick // ticks_per_bar + 1
            notes_by_bar.setdefault(bar_num, []).append(note)

        if time_sig_parts:
            events.append(Event(TIME_SIGNATURE_KEY, None, f"{beats}/{beat_type}", f"{beats}/{beat_type}"))
        if self.metadata.get("KeySignature") is not None:
            events.append(Event(KEY_SIGNATURE_KEY, None, self.metadata["KeySignature"], str(self.metadata["KeySignature"])))
        if self.metadata.get("Tempo") is not None:
            tempo = int(self.metadata["Tempo"])
            tempo_idx = np.argmin(np.abs(DEFAULT_TEMPO_BINS - tempo))
            events.append(Event(TEMPO_KEY, None, tempo_idx, str(tempo)))

        for bar_num in sorted(notes_by_bar.keys()):
            bar_notes = notes_by_bar[bar_num]
            events.append(Event(BAR_KEY, None, bar_num, str(bar_num)))

            # Compute bar-level statistics. ``_safe_mean`` keeps a rest-only
            # bar from emitting ``RuntimeWarning: Mean of empty slice``.
            note_density = len(bar_notes) / positions_per_bar
            avg_velocity = self._safe_mean([n.velocity for n in bar_notes if n.velocity is not None])
            avg_pitch = self._safe_mean([n.to_24edo_pitch() for n in bar_notes
                                          if not n.is_rest and n.to_24edo_pitch() is not None])
            avg_duration = self._safe_mean([n.duration for n in bar_notes])

            d_idx = np.argmin(np.abs(DEFAULT_NOTE_DENSITY_BINS - note_density))
            v_idx = np.argmin(np.abs(DEFAULT_MEAN_VELOCITY_BINS - avg_velocity))
            p_idx = np.argmin(np.abs(DEFAULT_MEAN_PITCH_BINS - avg_pitch))
            dur_idx = np.argmin(np.abs(DEFAULT_MEAN_DURATION_BINS - avg_duration))

            events.append(Event(NOTE_DENSITY_KEY, None, d_idx, str(note_density)))
            events.append(Event(MEAN_VELOCITY_KEY, None, v_idx, str(avg_velocity)))
            events.append(Event(MEAN_PITCH_KEY, None, p_idx, str(avg_pitch)))
            events.append(Event(MEAN_DURATION_KEY, None, dur_idx, str(avg_duration)))

            for note in bar_notes:
                rel_tick = note.start_tick % ticks_per_bar
                position = int(rel_tick / ticks_per_bar * positions_per_bar)
                events.append(Event(POSITION_KEY, note.start_tick, position, str(position)))

                if note.is_rest:
                    events.append(Event(PITCH_KEY, note.start_tick, "Rest", "Rest"))
                    events.append(Event(VELOCITY_KEY, note.start_tick, 0, "0"))
                else:
                    pitch_val = note.to_24edo_pitch()
                    events.append(Event(PITCH_KEY, note.start_tick, pitch_val, str(pitch_val)))
                    if note.velocity is not None:
                        vel_idx = np.argmin(np.abs(DEFAULT_VELOCITY_BINS - note.velocity))
                        events.append(Event(VELOCITY_KEY, note.start_tick, vel_idx, str(note.velocity)))

                duration_pos = int(note.duration / DEFAULT_RESOLUTION * DEFAULT_POS_PER_QUARTER)
                duration_idx = np.argmin(np.abs(DEFAULT_DURATION_BINS - duration_pos))
                events.append(Event(DURATION_KEY, note.start_tick, duration_idx, str(duration_pos)))

                if note.instrument:
                    events.append(Event(INSTRUMENT_KEY, note.start_tick, note.instrument, note.instrument))

        return [f"{e.name}_{e.value}" for e in events]