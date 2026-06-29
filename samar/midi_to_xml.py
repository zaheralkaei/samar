# -*- coding: utf-8 -*-
"""
Round-21: MIDI -> MusicXML conversion for SAMAR training.

Instead of duplicating the entire MusicXMLParser pipeline, this
module converts a .mid file into an in-memory MusicXML tree
that MusicXMLParser can parse directly.

Round-21 additions (vs round-10):
  - Preserves polyphony via <chord/> elements (notes that share
    the same start_tick become chord members in MusicXML)
  - Emits <tie type="start"/>/<tie type="stop"/> for notes that
    cross bar boundaries. This is the only kind of tie detectable
    from MIDI, since MIDI itself doesn't have a tie concept.
  - Quantizes note durations to standard values BEFORE emitting
    so the round-19 duration fix doesn't drop them as non-standard.

MIDI files don't preserve articulation, slur, fermata, or
explicit tuplet info -- those tokens stay at 0 in the resulting
training data. Tuplets, slurs, and articulations can only be
learned from non-MIDI sources (e.g. Arabic MuseScore export).
"""

import xml.etree.ElementTree as ET
import pretty_midi
from collections import defaultdict
from .constants import DEFAULT_RESOLUTION


# Standard note values at div=480 (ticks per quarter).
# Same set the round-19 reconstructor uses. We snap to these
# so the round-20 tokenizer won't drop them as non-standard.
STANDARD_DURATIONS = [30, 60, 120, 240, 480, 720, 960, 1440, 1920]


def _quantize_duration(d):
    """Snap a tick value to the nearest standard note duration.

    Returns the closest of STANDARD_DURATIONS (or `d` itself if it's
    already a standard value).
    """
    if d in STANDARD_DURATIONS:
        return d
    return min(STANDARD_DURATIONS, key=lambda s: abs(s - d))


def midi_to_musicxml_root(midi_path):
    """Parse a .mid file and return an ElementTree root for MusicXML.

    The returned root has the same structure as a parsed MusicXML
    score-partwise document, so MusicXMLParser can use it directly.

    Multi-instrument pieces are collapsed into one part since
    SAMAR's training chunks are single-stream; instruments are
    concatenated in order.

    Returns: ElementTree.Element (the root) or None on failure.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"  [midi2xml] failed to load {midi_path}: {e}")
        return None

    # Determine tempo (use first non-zero tempo, default 120).
    times, tempos = midi.get_tempo_changes()
    tempo_bpm = 120.0
    for t in tempos:
        if t > 1.0:
            tempo_bpm = float(t)
            break
    sec_per_tick = 60.0 / tempo_bpm / DEFAULT_RESOLUTION

    # Build MusicXML
    root = ET.Element("score-partwise")
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"
    part = ET.SubElement(root, "part", id="P1")

    # 4/4 timing assumed (round-19 fix); the round-20 reconstructor
    # pads under-filled measures with rests to fill 1920 ticks.
    ticks_per_bar = DEFAULT_RESOLUTION * 4  # 1920 ticks per bar in 4/4

    # Group notes by instrument (so chord detection is per-instrument),
    # then sort by start tick to find chord members.
    inst_notes = defaultdict(list)
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            start_tick = int(round(note.start / sec_per_tick))
            end_tick = int(round(note.end / sec_per_tick))
            duration = end_tick - start_tick
            if duration <= 0:
                continue
            inst_notes[id(inst)].append(
                (start_tick, end_tick, note.pitch, int(note.velocity))
            )

    # Collect all notes (across instruments) with cross-barline
    # handling. For each note that crosses a barline, split it
    # into a "kept" portion (in current bar) and an "overflow"
    # portion (in next bar) with a tie marker.
    notes_by_bar = defaultdict(list)
    # Format: list of (start_tick, pitch, duration, velocity, flags)
    # where flags is a dict: {"tie_start": bool, "tie_stop": bool,
    # "is_chord": bool (set later at emit time)}.

    for inst_id in sorted(inst_notes.keys()):
        notes = inst_notes[inst_id]
        # Sort by start tick; ties broken by pitch (descending) so
        # the highest note is emitted first.
        notes.sort(key=lambda n: (n[0], -n[2]))

        for start_tick, end_tick, pitch, vel in notes:
            current_bar = (start_tick // ticks_per_bar) + 1
            current_bar_end = current_bar * ticks_per_bar

            if end_tick > current_bar_end:
                # Cross-barline: split into kept + overflow.
                kept_duration = current_bar_end - start_tick
                overflow = end_tick - current_bar_end

                if kept_duration > 0:
                    qkept = _quantize_duration(kept_duration)
                    notes_by_bar[current_bar].append(
                        (start_tick, pitch, qkept, vel,
                         {"tie_start": True, "tie_stop": False})
                    )

                if overflow > 0:
                    # The overflow note lives in the next bar and
                    # gets tie_stop = True (it's the second half of
                    # a tied run).
                    qoverflow = _quantize_duration(overflow)
                    notes_by_bar[current_bar + 1].append(
                        (current_bar_end, pitch, qoverflow, vel,
                         {"tie_start": False, "tie_stop": True})
                    )
            else:
                # Note fits in one bar.
                qdur = _quantize_duration(end_tick - start_tick)
                notes_by_bar[current_bar].append(
                    (start_tick, pitch, qdur, vel,
                     {"tie_start": False, "tie_stop": False})
                )

    # Emit each bar
    for bar_num in sorted(notes_by_bar.keys()):
        measure = ET.SubElement(part, "measure", number=str(bar_num))

        # First measure gets attributes
        if bar_num == 1:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(DEFAULT_RESOLUTION)
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = "4"
            ET.SubElement(time, "beat-type").text = "4"
            key = ET.SubElement(attrs, "key")
            ET.SubElement(key, "fifths").text = "0"

        # Sort notes by start_tick; chord members inherit previous start
        prev_start = None
        for start_tick, midi_pitch, duration, velocity, flags in sorted(
            notes_by_bar[bar_num]
        ):
            step, alter, octave = _midi_to_step_octave(midi_pitch)
            note_el = ET.SubElement(measure, "note")

            # Round-21: <chord/> for notes that share start_tick with
            # the previous note in this measure (polyphony marker).
            if prev_start is not None and start_tick == prev_start:
                ET.SubElement(note_el, "chord")
            prev_start = start_tick

            # Round-21: <tie type="start"/> or <tie type="stop"/> based
            # on the flags set during cross-barline splitting.
            if flags.get("tie_start") or flags.get("tie_stop"):
                notations = ET.SubElement(note_el, "notations")
                tie_type = "start" if flags["tie_start"] else "stop"
                ET.SubElement(notations, "tied", type=tie_type)

            pitch_el = ET.SubElement(note_el, "pitch")
            ET.SubElement(pitch_el, "step").text = step
            ET.SubElement(pitch_el, "octave").text = str(octave)
            if alter != 0:
                ET.SubElement(pitch_el, "alter").text = str(alter)

            ET.SubElement(note_el, "duration").text = str(duration)
            ET.SubElement(note_el, "voice").text = '1'
            ET.SubElement(note_el, "instrument").text = "Piano"

            # Add velocity as a <sound> element with dynamics attribute
            # (MusicXML's standard way to convey MIDI velocity)
            sound = ET.SubElement(note_el, "sound")
            sound.set("dynamics", str(velocity))

    return root


def _midi_to_step_octave(midi_pitch):
    """Convert MIDI pitch (0-127) to (step, alter, octave)."""
    octave = midi_pitch // 12 - 1
    semitone_in_octave = midi_pitch % 12
    step_map = {0: 'C', 1: 'C', 2: 'D', 3: 'D', 4: 'E', 5: 'F',
                6: 'F', 7: 'G', 8: 'G', 9: 'A', 10: 'A', 11: 'B'}
    alter_map = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0, 5: 0,
                 6: 1, 7: 0, 8: 1, 9: 0, 10: 1, 11: 0}
    return step_map[semitone_in_octave], alter_map[semitone_in_octave], octave
