# -*- coding: utf-8 -*-
"""
Round-10: MIDI -> MusicXML conversion for SAMAR training.

Instead of duplicating the entire MusicXMLParser pipeline, this
module converts a .mid file into an in-memory MusicXML tree
that MusicXMLParser can parse directly.
"""

import xml.etree.ElementTree as ET
import pretty_midi
from .constants import DEFAULT_RESOLUTION


def midi_to_musicxml_root(midi_path):
    """Parse a .mid file and return an ElementTree root for MusicXML.

    The returned root has the same structure as a parsed MusicXML
    score-partwise document, so MusicXMLParser can use it directly.

    Single-part output (we collapse all MIDI instruments into one
    part since SAMAR's training chunks are single-stream). For
    multi-instrument pieces, instruments are concatenated in
    order with empty measures between them.

    Returns: ElementTree.Element (the root) or None on failure.
    """
    try:
        midi = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f"  [midi2xml] failed to load {midi_path}: {e}")
        return None

    # Determine tempo (use first non-zero tempo, default 120).
    # NOTE: pretty_midi.get_tempo_changes() returns (times, tempos) in
    # that order. The first array is the times of tempo changes, the
    # second is the actual BPM values at those times.
    times, tempos = midi.get_tempo_changes()
    tempo_bpm = 120.0
    for t in tempos:
        if t > 1.0:
            tempo_bpm = float(t)
            break
    sec_per_tick = 60.0 / tempo_bpm / DEFAULT_RESOLUTION

    # Build MusicXML
    #   <score-partwise>
    #     <part-list><score-part id="P1"><part-name>Piano</part-name>
    #     <part id="P1">
    #       <measure number="1">
    #         <attributes>
    #           <divisions>480</divisions>
    #           <time><beats>4</beats><beat-type>4</beat-type></time>
    #           <key><fifths>0</fifths></key>
    #         </attributes>
    #         <note>...<pitch>...</pitch>...</note>
    #       </measure>
    #     </part>
    root = ET.Element("score-partwise")
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Piano"

    part = ET.SubElement(root, "part", id="P1")

    # Group notes by bar (assume 4/4 with DEFAULT_RESOLUTION ticks/quarter)
    # Bar 1 = ticks 0..1919, Bar 2 = ticks 1920..3839, etc.
    ticks_per_bar = DEFAULT_RESOLUTION * 4  # 4 quarters per bar in 4/4
    notes_by_bar = {}
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            start_tick = int(round(note.start / sec_per_tick))
            end_tick = int(round(note.end / sec_per_tick))
            duration = end_tick - start_tick
            if duration <= 0:
                continue
            bar = (start_tick // ticks_per_bar) + 1
            notes_by_bar.setdefault(bar, []).append((start_tick, note.pitch, duration, int(note.velocity)))

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

        # Sort notes by start_tick
        for start_tick, midi_pitch, duration, velocity in sorted(notes_by_bar[bar_num]):
            step, alter, octave = _midi_to_step_octave(midi_pitch)
            note_el = ET.SubElement(measure, "note")

            pitch_el = ET.SubElement(note_el, "pitch")
            ET.SubElement(pitch_el, "step").text = step
            ET.SubElement(pitch_el, "octave").text = str(octave)
            if alter != 0:
                ET.SubElement(pitch_el, "alter").text = str(alter)

            ET.SubElement(note_el, "duration").text = str(duration)
            ET.SubElement(note_el, "voice").text = "1"
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