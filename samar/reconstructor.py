# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:04:46 2025

@author: zaher
"""

# reconstructor.py


import xml.etree.ElementTree as ET
from .constants import (
    DEFAULT_RESOLUTION,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_DURATION_BINS,
    BAR_KEY,
    POSITION_KEY,
    PITCH_KEY,
    DURATION_KEY,
    INSTRUMENT_KEY,
    TIME_SIGNATURE_KEY
)

# Round-9: defaults for orphaned pitches (model emits pitch without
# duration/instrument). 9 ticks = eighth-note at divisions=480.
DEFAULT_DURATION_IDX = 9
DEFAULT_INSTRUMENT = "Piano"  # matches P1 default below

# The first instrument used when none is specified. Keep in sync with
# instrument_map below.
DEFAULT_PART_ID = "P1"


def _ensure_attributes(measure):
    """Add MusicXML <attributes> block to a measure (time, key, clef).

    Round-7 addition: MuseScore needs these on the first measure of
    each part or it renders empty bars.
    """
    attrs = ET.SubElement(measure, 'attributes')
    div = ET.SubElement(attrs, 'divisions')
    div.text = str(DEFAULT_RESOLUTION)
    time = ET.SubElement(attrs, 'time')
    beats = ET.SubElement(time, 'beats')
    beats.text = '4'
    beat_type = ET.SubElement(time, 'beat-type')
    beat_type.text = '4'
    key = ET.SubElement(attrs, 'key')
    fifths = ET.SubElement(key, 'fifths')
    fifths.text = '0'
    clef = ET.SubElement(attrs, 'clef')
    sign = ET.SubElement(clef, 'sign')
    sign.text = 'G'
    line = ET.SubElement(clef, 'line')
    line.text = '2'


def _flush_note_buffer(note_buffer, instrument_map, measures, part_map,
                       current_bar, current_tick, last_tick,
                       original_divisions):
    """Convert a complete note_buffer (pitch + duration + instrument)
    into a <note> XML element appended to the right measure.

    The caller is responsible for managing state (current_bar,
    current_tick, last_tick, measures, part_map) and for clearing the
    buffer afterwards. This is the round-9 refactor that lets us
    flush orphaned pitches (those without full Duration/Instrument)
    before they get dropped on the next Bar token.
    """
    if not all(k in note_buffer for k in ('pitch', 'duration_idx', 'instrument')):
        return False  # nothing to flush

    inst_name = note_buffer['instrument'].strip().lower()
    pid = instrument_map.get(inst_name, DEFAULT_PART_ID)
    meas = measures.get((pid, current_bar))
    if meas is None:
        meas = ET.SubElement(part_map[pid], 'measure', number=str(current_bar))
        measures[(pid, current_bar)] = meas

    if current_tick > last_tick:
        forward = ET.SubElement(meas, 'forward')
        ET.SubElement(forward, 'duration').text = str(current_tick - last_tick)

    note_el = ET.SubElement(meas, 'note')
    pitch_val = note_buffer.get('pitch')

    if pitch_val in [None, 'Rest', 'None', '24EDO_Rest']:
        ET.SubElement(note_el, 'rest')
    else:
        try:
            midi24 = int(float(pitch_val))
            midi_pitch = midi24 / 2.0
            octave = int(midi_pitch // 12 - 1)
            semitone_exact = midi_pitch % 12
            step_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
            step = min(step_map, key=lambda k: abs(semitone_exact - step_map[k]))
            alter_qtone = round((semitone_exact - step_map[step]) * 2) / 2.0

            pitch_el = ET.SubElement(note_el, 'pitch')
            ET.SubElement(pitch_el, 'step').text = step
            if abs(alter_qtone) > 1e-6:
                ET.SubElement(pitch_el, 'alter').text = f"{alter_qtone:.1f}"
            ET.SubElement(pitch_el, 'octave').text = str(octave)
        except Exception:
            ET.SubElement(note_el, 'rest')

    idx = min(note_buffer.get('duration_idx', 0), len(DEFAULT_DURATION_BINS) - 1)
    dur_pos = DEFAULT_DURATION_BINS[idx]
    divisions = int(round(dur_pos * original_divisions / DEFAULT_POS_PER_QUARTER))
    ET.SubElement(note_el, 'duration').text = str(divisions)
    ET.SubElement(note_el, 'voice').text = '1'

    # NOTE: we cannot update `last_tick` from here because we don't
    # have a reference to it. The caller updates last_tick before
    # clearing the buffer.
    return True


def reconstruct_musicxml_from_events(input_or_events, output_xml_path: str):
    if isinstance(input_or_events, list):
        events = input_or_events
    elif isinstance(input_or_events, str) and input_or_events.endswith(('.txt', '.remi')):
        with open(input_or_events, 'r') as f:
            events = [l.strip() for l in f if l.strip()]
    elif isinstance(input_or_events, str):
        # raw event string, newline-separated
        events = [l.strip() for l in input_or_events.splitlines() if l.strip()]
    else:
        raise TypeError("Only supports REMI+ event lists (list, .txt file, or newline string)")

    if events:
        # === Build score-part / part skeleton ===
        # Round-9: simplified. We use a single P1 part for everything
        # (the round-7 logic built a part per instrument seen; the
        # round-9 orphaned-pitch flush means any leftover incomplete
        # pitch goes to P1 by default).
        part_list = ET.Element('part-list')
        score_part = ET.SubElement(part_list, 'score-part', id=DEFAULT_PART_ID)
        ET.SubElement(score_part, 'part-name').text = 'Piano'

        new_root = ET.Element('score-partwise')
        new_root.append(part_list)
        part_map = {DEFAULT_PART_ID: ET.SubElement(new_root, 'part', id=DEFAULT_PART_ID)}
        instrument_map = {DEFAULT_PART_ID.lower(): DEFAULT_PART_ID}

        # === State ===
        original_divisions = DEFAULT_RESOLUTION
        measures = {}
        note_buffer = {}
        current_bar = None
        current_tick = 0
        last_tick = 0
        current_time_signature = None
        # Round-9: bar_offset was used to remap Bar_N (vocab index) to a
        # 1-based measure number, but that broke when the model emitted
        # Bar tokens out of order (e.g. Bar_397 then Bar_195). The fix is
        # to count Bar tokens in sequence instead of using raw values:
        # first Bar_ -> measure 1, second -> measure 2, etc.
        bar_count = 0

        def _advance_last_tick():
            """Update last_tick based on note_buffer after a note is flushed."""
            nonlocal last_tick
            idx = min(note_buffer.get('duration_idx', 0), len(DEFAULT_DURATION_BINS) - 1)
            dur_pos = DEFAULT_DURATION_BINS[idx]
            divisions = int(round(dur_pos * original_divisions / DEFAULT_POS_PER_QUARTER))
            last_tick = current_tick + divisions

        for ev in events:
            if ev.startswith(BAR_KEY + "_"):
                # Flush any orphaned pitch in the buffer before advancing
                # to a new measure (round-9 fix).
                if "pitch" in note_buffer and (
                    "duration_idx" not in note_buffer
                    or "instrument" not in note_buffer
                ):
                    if "duration_idx" not in note_buffer:
                        note_buffer["duration_idx"] = DEFAULT_DURATION_IDX
                    if "instrument" not in note_buffer:
                        note_buffer["instrument"] = DEFAULT_INSTRUMENT
                    if _flush_note_buffer(
                        note_buffer, instrument_map, measures, part_map,
                        current_bar or 1, current_tick, last_tick,
                        original_divisions,
                    ):
                        _advance_last_tick()
                        note_buffer.clear()

                # Round-9: count Bar tokens sequentially. Bar_N is the
                # token's vocab index (0-511), NOT a measure number. The
                # Arabic and MIDI training data both emit Bar tokens in
                # sequence, so the i-th Bar token = measure i+1 (1-based).
                bar_count += 1
                current_bar = bar_count
                current_tick = 0
                last_tick = 0

                # Round-7: ensure measure exists with attributes block on
                # measure 1 (so MuseScore renders the time signature, key,
                # clef).
                m = ET.SubElement(part_map[DEFAULT_PART_ID], 'measure',
                                  number=str(current_bar))
                measures[(DEFAULT_PART_ID, current_bar)] = m
                if current_bar == 1:
                    _ensure_attributes(m)

            elif ev.startswith(TIME_SIGNATURE_KEY + "_"):
                raw_val = ev[len(TIME_SIGNATURE_KEY)+1:]
                if "/" in raw_val:
                    beats, beat_type = map(int, raw_val.split("/"))
                    current_time_signature = (beats, beat_type)

            elif ev.startswith(POSITION_KEY + "_"):
                raw_val = ev[len(POSITION_KEY)+1:]
                current_tick = int(raw_val) * original_divisions // DEFAULT_POS_PER_QUARTER

            elif ev.startswith(PITCH_KEY + "_"):
                # Round-9: flush any orphaned pitch from the buffer
                # before storing the new one. This happens when the
                # model emits Pitches without a matching Duration /
                # Instrument. Without this flush, the old pitch sits
                # in the buffer forever and is silently dropped on
                # the next Bar token -- leaving the measure empty.
                if "pitch" in note_buffer and (
                    "duration_idx" not in note_buffer
                    or "instrument" not in note_buffer
                ):
                    if "duration_idx" not in note_buffer:
                        note_buffer["duration_idx"] = DEFAULT_DURATION_IDX
                    if "instrument" not in note_buffer:
                        note_buffer["instrument"] = DEFAULT_INSTRUMENT
                    if _flush_note_buffer(
                        note_buffer, instrument_map, measures, part_map,
                        current_bar or 1, current_tick, last_tick,
                        original_divisions,
                    ):
                        _advance_last_tick()
                        note_buffer.clear()
                note_buffer['pitch'] = ev[len(PITCH_KEY)+1:]

            elif ev.startswith(DURATION_KEY + "_"):
                note_buffer['duration_idx'] = int(ev[len(DURATION_KEY)+1:])

            elif ev.startswith(INSTRUMENT_KEY + "_"):
                note_buffer['instrument'] = ev[len(INSTRUMENT_KEY)+1:]

            if all(k in note_buffer for k in ('pitch', 'duration_idx', 'instrument')):
                # Round-7: if no Bar_ token has been seen yet, start
                # in measure 1 (1-indexed for MuseScore). This
                # happens for short generations or if the model
                # skipped the Bar_0 token.
                if current_bar is None:
                    if bar_offset is None:
                        bar_offset = -1  # so Bar_0 -> 1
                    current_bar = 1
                    current_tick = 0
                    last_tick = 0
                    m = ET.SubElement(part_map[DEFAULT_PART_ID], 'measure',
                                      number='1')
                    measures[(DEFAULT_PART_ID, 1)] = m
                    _ensure_attributes(m)

                if _flush_note_buffer(
                    note_buffer, instrument_map, measures, part_map,
                    current_bar, current_tick, last_tick,
                    original_divisions,
                ):
                    _advance_last_tick()
                    note_buffer.clear()

        # Flush any remaining orphaned pitch at end of stream so it
        # doesn't disappear (round-9).
        if "pitch" in note_buffer and (
            "duration_idx" not in note_buffer
            or "instrument" not in note_buffer
        ):
            if "duration_idx" not in note_buffer:
                note_buffer["duration_idx"] = DEFAULT_DURATION_IDX
            if "instrument" not in note_buffer:
                note_buffer["instrument"] = DEFAULT_INSTRUMENT
            if _flush_note_buffer(
                note_buffer, instrument_map, measures, part_map,
                current_bar or 1, current_tick, last_tick,
                original_divisions,
            ):
                note_buffer.clear()

        tree = ET.ElementTree(new_root)
        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    else:
        raise TypeError("Only supports REMI+ event lists")