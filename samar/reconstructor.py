# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:04:46 2025

@author: zaher
"""

# reconstructor.py


import xml.etree.ElementTree as ET
import copy
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

            # Round-11 audit: clamp octave to valid MusicXML range [0..8].
            # Out-of-range octaves are physically valid (some pitches
            # cross MIDI's 0-127 range) but produce warnings or are
            # rejected by strict parsers (MuseScore). The pitch token
            # comes from the model's 24-EDO pitch vocabulary (0-191);
            # a value like Pitch_24EDO_0 maps to MIDI 0 (C-1) which is
            # outside the spec.
            octave = max(0, min(8, octave))
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
        # Round-9/11: bar_count tracks the sequential bar number. Earlier
        # rounds (round-7) tried to remap Bar_N (vocab index) to a
        # 1-based measure number via a ``bar_offset``, but that broke
        # when the model emitted Bar tokens out of order (e.g. Bar_397
        # then Bar_195). Round-9 fixed this by counting Bar tokens
        # sequentially (first Bar_ -> measure 1, etc.). Round-11 audit
        # removed a dead ``bar_offset`` reference in the "no Bar seen
        # yet" branch -- the assignment was a no-op NameError waiting
        # to fire for any generation that omitted the initial Bar_0
        # token.
        bar_count = 0

        def _advance_last_tick():
            """Update last_tick based on note_buffer after a note is flushed.
            Also checks for measure overflow and auto-advances if needed.
            """
            nonlocal last_tick, current_bar, current_tick, bar_count
            idx = min(note_buffer.get('duration_idx', 0), len(DEFAULT_DURATION_BINS) - 1)
            dur_pos = DEFAULT_DURATION_BINS[idx]
            divisions = int(round(dur_pos * original_divisions / DEFAULT_POS_PER_QUARTER))
            new_last_tick = current_tick + divisions

            # Round-10: auto-advance if note overflows measure capacity.
            # The model doesn't track measure capacity, so notes often
            # exceed 4/4 (1920 ticks). We trim the note to fit, then
            # create a new measure with the overflow.
            beats, beat_type = current_time_signature or (4, 4)
            quarters_per_bar = 4 * beats / beat_type
            capacity = int(original_divisions * quarters_per_bar)

            if new_last_tick > capacity:
                # Trim the just-flushed note
                meas = measures.get((DEFAULT_PART_ID, current_bar))
                if meas is not None:
                    notes_in_m = meas.findall('note')
                    if notes_in_m:
                        last_note_el = notes_in_m[-1]
                        dur_el = last_note_el.find('duration')
                        if dur_el is not None:
                            orig_dur = int(dur_el.text)
                            keep = capacity - current_tick
                            overflow = orig_dur - keep
                            if keep > 0 and overflow > 0:
                                # Trim and split
                                dur_el.text = str(keep)
                                # Create next measure
                                bar_count += 1
                                current_bar = bar_count
                                current_tick = 0
                                last_tick = overflow
                                m_new = ET.SubElement(
                                    part_map[DEFAULT_PART_ID],
                                    'measure',
                                    number=str(current_bar)
                                )
                                measures[(DEFAULT_PART_ID, current_bar)] = m_new
                                # Move overflow portion to new measure
                                new_note = ET.SubElement(m_new, 'note')
                                for child in last_note_el:
                                    # Round-11 audit: skip duration (set
                                    # below) AND voice (we add a single
                                    # fresh one). Copying the original
                                    # voice produces duplicate <voice>
                                    # elements when this branch fires
                                    # repeatedly for the same overflow.
                                    if child.tag in ('duration', 'voice'):
                                        continue
                                    new_note.append(copy.deepcopy(child))
                                ET.SubElement(new_note, 'duration').text = str(overflow)
                                ET.SubElement(new_note, 'voice').text = '1'
                                return
                            elif keep <= 0:
                                # Note doesn't fit at all, drop it from
                                # current measure and recreate in next.
                                meas.remove(last_note_el)
                                bar_count += 1
                                current_bar = bar_count
                                current_tick = 0
                                last_tick = 0
                                m_new = ET.SubElement(
                                    part_map[DEFAULT_PART_ID],
                                    'measure',
                                    number=str(current_bar)
                                )
                                measures[(DEFAULT_PART_ID, current_bar)] = m_new
                                # Add the note to the new measure
                                new_note = ET.SubElement(m_new, 'note')
                                for child in last_note_el:
                                    # Round-11 audit: skip duration and
                                    # voice (see comment above).
                                    if child.tag in ('duration', 'voice'):
                                        continue
                                    new_note.append(copy.deepcopy(child))
                                ET.SubElement(new_note, 'duration').text = str(orig_dur)
                                ET.SubElement(new_note, 'voice').text = '1'
                                last_tick = orig_dur
                                return
            last_tick = new_last_tick

        def _measure_capacity():
            """Return the current measure's tick capacity (4/4 default = 1920)."""
            beats, beat_type = current_time_signature or (4, 4)
            quarters_per_bar = 4 * beats / beat_type
            return int(original_divisions * quarters_per_bar)

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
                # Round-10: cap position to within measure capacity
                # (positions 0-47 = 0-1920 ticks for 4/4). The model
                # sometimes emits positions > 47 (e.g. Position_140)
                # which would create a huge forward. Cap to capacity.
                pos = int(raw_val)
                capacity = int(original_divisions * 4)  # 4/4 default
                max_pos = (capacity * DEFAULT_POS_PER_QUARTER) // original_divisions
                pos = min(pos, max_pos)
                current_tick = pos * original_divisions // DEFAULT_POS_PER_QUARTER

                # If the new position would exceed current measure
                # capacity (last_tick + forward), auto-advance to a new
                # measure and put the overflow in the new measure's
                # forward.
                beats, beat_type = current_time_signature or (4, 4)
                quarters_per_bar = 4 * beats / beat_type
                full_capacity = int(original_divisions * quarters_per_bar)
                if last_tick + (current_tick - last_tick) > full_capacity:
                    # The forward that will be created would exceed
                    # the measure. Skip ahead to next measure and
                    # reduce current_tick accordingly.
                    overflow = last_tick + (current_tick - last_tick) - full_capacity
                    if overflow > 0:
                        bar_count += 1
                        current_bar = bar_count
                        last_tick = 0
                        current_tick = overflow
                        m = ET.SubElement(part_map[DEFAULT_PART_ID], 'measure',
                                          number=str(current_bar))
                        measures[(DEFAULT_PART_ID, current_bar)] = m
                        # Emit a forward for the overflow in the new measure
                        fwd = ET.SubElement(m, 'forward')
                        ET.SubElement(fwd, 'duration').text = str(overflow)

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
                #
                # Round-11 audit: removed the dead `bar_offset`
                # reference here. The old code read `if bar_offset
                # is None` then assigned `bar_offset = -1`, but
                # `bar_offset` was never defined in this scope and
                # `current_bar = 1` was set unconditionally anyway,
                # so the branch was a no-op NameError waiting to fire
                # for any generation that omitted the initial
                # Bar_0 token. See commit log round-11.
                if current_bar is None:
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

        # Round-10: post-processing step. Walk through all measures and
        # split any that overflow capacity (MuseScore rejects these as
        # corrupt). We iterate to a fixed point: each split might create
        # new overflowing measures that need to be split again.
        for part in list(tree.iter('part')):
            # Iterate to fixed point: keep splitting until no overflows
            changed = True
            while changed:
                changed = False
                for meas in list(part.findall('measure')):
                    total = 0
                    for child in meas:
                        if child.tag == 'note':
                            total += int(child.find('duration').text)
                        elif child.tag == 'forward':
                            total += int(child.find('duration').text)
                    if total <= 1920:
                        continue
                    # Split: walk children, accumulate, start new measure
                    # when would exceed capacity
                    capacity = 1920
                    current_total = 0
                    split_count = 0
                    for child in list(meas):
                        child_dur = 0
                        if child.tag == 'note':
                            child_dur = int(child.find('duration').text)
                        elif child.tag == 'forward':
                            child_dur = int(child.find('duration').text
                            )
                        # Even a single child > capacity should be
                        # trimmed to fit and the overflow moved to next measure
                        if child_dur > capacity:
                            # Trim this child
                            dur_el = child.find('duration')
                            if dur_el is not None:
                                dur_el.text = str(capacity)
                            # Move overflow to new measure
                            overflow = child_dur - capacity
                            new_meas = ET.Element('measure',
                                                  number=str(int(meas.get('number')) + 1 + split_count))
                            split_count += 1
                            new_note = ET.Element(child.tag)
                            if child.tag == 'note':
                                # Round-11 audit: skip <voice> when copying.
                                # The original note already has <voice>
                                # (added by _flush_note_buffer); appending
                                # a second one produces malformed XML when
                                # the same overflow fragment is split again
                                # in a later iteration. MusicXML allows at
                                # most one <voice> per <note>.
                                for c2 in child:
                                    if c2.tag in ('duration', 'voice'):
                                        continue
                                    new_note.append(copy.deepcopy(c2))
                                ET.SubElement(new_note, 'duration').text = str(overflow)
                                ET.SubElement(new_note, 'voice').text = '1'
                            else:
                                # forward
                                ET.SubElement(new_note, 'duration').text = str(overflow)
                            new_meas.append(new_note)
                            part.append(new_meas)
                            current_total = capacity  # current is now full
                            changed = True
                            continue
                        if current_total + child_dur > capacity and current_total > 0:
                            new_meas = ET.Element('measure',
                                                  number=str(int(meas.get('number')) + 1 + split_count))
                            split_count += 1
                            meas.remove(child)
                            new_meas.append(child)
                            part.append(new_meas)
                            current_total = child_dur
                            changed = True
                        else:
                            current_total += child_dur
                    # Renumber all measures in this part sequentially
                    if split_count > 0:
                        all_measures_in_part = part.findall('measure')
                        for i, m in enumerate(all_measures_in_part):
                            m.set('number', str(i + 1))
                        # Also ensure measure 1 has attributes (only after renumbering)
                        if all_measures_in_part:
                            first = all_measures_in_part[0]
                            if first.find('attributes') is None:
                                _ensure_attributes(first)

        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    else:
        raise TypeError("Only supports REMI+ event lists")