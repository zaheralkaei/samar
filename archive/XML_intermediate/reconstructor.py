# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:04:46 2025

@author: zaher
"""

# reconstructor.py


import xml.etree.ElementTree as ET
from constants import (
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

def reconstruct_musicxml_from_events(input_or_events, output_xml_path: str):
    if isinstance(input_or_events, list):
        events = input_or_events
        new_root = ET.Element('score-partwise', version="4.0")
        part_list_elem = ET.SubElement(new_root, "part-list")

        instruments = []
        time_signatures = {}

        for ev in events:
            if ev.startswith(f"{INSTRUMENT_KEY}_"):
                inst = ev.split("_", 1)[1].strip().lower()
                if inst not in instruments:
                    instruments.append(inst)
            elif ev.startswith(f"{TIME_SIGNATURE_KEY}_"):
                current_ts = ev.split("_", 1)[1].strip()
                if "/" in current_ts:
                    beats, beat_type = map(int, current_ts.split("/"))
                    time_signatures[beats, beat_type] = True  # just to track usage

        instrument_map = {}
        part_map = {}
        for i, inst in enumerate(instruments):
            part_id = f"P{i+1}"
            instrument_map[inst] = part_id
            score_part = ET.SubElement(part_list_elem, "score-part", id=part_id)
            ET.SubElement(score_part, "part-name").text = inst.capitalize()
            part_elem = ET.SubElement(new_root, "part", id=part_id)
            part_map[part_id] = part_elem

        if not instrument_map:
            instrument_map = {"generated": "P1"}
            score_part = ET.SubElement(part_list_elem, "score-part", id="P1")
            ET.SubElement(score_part, "part-name").text = "Generated"
            part_elem = ET.SubElement(new_root, "part", id="P1")
            part_map = {"P1": part_elem}

        original_divisions = DEFAULT_RESOLUTION
        measures = {}
        note_buffer = {}
        current_bar = None
        current_tick = 0
        last_tick = 0
        current_time_signature = (4, 4)

        for ev in events:
            if ev.startswith(BAR_KEY + "_"):
                name = BAR_KEY
                raw_val = ev[len(BAR_KEY)+1:]
                current_bar = int(raw_val)
                current_tick = 0
                for pid, part_elem in part_map.items():
                    m = ET.SubElement(part_elem, 'measure', number=str(current_bar))
                    measures[(pid, current_bar)] = m
                    if current_bar == 1:
                        attr = ET.SubElement(m, 'attributes')
                        ET.SubElement(attr, 'divisions').text = str(original_divisions)
                        time = ET.SubElement(attr, 'time')
                        ET.SubElement(time, 'beats').text = str(current_time_signature[0])
                        ET.SubElement(time, 'beat-type').text = str(current_time_signature[1])
                        key = ET.SubElement(attr, 'key')
                        ET.SubElement(key, 'fifths').text = '0'
                        clef = ET.SubElement(attr, 'clef')
                        ET.SubElement(clef, 'sign').text = 'G'
                        ET.SubElement(clef, 'line').text = '2'

            elif ev.startswith(TIME_SIGNATURE_KEY + "_"):
                raw_val = ev[len(TIME_SIGNATURE_KEY)+1:]
                if "/" in raw_val:
                    beats, beat_type = map(int, raw_val.split("/"))
                    current_time_signature = (beats, beat_type)

            elif ev.startswith(POSITION_KEY + "_"):
                raw_val = ev[len(POSITION_KEY)+1:]
                current_tick = int(raw_val) * original_divisions // DEFAULT_POS_PER_QUARTER

            elif ev.startswith(PITCH_KEY + "_"):
                note_buffer['pitch'] = ev[len(PITCH_KEY)+1:]

            elif ev.startswith(DURATION_KEY + "_"):
                note_buffer['duration_idx'] = int(ev[len(DURATION_KEY)+1:])

            elif ev.startswith(INSTRUMENT_KEY + "_"):
                note_buffer['instrument'] = ev[len(INSTRUMENT_KEY)+1:]

            if all(k in note_buffer for k in ['pitch', 'duration_idx', 'instrument']):
                inst_name = note_buffer['instrument'].strip().lower()
                pid = instrument_map.get(inst_name, "P1")
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
                    except:
                        ET.SubElement(note_el, 'rest')

                idx = min(note_buffer.get('duration_idx', 0), len(DEFAULT_DURATION_BINS) - 1)
                dur_pos = DEFAULT_DURATION_BINS[idx]
                divisions = int(round(dur_pos * original_divisions / DEFAULT_POS_PER_QUARTER))
                ET.SubElement(note_el, 'duration').text = str(divisions)
                ET.SubElement(note_el, 'voice').text = '1'

                last_tick = current_tick + divisions
                note_buffer.clear()

        tree = ET.ElementTree(new_root)
        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    else:
        raise TypeError("Only supports REMI+ event lists")


