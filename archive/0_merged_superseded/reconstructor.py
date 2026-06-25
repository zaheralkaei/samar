# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:21:24 2025

@author: zaher
"""

# === File: reconstructor.py ===

# -*- coding: utf-8 -*-
"""
Reconstructs MusicXML from REMI+ event sequences.
"""

import xml.etree.ElementTree as ET
import copy
from core import SAMARInputRepresentation
from constants import (
    DEFAULT_RESOLUTION,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_DURATION_BINS,
    BAR_KEY,
    POSITION_KEY,
    PITCH_KEY,
    DURATION_KEY,
    INSTRUMENT_KEY
)

def reconstruct_musicxml_from_events(input_or_events, output_xml_path: str):
    if isinstance(input_or_events, list):
        events = input_or_events
        new_root = ET.Element('score-partwise', version="4.0")
        part_list = ET.SubElement(new_root, "part-list")
        score_part = ET.SubElement(part_list, "score-part", id="P1")
        ET.SubElement(score_part, "part-name").text = "Generated"
        part_elem = ET.SubElement(new_root, "part", id="P1")
        part_map = {"P1": part_elem}
        instrument_map = {"Generated": "P1"}
        original_divisions = 480
    else:
        rep = SAMARInputRepresentation(input_or_events)
        events = rep.get_event_sequence()
        original_tree = ET.parse(input_or_events)
        original_root = original_tree.getroot()
        part_list = original_root.find('part-list')
        divisions_elem = original_root.find('.//divisions')
        original_divisions = int(divisions_elem.text) if divisions_elem is not None else 480

        new_root = ET.Element('score-partwise', version=original_root.get('version', '4.0'))
        new_root.append(copy.deepcopy(part_list))

        part_map = {}
        for score_part in part_list.findall('score-part'):
            pid = score_part.get('id')
            part_map[pid] = ET.SubElement(new_root, 'part', id=pid)

        instrument_map = {
            sp.findtext('part-name'): sp.get('id')
            for sp in part_list.findall('score-part')
        }

    measures = {}
    note_buffer = {}
    current_bar = None
    current_tick = 0
    last_tick = 0
    missing_notes = 0

    for ev in events:
        if "_" not in ev:
            continue
        name, raw_val = ev.rsplit("_", 1)

        if name == BAR_KEY:
            current_bar = int(raw_val)
            current_tick = 0
            for pid, part_elem in part_map.items():
                m = ET.SubElement(part_elem, 'measure', number=str(current_bar))
                measures[(pid, current_bar)] = m

                if current_bar == 1:
                    attr = ET.SubElement(m, 'attributes')
                    ET.SubElement(attr, 'divisions').text = str(original_divisions)
                    time = ET.SubElement(attr, 'time')
                    ET.SubElement(time, 'beats').text = '4'
                    ET.SubElement(time, 'beat-type').text = '4'
                    key = ET.SubElement(attr, 'key')
                    ET.SubElement(key, 'fifths').text = '0'
                    clef = ET.SubElement(attr, 'clef')
                    ET.SubElement(clef, 'sign').text = 'G'
                    ET.SubElement(clef, 'line').text = '2'

        elif name == POSITION_KEY:
            # Convert to tick
            current_tick = int(raw_val) * original_divisions // DEFAULT_POS_PER_QUARTER

        elif name == PITCH_KEY:
            note_buffer['pitch'] = raw_val

        elif name == DURATION_KEY:
            note_buffer['duration_idx'] = int(raw_val)

        elif name == INSTRUMENT_KEY:
            note_buffer['instrument'] = raw_val

        if all(k in note_buffer for k in ['pitch', 'duration_idx']):
            if 'instrument' not in note_buffer:
                note_buffer['instrument'] = 'Generated'

            pid = instrument_map.get(note_buffer['instrument'], "P1")
            meas = measures.get((pid, current_bar))
            if meas is None:
                missing_notes += 1
                note_buffer.clear()
                continue

            # Advance time if needed
            if current_tick > last_tick:
                forward = ET.SubElement(meas, 'forward')
                ET.SubElement(forward, 'duration').text = str(current_tick - last_tick)

            note_el = ET.SubElement(meas, 'note')
            pitch_val = note_buffer.get('pitch')

            if pitch_val in [None, 'Rest', 'None']:
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
                except Exception as e:
                    print(f"⚠️ Invalid pitch '{pitch_val}': {e}")
                    ET.SubElement(note_el, 'rest')

            idx = note_buffer.get('duration_idx', 0)
            dur_pos = DEFAULT_DURATION_BINS[idx]
            divisions = int(round(dur_pos * original_divisions / DEFAULT_POS_PER_QUARTER))
            ET.SubElement(note_el, 'duration').text = str(divisions)
            ET.SubElement(note_el, 'voice').text = '1'

            last_tick = current_tick + divisions
            note_buffer.clear()

    tree = ET.ElementTree(new_root)
    tree.write(output_xml_path, encoding='UTF-8', xml_declaration=True)
    print(f"✅ Saved MusicXML to {output_xml_path}")
    if missing_notes > 0:
        print(f"⚠️ Skipped {missing_notes} notes due to missing measure or fields.")




if __name__ == '__main__':
    input_path = "test_files/sample.xml"       # Replace with your file
    output_path = "reconstructions/output.xml" # Replace with desired output path
    reconstruct_musicxml_from_events(input_path, output_path)
