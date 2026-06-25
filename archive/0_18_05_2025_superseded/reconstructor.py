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
    print("\n🧪 Starting reconstruction...")

    if isinstance(input_or_events, list):
        print("📥 Received REMI+ event list directly.")

        events = input_or_events
        new_root = ET.Element('score-partwise', version="4.0")
        part_list_elem = ET.SubElement(new_root, "part-list")

        # ✅ Build instrument map dynamically
        instruments = []
        for ev in events:
            if ev.startswith(f"{INSTRUMENT_KEY}_"):
                inst = ev.split("_", 1)[1].strip().lower()
                if inst not in instruments:
                    instruments.append(inst)

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

        original_divisions = 480

    else:
        raise TypeError("❌ This function only supports direct REMI+ event lists for reconstruction.")

    # --- Reconstruction Loop ---
    measures = {}
    note_buffer = {}
    current_bar = None
    current_tick = 0
    last_tick = 0
    missing_notes = 0

    for ev in input_or_events:
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
            current_tick = int(raw_val) * original_divisions // DEFAULT_POS_PER_QUARTER

        elif name == PITCH_KEY:
            note_buffer['pitch'] = raw_val

        elif name == DURATION_KEY:
            note_buffer['duration_idx'] = int(raw_val)

        elif name == INSTRUMENT_KEY:
            note_buffer['instrument'] = raw_val

        if all(k in note_buffer for k in ['pitch', 'duration_idx', 'instrument']):
            inst_name = note_buffer['instrument'].strip().lower()
            pid = instrument_map.get(inst_name)

            if pid is None:
                print(f"⚠️ Instrument '{inst_name}' not mapped, assigning to 'P1'")
                pid = "P1"

            meas = measures.get((pid, current_bar))
            if meas is None:
                missing_notes += 1
                note_buffer.clear()
                continue

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

    ET.ElementTree(new_root).write(output_xml_path, encoding='UTF-8', xml_declaration=True)
    print(f"\n✅ Saved reconstructed MusicXML to: {output_xml_path}")
    if missing_notes > 0:
        print(f"⚠️ Skipped {missing_notes} notes due to missing measure or fields.")