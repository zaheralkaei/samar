# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:05:57 2025

@author: zaher
"""

# test_reconstruct_from_remi.py

# test_full_roundtrip.py

from core import SAMARInputRepresentation
from reconstructor import reconstruct_musicxml_from_events

def save_remi_to_text(events, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(ev + "\n")
    print(f"✅ Saved REMI+24 event sequence to: {output_path}")

def load_remi_from_text(path):
    with open(path, "r", encoding="utf-8") as f:
        events = [line.strip() for line in f if line.strip()]
    print(f"📥 Loaded {len(events)} REMI+24 events from: {path}")
    return events

def main():
    # Step 1: Set paths
    input_xml = "1.xml"
    remi_txt = "1.txt"
    output_xml = "1re.xml"

    # Step 2: Parse MusicXML into REMI+24 events
    print(f"\n🔍 Parsing MusicXML: {input_xml}")
    rep = SAMARInputRepresentation(input_xml)
    remi_events = rep.get_event_sequence()

    # Step 3: Save REMI+24 events to file
    save_remi_to_text(remi_events, remi_txt)

    # Step 4: Load events from text
    loaded_events = load_remi_from_text(remi_txt)

    # Step 5: Reconstruct MusicXML
    reconstruct_musicxml_from_events(loaded_events, output_xml)

    print(f"\n✅ Roundtrip complete. Output XML saved to: {output_xml}")

if __name__ == "__main__":
    main()
