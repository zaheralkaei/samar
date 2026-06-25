# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:27:39 2025

@author: zaher
"""

import os
from samar.reconstructor import reconstruct_musicxml_from_events

_HERE = os.path.dirname(os.path.abspath(__file__))

def load_remi_from_text(path):
    with open(path, "r", encoding="utf-8") as f:
        events = [line.strip() for line in f if line.strip()]
    print(f"📥 Loaded {len(events)} REMI+24 events from: {path}")
    return events

def main():
    # Step 1: Set paths (fixture file kept in tests/data/)
    data_dir = os.path.join(_HERE, "data")
    remi_txt = os.path.join(data_dir, "1.txt")
    output_xml = os.path.join(data_dir, "desert_echoes_structured.xml")

    # Step 2: Load REMI+24 events from text
    remi_events = load_remi_from_text(remi_txt)

    # Step 3: Reconstruct MusicXML
    reconstruct_musicxml_from_events(remi_events, output_xml)

    print(f"\n✅ Reconstruction complete. Output XML saved to: {output_xml}")

if __name__ == "__main__":
    main()
