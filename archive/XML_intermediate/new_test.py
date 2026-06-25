# -*- coding: utf-8 -*-
"""
Created on Tue May 20 18:27:39 2025

@author: zaher
"""

from reconstructor import reconstruct_musicxml_from_events

def load_remi_from_text(path):
    with open(path, "r", encoding="utf-8") as f:
        events = [line.strip() for line in f if line.strip()]
    print(f"📥 Loaded {len(events)} REMI+24 events from: {path}")
    return events

def main():
    # Step 1: Set paths
    remi_txt = "desert_echoes_structured.txt"
    output_xml = "desert_echoes_structured.xml"

    # Step 2: Load REMI+24 events from text
    remi_events = load_remi_from_text(remi_txt)

    # Step 3: Reconstruct MusicXML
    reconstruct_musicxml_from_events(remi_events, output_xml)

    print(f"\n✅ Reconstruction complete. Output XML saved to: {output_xml}")

if __name__ == "__main__":
    main()
