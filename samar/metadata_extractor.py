# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 20:12:53 2025

@author: zaher
"""

# === File: samar/metadata_extractor.py ===
# Extracts title/composer/instruments/time/key/tempo
import xml.etree.ElementTree as ET

# Extract metadata (e.g., title, composer, instruments, time signature, key, tempo) from a MusicXML file
def extract_metadata(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    metadata = {}

    # Extract title and subtitle from <credit> elements
    for credit in root.findall("credit"):
        credit_type = credit.find("credit-type")
        credit_words = credit.find("credit-words")
        if credit_type is not None and credit_words is not None:
            key = f"Description_{credit_type.text.capitalize()}"
            metadata[key] = credit_words.text.strip()

    # Extract composer name
    creator = root.find(".//creator[@type='composer']")
    if creator is not None:
        metadata["Description_Composer"] = creator.text.strip()

    # Extract instrument names from <score-part> entries
    for score_part in root.findall(".//score-part"):
        part_name = score_part.find("part-name")
        if part_name is not None:
            metadata.setdefault("Instruments", []).append(part_name.text.strip())

    # Extract time signature, key signature, and tempo from the first measure
    first_measure = root.find(".//part/measure")
    if first_measure is not None:
        # Time signature from <attributes>/<time>
        time = first_measure.find("attributes/time")
        if time is not None:
            beats = time.find("beats")
            beat_type = time.find("beat-type")
            if beats is not None and beat_type is not None:
                metadata["TimeSignature"] = f"{beats.text}/{beat_type.text}"

        # Key signature from <attributes>/<key>/<fifths>
        key = first_measure.find("attributes/key/fifths")
        if key is not None:
            metadata["KeySignature"] = int(key.text)

        # Tempo from <sound> element with tempo attribute
        sound = first_measure.find(".//sound")
        if sound is not None:
            tempo = sound.attrib.get("tempo")
            if tempo is not None:
                metadata["Tempo"] = int(float(tempo))

    return metadata
