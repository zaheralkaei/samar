"""
Convert MIDI dataset (.mid files in data/midi_dataset/) to MusicXML
(.xml files in data/midi_as_xml/) so the r18 SAMAR pipeline can use them
unchanged.

Output preserves the composer-folder structure:
  data/midi_dataset/chopin/chop_*.mid
  -> data/midi_as_xml/chopin/chop_*.xml

This lets the r18 precompute (which reads from data/xml) work without
modification to the dataset.py or core.py code paths.

Usage:
  python -m samar.convert_midi_to_xml
  python -m samar.convert_midi_to_xml --src data/midi_dataset --dst data/midi_as_xml
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from tqdm import tqdm

from .midi_to_xml import midi_to_musicxml_root


def convert_one(midi_path: str, xml_path: str) -> bool:
    """Convert a single .mid to .xml. Returns True on success, False on failure."""
    try:
        root = midi_to_musicxml_root(midi_path)
        if root is None:
            return False
        os.makedirs(os.path.dirname(xml_path), exist_ok=True)
        ET.ElementTree(root).write(xml_path, xml_declaration=True, encoding='utf-8')
        return True
    except Exception as e:
        print(f"  [convert] FAIL {midi_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert MIDI dataset to MusicXML.")
    parser.add_argument("--src", default="data/midi_dataset",
                        help="Source MIDI directory (recursive .mid search)")
    parser.add_argument("--dst", default="data/midi_as_xml",
                        help="Destination XML directory (preserves folder structure)")
    args = parser.parse_args()

    src = args.src
    dst = args.dst

    if not os.path.isdir(src):
        print(f"ERROR: source dir does not exist: {src}")
        sys.exit(1)

    midi_files = []
    for root, dirs, files in os.walk(src):
        for fname in sorted(files):
            if fname.lower().endswith(('.mid', '.midi')):
                midi_files.append(os.path.join(root, fname))

    print(f"Found {len(midi_files)} MIDI files in {src}")
    print(f"Converting to: {dst}")
    print()

    n_ok = 0
    n_fail = 0
    for midi_path in tqdm(midi_files, desc="MIDI -> XML"):
        # Compute relative path to preserve folder structure
        rel = os.path.relpath(midi_path, src)
        # Replace .mid/.midi with .xml
        rel_xml = os.path.splitext(rel)[0] + '.xml'
        xml_path = os.path.join(dst, rel_xml)
        if convert_one(midi_path, xml_path):
            n_ok += 1
        else:
            n_fail += 1

    print()
    print(f"Done: {n_ok} converted, {n_fail} failed")
    print(f"Output: {dst}/")


if __name__ == "__main__":
    main()