# -*- coding: utf-8 -*-
"""
Round-10: MIDI loader for SAMAR.

Wraps MIDI files into MusicXML in-memory, then lets SAMARInputRepresentation
parse the result. This way the entire existing pipeline (events, precompute,
training) works without modification.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from .core import SAMARInputRepresentation
from .midi_to_xml import midi_to_musicxml_root


def load_midi_as_samar(midi_path):
    """Load a .mid file and return a SAMARInputRepresentation.

    The representation will have the same structure as one loaded from
    a .xml/.mxl file -- notes, events, time_sig, etc.

    Returns None if the file cannot be parsed.
    """
    root = midi_to_musicxml_root(midi_path)
    if root is None:
        return None

    # Write to a temp file so SAMARInputRepresentation can read it
    # from disk. The temp file is deleted immediately after parsing.
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.xml', delete=False) as f:
        # Add XML declaration manually since ET.tostring doesn't include it
        xml_bytes = ET.tostring(root, encoding='utf-8', xml_declaration=False)
        f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml_bytes)
        tmp_path = f.name

    try:
        samar_ir = SAMARInputRepresentation(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return samar_ir


def iter_midi_dataset(dataset_dir, recursive=True):
    """Yield (.mid path, SAMARInputRepresentation) tuples for all MIDI files.

    Skips files that fail to parse (with a printed warning).
    """
    for root, dirs, files in os.walk(dataset_dir):
        for fname in sorted(files):
            if fname.lower().endswith(('.mid', '.midi')):
                full_path = os.path.join(root, fname)
                samar_ir = load_midi_as_samar(full_path)
                if samar_ir is not None and len(samar_ir.notes) > 0:
                    yield full_path, samar_ir
        if not recursive:
            break