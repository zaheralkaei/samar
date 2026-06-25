# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:52:37 2025

@author: zaher
"""

import os
from samar.chord_recognition_xml import XMLChord
from samar.core import MusicXMLParser

_HERE = os.path.dirname(os.path.abspath(__file__))
parser = MusicXMLParser(os.path.join(_HERE, "data", "2.xml"))
grouped = parser._group_notes_by_measure()

chord_estimator = XMLChord()
chords_per_bar = chord_estimator.extract(grouped)

for bar_idx, chord in chords_per_bar:
    print(f"Bar {bar_idx}: {chord}")