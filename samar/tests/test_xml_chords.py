# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:52:37 2025

@author: zaher
"""

from chord_recognition_xml import XMLChord
from core import MusicXMLParser

parser = MusicXMLParser("2.xml")
grouped = parser._group_notes_by_measure()

chord_estimator = XMLChord()
chords_per_bar = chord_estimator.extract(grouped)

for bar_idx, chord in chords_per_bar:
    print(f"Bar {bar_idx}: {chord}")