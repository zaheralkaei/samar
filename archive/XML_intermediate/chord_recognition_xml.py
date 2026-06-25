# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:52:10 2025

@author: zaher
"""

import numpy as np
from collections import Counter

class XMLChord:
    def __init__(self):
        self.PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.CHORD_MAPS = {'maj': [0, 4],
                           'min': [0, 3],
                           'dim': [0, 3, 6],
                           'aug': [0, 4, 8],
                           'dom7': [0, 4, 10],
                           'maj7': [0, 4, 11],
                           'min7': [0, 3, 10]}
        self.CHORD_INSIDERS = {'maj': [7],
                               'min': [7],
                               'dim': [9],
                               'aug': [],
                               'dom7': [7],
                               'maj7': [7],
                               'min7': [7]}
        self.CHORD_OUTSIDERS_1 = {'maj': [2, 5, 9],
                                  'min': [2, 5, 8],
                                  'dim': [2, 5, 10],
                                  'aug': [2, 5, 9],
                                  'dom7': [2, 5, 9],
                                  'maj7': [2, 5, 9],
                                  'min7': [2, 5, 8]}
        self.CHORD_OUTSIDERS_2 = {'maj': [1, 3, 6, 8, 10, 11],
                                  'min': [1, 4, 6, 9, 11],
                                  'dim': [1, 4, 7, 8, 11],
                                  'aug': [1, 3, 6, 7, 10],
                                  'dom7': [1, 3, 6, 8, 11],
                                  'maj7': [1, 3, 6, 8, 10],
                                  'min7': [1, 4, 6, 9, 11]}

    def pitch_class_histogram(self, notes):
        pc_hist = np.zeros(12, dtype=int)
        for note in notes:
            if note.is_rest:
                continue
            midi_pitch = note.to_24edo_pitch() // 2  # Convert 24EDO → 12EDO MIDI
            pc = int(midi_pitch % 12)
            pc_hist[pc] += 1
        return pc_hist

    def sequencing(self, chroma):
        candidates = {}
        for index in range(len(chroma)):
            if chroma[index]:
                root_note = index
                _chroma = np.roll(chroma, -root_note)
                sequence = np.where(_chroma == 1)[0]
                candidates[root_note] = list(sequence)
        return candidates

    def scoring(self, candidates):
        scores = {}
        qualities = {}
        for root_note, sequence in candidates.items():
            if 3 not in sequence and 4 not in sequence:
                scores[root_note] = -100
                qualities[root_note] = 'None'
            elif 3 in sequence and 4 in sequence:
                scores[root_note] = -100
                qualities[root_note] = 'None'
            else:
                # decide quality
                if 3 in sequence:
                    quality = 'dim' if 6 in sequence else 'min7' if 10 in sequence else 'min'
                elif 4 in sequence:
                    quality = 'aug' if 8 in sequence else 'dom7' if 10 in sequence else 'maj7' if 11 in sequence else 'maj'
                else:
                    quality = 'None'

                maps = self.CHORD_MAPS.get(quality, [])
                _notes = [n for n in sequence if n not in maps]
                score = 0
                for n in _notes:
                    if n in self.CHORD_OUTSIDERS_1.get(quality, []):
                        score -= 1
                    elif n in self.CHORD_OUTSIDERS_2.get(quality, []):
                        score -= 2
                    elif n in self.CHORD_INSIDERS.get(quality, []):
                        score += 10
                scores[root_note] = score
                qualities[root_note] = quality
        return scores, qualities

    def find_chord(self, pc_hist):
        chroma = (pc_hist > 0).astype(int)
        if np.sum(chroma) == 0:
            return 'N:N'

        candidates = self.sequencing(chroma)
        scores, qualities = self.scoring(candidates)

        sorted_notes = np.nonzero(chroma)[0]
        bass_note = sorted_notes[0] if len(sorted_notes) > 0 else 0

        top_score = max(scores.values())
        top_roots = [k for k, v in scores.items() if v == top_score]
        root_note = top_roots[0]
        if len(top_roots) > 1 and bass_note in top_roots:
            root_note = bass_note

        quality = qualities[root_note]
        root = self.PITCH_CLASSES[root_note]
        bass = self.PITCH_CLASSES[bass_note]

        return f"{root}:{quality}/{bass}" if root != bass else f"{root}:{quality}"

    def extract(self, grouped_notes_by_bar):
        """
        Input: dict of bar_index -> list of SamarNote
        Output: list of (bar_index, chord_label)
        """
        chords = []
        for bar_idx, notes in sorted(grouped_notes_by_bar.items()):
            pc_hist = self.pitch_class_histogram(notes)
            chord = self.find_chord(pc_hist)
            chords.append((bar_idx, chord))
        return chords
