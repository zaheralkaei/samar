# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 20:12:24 2025

@author: zaher
"""

# === File: samar/parser.py ===
# MusicXMLParser & SamarNote for note extraction.
#
# NOTE: This module used to define its own ``MusicXMLParser`` (a less-complete
# version that hardcoded 4/4 bar length and ignored per-bar time-signature
# changes). It now re-exports the canonical parser from :mod:`samar.core`,
# which is the version that handles time-signature changes correctly. See the
# audit finding (#1, ``docs/audit-2026-06-25.md``) for the rationale.

from .core import SamarNote, MusicXMLParser

__all__ = ["SamarNote", "MusicXMLParser"]