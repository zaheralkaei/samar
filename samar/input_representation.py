# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:54:36 2025

@author: zaher
"""

# === File: samar/input_representation.py ===
# Re-export of the canonical input-representation from :mod:`samar.core`.
# Older codebases had a separate duplicate here; it now lives in core.py
# alongside the parser and metadata extractor so there is one source of
# truth. See audit finding #1.

from .core import Event, SAMARInputRepresentation

__all__ = ["Event", "SAMARInputRepresentation"]