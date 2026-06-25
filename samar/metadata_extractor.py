# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:35:35 2025

@author: zaher
"""

# === File: samar/metadata_extractor.py ===
# Re-export of the canonical metadata extractor from :mod:`samar.core`.
# Older codebases had a separate duplicate implementation here; it now lives
# in core.py alongside the parser that produces it. See audit finding #1.

from .core import extract_metadata

__all__ = ["extract_metadata"]