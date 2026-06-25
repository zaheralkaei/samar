# -*- coding: utf-8 -*-
"""
Created on Sat May 10 02:35:35 2025

@author: zaher
"""

# test_debug_parsing_pipeline.py
# Zaher - Debug script to identify failure point in XML → events → tokens pipeline

import os
import traceback
# Resolve test fixtures relative to this file so the test runs from any CWD.
_HERE = os.path.dirname(os.path.abspath(__file__))
TEST_FOLDER = os.path.join(_HERE, "data")
VOCAB_PATH = os.path.join(os.path.dirname(_HERE), "samar_vocab.pkl")

# Use package-relative imports so this works whether run via ``python -m``
# or pytest with the project root on sys.path.
from samar.parser import MusicXMLParser
from samar.input_representation import SAMARInputRepresentation
from samar.tokenizer import SamarTokenizer

# Load tokenizer
try:
    tokenizer = SamarTokenizer.load(VOCAB_PATH)
    print("✅ Tokenizer loaded successfully.")
except Exception as e:
    print("❌ Failed to load tokenizer:", e)
    exit()

# Scan test files
files = [f for f in os.listdir(TEST_FOLDER) if f.endswith(".xml")]
print(f"Found {len(files)} XML files.")

# Test each file
for fname in files:
    path = os.path.join(TEST_FOLDER, fname)
    print("\n--- Testing:", fname)

    # Step 1: XML parsing
    try:
        parser = MusicXMLParser(path)
        print("✅ XML parsed: Measures =", len(parser.measures))
    except Exception as e:
        print("❌ Failed at MusicXMLParser:")
        traceback.print_exc()
        continue

    # Step 2: REMI+ event building
    try:
        ir = SAMARInputRepresentation(path)
        events = ir.get_event_sequence()
        print("✅ Event sequence length:", len(events))
        print("  First 10 events:", events[:10])
    except Exception as e:
        print("❌ Failed at SAMARInputRepresentation:")
        traceback.print_exc()
        continue

    # Step 3: Tokenization
    try:
        token_ids = tokenizer.encode(events)
        print("✅ Tokenization OK — First 10 IDs:", token_ids[:10])
    except Exception as e:
        print("❌ Failed at tokenizer.encode():")
        traceback.print_exc()
        continue
