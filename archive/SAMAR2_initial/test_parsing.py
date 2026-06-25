# -*- coding: utf-8 -*-
"""
Created on Sat May 10 02:35:35 2025

@author: zaher
"""

# test_debug_parsing_pipeline.py
# Zaher - Debug script to identify failure point in XML → events → tokens pipeline

import os
import traceback
from parser import MusicXMLParser
from input_representation import SAMARInputRepresentation
from tokenizer import SamarTokenizer

# Path to your test XML folder
TEST_FOLDER = "./test_data"
VOCAB_PATH = "samar_vocab.pkl"

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
