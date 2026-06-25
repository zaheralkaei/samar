# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:57:07 2025

@author: zaher
"""

# === File: samar/tokenize_musicxml.py ===

from input_representation import SAMARInputRepresentation
from tokenizer import SamarTokenizer

# Load MusicXML file
xml_file_path = "Hello.xml"  

# Step 1: Parse to REMI+ events
samar_input = SAMARInputRepresentation(xml_file_path)
events = samar_input.get_event_sequence()

print("=== REMI+ EVENTS ===")
for e in events:
    print(e)

# Step 2: Tokenize
print("\n=== TOKENIZED ===")
tokenizer = SamarTokenizer()
tokens = tokenizer.encode(events)

print(tokens)

# Optionally: decode back to event strings
print("\n=== DECODED BACK ===")
print(tokenizer.decode(tokens))

# Save tokenizer
tokenizer.save("samar_vocab.pkl")
