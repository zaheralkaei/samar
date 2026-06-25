# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:26:48 2025

@author: zaher
"""

import os
from core import SAMARInputRepresentation

os.makedirs("data/remi_txt", exist_ok=True)

for fname in os.listdir("xml_data"):
    if fname.endswith(".xml"):
        path = os.path.join("xml_data", fname)
        rep = SAMARInputRepresentation(path)
        events = rep.get_event_sequence()
        out_path = f"data/remi_txt/{fname.replace('.xml', '.txt')}"
        with open(out_path, "w", encoding="utf-8") as f:
            for e in events:
                f.write(e + "\n")
        print(f"✅ Converted {fname} → {out_path}")
