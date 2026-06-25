# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:32:22 2025

@author: zaher
"""

# samar/collate.py

import torch
from torch.nn.utils.rnn import pad_sequence

def pad_collate(batch, pad_idx=0):
    xs, ys = zip(*batch)
    x_pad = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    y_pad = pad_sequence(ys, batch_first=True, padding_value=pad_idx)
    return x_pad, y_pad
