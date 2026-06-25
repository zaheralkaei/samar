# -*- coding: utf-8 -*-
"""
Created on Mon May 19 12:29:06 2025

@author: zaher
"""
# samar/model.py

# === File: model.py ===

import torch
import torch.nn as nn

class SamarTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=4, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.pad_idx = pad_idx

    def forward(self, x):
        padding_mask = x.eq(self.pad_idx)
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        attn_mask = torch.triu(torch.full((x.size(0), x.size(0)), float('-inf')), diagonal=1).to(x.device)
        out = self.decoder(x, x, tgt_mask=attn_mask, tgt_key_padding_mask=padding_mask)
        return self.fc(out.permute(1, 0, 2))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)