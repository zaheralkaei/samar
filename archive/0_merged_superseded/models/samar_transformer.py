# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 03:17:46 2025

@author: zaher
"""

# === File: models/samar_transformer.py ===
# Transformer implementation for FIGARO-style description-to-sequence learning

import torch
import torch.nn as nn

class SamarTransformer(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dim_feedforward, dropout, vocab_size, latent_dim=None, max_len=512, description_vocab_size=32):
        super(SamarTransformer, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.description_vocab_size = description_vocab_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.description_embedding = nn.Embedding(description_vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, description=None):
        B, T = input_ids.size()
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos_ids)

        if description is not None:
            desc_emb = self.description_embedding(description)  # (B, D)
            desc_emb = desc_emb.unsqueeze(1)  # (B, 1, D)
            x = x + desc_emb  # Broadcast addition

        encoded = self.transformer_encoder(x)  # (B, T, D)
        logits = self.output_layer(encoded)    # (B, T, vocab_size)
        return logits

    def sample(self, start_tokens, description=None, max_length=256, pad_id=None):
        self.eval()
        generated = start_tokens
        for _ in range(max_length - start_tokens.size(1)):
            logits = self(generated, description=description)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if pad_id is not None and next_token.item() == pad_id:
                break
        return generated

    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_head": self.n_head,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "vocab_size": self.vocab_size,
            "latent_dim": self.latent_dim,
            "max_len": self.max_len,
            "description_vocab_size": self.description_vocab_size
        }