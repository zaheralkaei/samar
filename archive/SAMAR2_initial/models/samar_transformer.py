# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 03:17:46 2025

@author: zaher
"""

# === File: models/samar_transformer.py ===
# Transformer implementation

import torch
import torch.nn as nn

class GroupEmbedding(nn.Module):
    def __init__(self, n_tokens, n_groups, out_dim, inner_dim=128):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_groups = n_groups
        self.inner_dim = inner_dim
        self.out_dim = out_dim

        self.embedding = nn.Embedding(n_tokens, inner_dim)
        self.proj = nn.Linear(n_groups * inner_dim, out_dim, bias=False)

    def forward(self, x):
        shape = x.shape
        emb = self.embedding(x)  # [batch_size, seq_len, inner_dim]
        emb = emb.view(*shape[:-1], -1)
        assert emb.size(-1) == self.n_groups * self.inner_dim, f"Expected {self.n_groups * self.inner_dim}, but got {emb.size(-1)}"
        return self.proj(emb)  # [batch_size, seq_len, out_dim]

class SamarTransformer(nn.Module):
    def __init__(self, d_model, n_head, num_layers, dim_feedforward, dropout, vocab_size, latent_dim, max_len=512):
        super(SamarTransformer, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.description_embedding = nn.Embedding(vocab_size, d_model)
        self.latent_embedding = nn.Linear(latent_dim, d_model)

        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.output_layer = nn.Linear(d_model, latent_dim)

    def forward(self, input_ids, latent=None, description=None, tgt=None):
        B, T = input_ids.size()
        src = self.token_embedding(input_ids)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        src = src + self.pos_embedding(pos_ids)

        if description is not None:
            B_d, T_d = description.size()
            desc_emb = self.description_embedding(description)
            desc_pos_ids = torch.arange(T_d, device=description.device).unsqueeze(0).expand(B_d, -1)
            desc_emb = desc_emb + self.pos_embedding(desc_pos_ids)
            src = torch.cat([desc_emb, src], dim=1)

        if latent is not None:
            tgt = self.latent_embedding(latent)
        else:
            tgt = torch.zeros_like(src)

        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        output = self.transformer(src, tgt)
        latent_output = self.output_layer(output)
        return latent_output

    def sample(self, start_tokens, latent=None, max_length=256, pad_id=None):
        self.eval()
        generated = start_tokens  # shape: [1, T]
        for _ in range(max_length - start_tokens.size(1)):
            logits = self(generated, latent)  # [seq_len, batch, dim]
            logits = logits.permute(1, 0, 2)  # [batch, seq_len, dim]
            next_token_logits = logits[:, -1, :]  # last token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # greedy decode
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
            "max_len": self.max_len
        }