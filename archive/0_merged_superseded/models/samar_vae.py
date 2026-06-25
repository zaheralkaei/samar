# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 02:29:25 2025

@author: zaher
"""
# === File: models/samar_vae.py ===
# VQ-VAE implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from vocab_and_tokenizer import SamarVocab


class VqEmbeddingEMA(nn.Module):
    def __init__(self, n_embed, embed_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Parameter(torch.randn(n_embed, embed_dim))
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", self.embedding.data.clone())

    def forward(self, z):
        flat_z = z.view(-1, self.embed_dim)
        distances = (
            (flat_z ** 2).sum(1, keepdim=True)
            - 2 * flat_z @ self.embedding.T
            + (self.embedding ** 2).sum(1)
        )
        indices = distances.argmin(1)
        z_q = self.embedding[indices].view_as(z)

        if self.training:
            encodings = F.one_hot(indices, self.n_embed).float()
            cluster_size = encodings.sum(0)
            embed_sum = encodings.T @ flat_z

            self.cluster_size.data.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = ((self.cluster_size + self.eps) / (n + self.n_embed * self.eps)) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)

        return z_q, indices.view(z.shape[:-1])


class SamarVQVAE(nn.Module):
    def __init__(self, d_model=256, n_embed=1024, vocab_size=None, lr=3e-4, pad_id=0):
        super().__init__()
        self.d_model = d_model
        self.n_embed = n_embed
        self.lr = lr
        self.pad_id = pad_id

        self.vocab = SamarVocab()
        self.vocab_size = vocab_size if vocab_size is not None else len(self.vocab)

        self.encoder = nn.Sequential(
            nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

        self.quantizer = VqEmbeddingEMA(n_embed, d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.vocab_size)
        )

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def encode(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.quantizer(z_e)
        return indices, z_q

    def decode(self, z_q):
        return self.decoder(z_q)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, _ = self.quantizer(z_e)
        return self.decoder(z_q)

    def compute_loss(self, input_ids, labels):
        z_e = self.encoder(input_ids)
        z_q, _ = self.quantizer(z_e)
        logits = self.decoder(z_q)
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        mask = labels != self.pad_id
        acc = (logits.argmax(dim=-1) == labels).masked_fill(~mask, 0).float().sum() / mask.sum().clamp(min=1)
        return loss, acc

    def encode_latent(self, x):
        z_e = self.encoder(x)
        z_q, _ = self.quantizer(z_e)
        return z_q

    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_embed": self.n_embed,
            "vocab_size": self.vocab_size,
            "lr": self.lr,
            "pad_id": self.pad_id
        }
