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
        # ``description_embedding`` is intentionally a separate embedding matrix
        # rather than sharing weights with ``token_embedding``. The two vocabularies
        # are distinct (SamarVocab vs DescriptionVocab, see vocab.py and the audit
        # round #2 notes) and the descriptions and events come from different
        # distributions -- mean pitch / note density / bar tokens behave nothing
        # like 24-EDO pitch / bar / position tokens. Sharing weights would let
        # the description-side vocabulary "eat" the event embedding capacity.
        self.description_embedding = nn.Embedding(vocab_size, d_model)
        self.latent_embedding = nn.Linear(latent_dim, d_model)

        # Positional encoding. Max length 512 matches ``MAX_N_BARS`` so we can
        # condition on a full sequence even if we never train at that length.
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

    def sample(self, start_tokens, latent=None, max_length=256, pad_id=None, description=None):
        """Greedy autoregressive decode.

        ``description`` (optional) is a ``[1, T_desc]`` tensor of description
        token IDs that conditions the generation. Pass it when you want to
        steer the output (e.g. with a specific time-signature / key /
        note-density profile). See ``SamarTransformer.forward`` for the
        conditioning path.
        """
        self.eval()
        generated = start_tokens  # shape: [1, T]
        for _ in range(max_length - start_tokens.size(1)):
            logits = self(generated, latent=latent, description=description)  # [seq_len, batch, dim]
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

    @classmethod
    def from_pretrained(cls, ckpt_path, config=None, device="cpu", warm_start_missing=True):
        """Load a SamarTransformer from a checkpoint.

        The checkpoint may pre-date later additions to the architecture (e.g.
        ``description_embedding`` and ``pos_embedding`` were added after the
        initial training run). We load with ``strict=False`` and, by default,
        warm-start any missing embedding from ``token_embedding`` so the model
        produces sensible outputs without retraining.
        """
        import torch as _torch
        sd = _torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        if config is None:
            cfg_keys = ["d_model", "n_head", "num_layers", "dim_feedforward",
                        "dropout", "vocab_size", "latent_dim"]
            config = {k: sd.get(k) or None for k in cfg_keys}
            config = {k: v for k, v in config.items() if v is not None}
        model = cls(**config)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if warm_start_missing and missing:
            with _torch.no_grad():
                for name in missing:
                    if name == "description_embedding.weight" and hasattr(model, "token_embedding"):
                        model.description_embedding.weight.copy_(model.token_embedding.weight)
                    elif name == "pos_embedding.weight":
                        # Sinusoidal positional init (Vaswani et al. 2017)
                        max_len, d = model.pos_embedding.weight.shape
                        pe = _torch.zeros(max_len, d)
                        pos = _torch.arange(0, max_len, dtype=_torch.float).unsqueeze(1)
                        div = _torch.exp(_torch.arange(0, d, 2).float() * (-_torch.log(_torch.tensor(10000.0)) / d))
                        pe[:, 0::2] = _torch.sin(pos * div)
                        pe[:, 1::2] = _torch.cos(pos * div)
                        model.pos_embedding.weight.copy_(pe)
        return model.to(device), {"missing": missing, "unexpected": unexpected}
