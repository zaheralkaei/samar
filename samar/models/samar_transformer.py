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

    def sample(self, start_tokens, latent=None, max_length=256, pad_id=None,
               description=None, temperature=1.0, top_k=0, top_p=0.0,
               vae_decoder=None):
        """Autoregressive decode with temperature / top-k / top-p sampling.

        The model outputs ``[seq_len, batch, latent_dim]`` (next-step latent
        predictions, not event-vocab logits). To get event tokens:

        - If ``vae_decoder`` is provided, the predicted latent at each step is
          passed through ``vae_decoder`` to get vocab logits, then sampled
          with the requested temperature / top-k / top-p.
        - Otherwise (legacy behaviour), we fall back to ``argmax`` of the
          latent-dim output. This produces invalid token IDs because the
          model was trained to predict latents, not tokens -- the round-3
          audit documented this architectural inconsistency.

        Parameters
        ----------
        start_tokens : [1, T] long
            Already-generated prefix. Generation starts AFTER these tokens.
        latent : [1, T_lat, latent_dim], optional
            Seed latent broadcast across the generation length.
        description : [1, T_desc] long, optional
            Per-bar description tokens (see ``forward``).
        temperature : float, default 1.0
            ``>1.0`` -> more random, ``<1.0`` -> more deterministic.
            ``0.0`` is treated as greedy (argmax).
        top_k : int, default 0
            Keep only the top-k logits before sampling. ``0`` disables.
        top_p : float, default 0.0
            Nucleus sampling -- keep the smallest set of tokens whose
            cumulative probability >= top_p. ``0.0`` disables.
        vae_decoder : callable, optional
            ``vae_decoder(latent) -> [B, T, vocab_size] logits``.
            If provided, used to project the predicted latent to vocab logits.
        """
        self.eval()
        generated = start_tokens
        for _ in range(max_length - start_tokens.size(1)):
            output = self(generated, latent=latent, description=description)
            output = output.permute(1, 0, 2)  # [B, T, dim]
            next_step = output[:, -1, :]      # [B, dim]

            if vae_decoder is not None:
                # Project latent -> vocab logits and sample.
                logits = vae_decoder(next_step)  # [B, vocab_size]
                if temperature <= 0.0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-6)
                    if top_k > 0:
                        k = min(top_k, logits.size(-1))
                        values, _ = torch.topk(logits, k, dim=-1)
                        threshold = values[:, -1:].expand_as(logits)
                        logits = torch.where(
                            logits < threshold,
                            torch.full_like(logits, float("-inf")),
                            logits,
                        )
                    if top_p > 0.0:
                        sorted_logits, sorted_idx = torch.sort(
                            logits, descending=True, dim=-1)
                        probs = torch.softmax(sorted_logits, dim=-1)
                        cum = torch.cumsum(probs, dim=-1)
                        # Remove tokens with cumulative prob > top_p
                        mask = cum - probs > top_p
                        sorted_logits = sorted_logits.masked_fill(
                            mask, float("-inf"))
                        logits = torch.zeros_like(logits).scatter_(
                            -1, sorted_idx, sorted_logits)
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(
                        probs, num_samples=1)  # [B, 1]
            else:
                # Legacy path: argmax of latent-dim output (produces
                # invalid token IDs but kept for backwards compat).
                next_token = torch.argmax(next_step, dim=-1, keepdim=True)

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

        Configuration resolution order (audit round-2 finding L2):
          1. ``config`` argument (caller-supplied overrides)
          2. ``config`` dict inside the checkpoint file (top-level key)
          3. Inferred from the checkpoint's ``state_dict`` shapes (only
             the keys it can derive)

        Whichever value is set last wins. This means a caller passing
        ``config={"vocab_size": 1249}`` to load the round-2 checkpoint
        (whose ``state_dict`` was trained with vocab_size=1129) gets the
        1249-row model and the first 1129 rows of the checkpoint's
        ``token_embedding`` are loaded; rows 1129..1248 are random init.
        That's the FIGARO-style "vocab extension" pattern -- extending
        the vocabulary without retraining the original rows.
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

        # Handle vocab extension: the model's ``vocab_size`` may be larger
        # than the checkpoint's. Resize the checkpoint's token/description
        # embeddings to match (new rows are zero-init -- the user should
        # retrain the model before relying on the new rows). See
        # ``docs/audit-round-2.md`` finding A3.
        ckpt_vocab_size = sd["token_embedding.weight"].shape[0]
        if ckpt_vocab_size != config.get("vocab_size"):
            new_size = config["vocab_size"]
            old_emb = sd["token_embedding.weight"]
            extended = _torch.zeros(new_size, old_emb.shape[1], dtype=old_emb.dtype)
            extended[:ckpt_vocab_size] = old_emb
            sd["token_embedding.weight"] = extended
            # If the description embedding has the same shape, extend it too.
            if "description_embedding.weight" in sd and sd["description_embedding.weight"].shape[0] == ckpt_vocab_size:
                old_desc = sd["description_embedding.weight"]
                ext_desc = _torch.zeros(new_size, old_desc.shape[1], dtype=old_desc.dtype)
                ext_desc[:ckpt_vocab_size] = old_desc
                sd["description_embedding.weight"] = ext_desc

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
