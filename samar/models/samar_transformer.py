# -*- coding: utf-8 -*-
"""
SAMAR Transformer — FIGARO-aligned seq2seq for Arabic music generation.

Round 18: Full architecture rewrite to align with FIGARO's figaro-expert
mode (description-only encoder, causal event decoder).

Key changes from round 17:
  - Encoder uses bar_embedding(desc_bar_ids) instead of absolute pos
  - Decoder uses bar_embedding(bar_ids) + position_embedding(position_ids)
  - No latent conditioning (VQ-VAE dropped)
  - Separate encoder/decoder layer counts (2+4 default)
  - Description scrolling during sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..constants import (
    BAR_KEY, POSITION_KEY, MAX_N_BARS, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN,
)


class SamarTransformer(nn.Module):
    def __init__(self, d_model, n_head, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, dropout, vocab_size, desc_vocab_size=None,
                 max_bars=512, max_positions=512):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.max_bars = max_bars
        self.max_positions = max_positions

        from ..vocab import DescriptionVocab
        if desc_vocab_size is None:
            desc_vocab_size = len(DescriptionVocab())
        self.desc_vocab_size = desc_vocab_size

        # Event token embedding (decoder)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Description token embedding (encoder)
        self.description_embedding = nn.Embedding(desc_vocab_size, d_model)

        # Structural embeddings (FIGARO pattern)
        self.bar_embedding = nn.Embedding(max_bars + 1, d_model)
        self.position_embedding = nn.Embedding(max_positions + 1, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )

        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, description=None, bar_ids=None,
                position_ids=None, desc_bar_ids=None, enc_output=None):
        """Forward pass.

        Encoder: description tokens + bar_embedding(desc_bar_ids).
        Decoder: event tokens + bar_embedding(bar_ids) + position_embedding(position_ids).
        Cross-attention connects them.
        """
        B, T = input_ids.size()

        # === Encoder ===
        if enc_output is not None:
            src = enc_output
        else:
            src = self._encode_description(description, desc_bar_ids)

        # === Decoder ===
        tgt = self.token_embedding(input_ids)
        if bar_ids is not None:
            bar_ids_clamped = bar_ids.clamp(0, self.max_bars)
            tgt = tgt + self.bar_embedding(bar_ids_clamped)
        if position_ids is not None:
            pos_ids_clamped = position_ids.clamp(0, self.max_positions)
            tgt = tgt + self.position_embedding(pos_ids_clamped)

        # [B, T, d_model] -> [T, B, d_model]
        tgt = tgt.permute(1, 0, 2)

        T_dec = tgt.size(0)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T_dec, device=tgt.device
        )

        output = self.transformer(src, tgt, tgt_mask=causal_mask)
        logits = self.output_layer(output)
        return logits

    def _encode_description(self, description, desc_bar_ids=None):
        """Encode description tokens. Returns [T_src, B, d_model]."""
        if description is None:
            B = 1
            device = next(self.parameters()).device
            placeholder = self.token_embedding(
                torch.zeros(1, 1, dtype=torch.long, device=device)
            )
            return self.transformer.encoder(placeholder.permute(1, 0, 2))

        desc_emb = self.description_embedding(description)
        if desc_bar_ids is not None:
            bar_ids_clamped = desc_bar_ids.clamp(0, self.max_bars)
            desc_emb = desc_emb + self.bar_embedding(bar_ids_clamped)

        src = desc_emb.permute(1, 0, 2)  # [T, B, d_model]
        return self.transformer.encoder(src)

    @torch.no_grad()
    def sample(self, start_tokens, description=None, desc_bar_ids=None,
               max_length=256, max_bars=-1, temperature=0.8, top_k=0,
               top_p=0.0, vocab=None):
        """Autoregressive sampling with dynamic bar/position tracking.

        Follows FIGARO's sample() pattern: tracks bar_ids and position_ids
        as tokens are generated, updates them based on token type.
        """
        self.eval()
        device = start_tokens.device

        if vocab is None:
            from ..vocab import SamarVocab
            vocab = SamarVocab()

        pad_id = vocab.to_i(PAD_TOKEN)
        eos_id = vocab.to_i(EOS_TOKEN)
        bos_id = vocab.to_i(BOS_TOKEN)

        # Pre-compute encoder output
        enc_output = self._encode_description(description, desc_bar_ids)

        generated = start_tokens  # [1, T_start]
        B = generated.size(0)

        # Initialize bar_ids and position_ids for start tokens
        bar_ids, position_ids = self._compute_structural_ids(
            generated, vocab, device
        )

        for step in range(max_length - start_tokens.size(1)):
            logits = self(
                generated, enc_output=enc_output,
                bar_ids=bar_ids, position_ids=position_ids,
            )
            logits = logits.permute(1, 0, 2)  # [B, T, vocab]
            next_logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-6)
                # Mask special tokens
                for special in [pad_id, bos_id]:
                    scaled[:, special] = float("-inf")
                if top_k > 0:
                    k = min(top_k, scaled.size(-1))
                    values, _ = torch.topk(scaled, k, dim=-1)
                    threshold = values[:, -1:].expand_as(scaled)
                    scaled = torch.where(
                        scaled < threshold,
                        torch.full_like(scaled, float("-inf")),
                        scaled,
                    )
                if top_p > 0.0:
                    sorted_logits, sorted_idx = torch.sort(
                        scaled, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs, dim=-1)
                    mask = cum - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                    scaled = torch.zeros_like(scaled).scatter_(
                        -1, sorted_idx, sorted_logits)
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Update structural IDs for the new token
            next_tok_str = vocab.to_s(next_token.item())
            cur_bar = bar_ids[0, -1].item()
            cur_pos = position_ids[0, -1].item()

            if next_tok_str == EOS_TOKEN:
                new_bar = 0
                new_pos = 0
            elif next_tok_str.startswith(f"{BAR_KEY}_"):
                new_bar = cur_bar + 1
                new_pos = 0
            elif next_tok_str.startswith(f"{POSITION_KEY}_"):
                try:
                    new_pos = int(next_tok_str.split("_")[-1])
                except ValueError:
                    new_pos = cur_pos
                new_bar = cur_bar
            else:
                new_bar = cur_bar
                new_pos = cur_pos

            bar_ids = torch.cat([
                bar_ids,
                torch.tensor([[new_bar]], device=device)
            ], dim=1)
            position_ids = torch.cat([
                position_ids,
                torch.tensor([[new_pos]], device=device)
            ], dim=1)

            if next_token.item() == eos_id:
                break
            if max_bars > 0 and new_bar > max_bars:
                break

        return generated

    def _compute_structural_ids(self, token_ids, vocab, device):
        """Compute bar_ids and position_ids for a token sequence."""
        B, T = token_ids.size()
        bar_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        position_ids = torch.zeros(B, T, dtype=torch.long, device=device)

        for b in range(B):
            cur_bar = 0
            cur_pos = 0
            for t in range(T):
                tok_str = vocab.to_s(token_ids[b, t].item())
                if tok_str == BOS_TOKEN:
                    cur_bar = 0
                    cur_pos = 0
                elif tok_str.startswith(f"{BAR_KEY}_"):
                    cur_bar += 1
                    cur_pos = 0
                elif tok_str.startswith(f"{POSITION_KEY}_"):
                    try:
                        cur_pos = int(tok_str.split("_")[-1])
                    except ValueError:
                        pass
                bar_ids[b, t] = cur_bar
                position_ids[b, t] = cur_pos

        return bar_ids, position_ids

    def get_config(self):
        return {
            "d_model": self.d_model,
            "n_head": self.n_head,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "vocab_size": self.vocab_size,
            "desc_vocab_size": self.desc_vocab_size,
            "max_bars": self.max_bars,
            "max_positions": self.max_positions,
        }

    @classmethod
    def from_pretrained(cls, ckpt_path, config=None, device="cpu"):
        """Load from checkpoint. Round-18 checkpoints only."""
        import torch as _torch
        loaded = _torch.load(ckpt_path, map_location=device, weights_only=False)

        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            sd = loaded["model_state_dict"]
        elif isinstance(loaded, dict) and "state_dict" in loaded:
            sd = loaded["state_dict"]
        else:
            sd = loaded

        if config is None:
            raise ValueError("config must be provided for round-18 models")

        model = cls(**config)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return model.to(device), {"missing": missing, "unexpected": unexpected}
