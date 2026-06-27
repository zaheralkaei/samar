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
    def __init__(self, d_model, n_head, num_layers, dim_feedforward, dropout, vocab_size, latent_dim, max_len=512, desc_vocab_size=None):
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
        #
        # Round-12 audit: ``description_embedding`` is sized to the
        # ``DescriptionVocab`` (837 tokens) rather than ``vocab_size`` (1254,
        # the event vocab). The previous ``nn.Embedding(vocab_size, d_model)``
        # wasted 33% of description-side capacity: IDs 837..1253 were never
        # reachable from ``DescriptionTokenizer`` so those rows stayed at
        # random init forever. ``desc_vocab_size`` defaults to
        # ``len(DescriptionVocab())`` at __init__ time and can be overridden
        # for backwards compatibility with old checkpoints (see
        # ``from_pretrained``).
        from ..vocab import DescriptionVocab
        default_desc_vocab_size = len(DescriptionVocab())
        if desc_vocab_size is None:
            desc_vocab_size = default_desc_vocab_size
        self.description_embedding = nn.Embedding(desc_vocab_size, d_model)
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

        # Round-8 architectural fix: project transformer output
        # directly to vocab logits, not to a 128-dim latent vector.
        # The previous design (nn.Linear(d_model, latent_dim) with
        # MSE loss against ground-truth latents) made sample() have
        # to project through vae.decoder at every step -- a hack.
        # Now the model is a proper next-token predictor: loss is
        # CrossEntropy against the next-event-id labels, and sample()
        # uses the raw logits directly.
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, latent=None, description=None, tgt=None, enc_output=None):
            """Forward pass.

            Architecture (round-12 architectural fix):

            - ``src`` (encoder input) = description ONLY + pos_embedding.
              The encoder is bidirectional over the per-bar description
              tokens (Bar_N, TimeSignature, MeanPitch, ...) so every
              description position can attend to every other.
            - ``tgt`` (decoder input) = input_ids (event tokens) + pos_embedding.
              The decoder self-attention is masked causal so position t
              cannot see positions > t. This makes the model a proper
              autoregressive language model.
            - Cross-attention: decoder attends to encoder output (the
              encoded description context). This is the standard seq2seq
              pattern from FIGARO.
            - ``latent`` is added to the decoder input as a per-step bias
              for style conditioning.

            Round-12 audit fixes the round-8 leak where the encoder was
            fed ``description + events`` bidirectionally. That made the
            encoder output at position 0 encode information about future
            event tokens, which leaked through cross-attention and let
            the decoder peek at the answer during training. With the
            encoder now restricted to description, the only path from
            events to decoder outputs is the causal decoder self-attention.

            Round-8 had previously fixed the round-7 latent-as-target
            mistake (the model was a latent->vocab mapper, not an LM).

            ``enc_output`` (round-12): if provided, skip the encoder and
            use this precomputed encoder output directly. ``sample()`` uses
            this to compute the encoder output ONCE on the static
            description, then reuses it across all sampling steps.
            ``enc_output`` shape: [T_src, B, d_model].
            """
            B, T = input_ids.size()

            # === Encoder src = description only (bidirectional) ===
            if enc_output is not None:
                # Use the cached encoder output. Sample() passes this.
                src = enc_output  # [T_src, B, d_model]
                # src is already [T, B, d_model] (i.e. already permuted).
                # We skip the permute below for this branch.
                skip_permute = True
            else:
                skip_permute = False
                if description is not None:
                    B_d, T_d = description.size()
                    desc_emb = self.description_embedding(description)
                    desc_pos_ids = torch.arange(T_d, device=description.device).unsqueeze(0).expand(B_d, -1)
                    src = desc_emb + self.pos_embedding(desc_pos_ids)
                else:
                    # No description provided (e.g. MIDI samples that have
                    # ``description=None``). Use a single dummy "no-context"
                    # token so the encoder still produces an output the
                    # decoder can attend to. We use the ``token_embedding(0)``
                    # row (which is the <pad> row) as a neutral placeholder.
                    placeholder = self.token_embedding(torch.zeros(B, 1, dtype=torch.long, device=input_ids.device))
                    src = placeholder  # [B, 1, d_model]

            # === Decoder tgt = event tokens (causal) ===
            if tgt is None:
                tgt = self.token_embedding(input_ids)
                pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
                tgt = tgt + self.pos_embedding(pos_ids)

            # === Add latent as per-step style bias to decoder input ===
            if latent is not None:
                # Broadcast latent over the sequence length and add it.
                # latent shape: [B, T_latent, latent_dim] -> [B, 1, d_model]
                latent_emb = self.latent_embedding(latent)  # [B, T_lat, d_model]
                if latent_emb.size(1) == 1:
                    # Already a single style vector, broadcast over T.
                    tgt = tgt + latent_emb
                elif latent_emb.size(1) == tgt.size(1):
                    # Same length, add per-step.
                    tgt = tgt + latent_emb
                else:
                    # Mismatched lengths: use the mean latent as a
                    # global style vector.
                    tgt = tgt + latent_emb.mean(dim=1, keepdim=True)

            if not skip_permute:
                src = src.permute(1, 0, 2)
            # Always permute tgt -- even in the skip_permute branch where
            # src is already in time-major [T, B, d_model] format, tgt still
            # needs the [B, T, d_model] -> [T, B, d_model] swap to match
            # nn.Transformer conventions.
            tgt = tgt.permute(1, 0, 2)

            # Round-12 audit: enforce causality on the decoder self-
            # attention. Without this mask, position t can attend to
            # position t+1 (and beyond), so the model learns to peek
            # at the answer during training. At inference (``lm.sample``)
            # the future tokens don't exist yet, so the learned
            # distribution is off-policy.
            #
            # PyTorch's nn.Transformer requires the mask as a [T, T]
            # tensor where masked positions are -inf.
            # ``generate_square_subsequent_mask`` is the canonical
            # upper-triangular -inf matrix.
            T_dec = tgt.size(0)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                T_dec, device=tgt.device
            )

            output = self.transformer(src, tgt, tgt_mask=causal_mask)
            # Output is [T_dec, B, d_model] -> [T_dec, B, vocab_size]
            logits = self.output_layer(output)
            return logits

    def _encode_description(self, description):
        """Run the encoder on the static description, return the encoded
        representation that the decoder cross-attends to.

        Round-12 audit: called once per ``sample()`` invocation so the
        encoder cost is amortized across all sampling steps. Returns
        ``[T_src, B, d_model]`` (already in ``nn.Transformer`` time-major
        format -- i.e. NOT batch-major).

        ``description`` may be ``None`` (e.g. MIDI samples); we return a
        single-token dummy encoder output (the ``<pad>`` row of
        ``token_embedding``) so the decoder's cross-attention still has
        a target.
        """
        if description is None:
            # 1-token dummy. Use the encoder directly so we don't go
            # through ``forward()`` (which would also build the decoder
            # inputs we don't need here).
            B = 1
            placeholder = self.token_embedding(
                torch.zeros(B, 1, dtype=torch.long, device=next(self.parameters()).device)
            )
            return self.transformer.encoder(placeholder.permute(1, 0, 2))

        B_d, T_d = description.size()
        desc_emb = self.description_embedding(description)
        desc_pos_ids = torch.arange(T_d, device=description.device).unsqueeze(0).expand(B_d, -1)
        src = desc_emb + self.pos_embedding(desc_pos_ids)
        # ``nn.TransformerEncoder`` expects [T, B, d_model] (time-major).
        return self.transformer.encoder(src.permute(1, 0, 2))

    def sample(self, start_tokens, latent=None, max_length=256, pad_id=None,
               description=None, temperature=1.0, top_k=0, top_p=0.0,
               vae_decoder=None):
        """Autoregressive decode with temperature / top-k / top-p sampling.

        Round-8: the model output is now ``[T, B, vocab_size]`` directly,
        so we just sample from the last-step logits. The ``vae_decoder``
        parameter is kept for backwards compatibility with the round-7
        generation pipeline but is no longer needed (model output is
        already vocab logits).

        Parameters
        ----------
        start_tokens : [1, T] long
            Already-generated prefix. Generation starts AFTER these tokens.
        latent : [1, T_lat, latent_dim], optional
            Seed latent for style conditioning.
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
        vae_decoder : callable, optional (DEPRECATED in round-8)
            No longer needed -- kept for backward compat with old
            generation scripts.
        """
        self.eval()
        generated = start_tokens

        # Round-12 audit: pre-compute the encoder output ONCE on the
        # static description, then reuse it across all sampling steps.
        # The description doesn't change as we generate, so re-running
        # the encoder every step is wasted compute. ``enc_output`` is
        # the encoder's output (after self-attention over description)
        # which the decoder cross-attends to.
        enc_output = self._encode_description(description)

        for _ in range(max_length - start_tokens.size(1)):
            logits = self(
                generated, latent=latent, description=description,
                enc_output=enc_output,
            )
            logits = logits.permute(1, 0, 2)  # [B, T, vocab_size]
            next_logits = logits[:, -1, :]    # [B, vocab_size]

            if temperature <= 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-6)
                # Mask out special tokens (<pad>, <unk>, <bos>, <eos>, <mask>)
                # so the model can't pick them as actual events. The VAE
                # decoder often assigned them non-trivial probability in
                # round-7, but the round-8 token-prediction architecture
                # can still occasionally pick <pad> at sequence boundaries.
                if getattr(self, "_special_token_ids", None):
                    scaled = scaled.clone()
                    scaled[:, self._special_token_ids] = float("-inf")
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
                    sorted_logits = sorted_logits.masked_fill(
                        mask, float("-inf"))
                    scaled = torch.zeros_like(scaled).scatter_(
                        -1, sorted_idx, sorted_logits)
                probs = torch.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

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
    def from_pretrained(cls, ckpt_path, config=None, device="cpu", warm_start_missing=True, output_legacy_check=True):
        """Load a SamarTransformer from a checkpoint.

        The checkpoint may pre-date later additions to the architecture (e.g.
        ``description_embedding`` and ``pos_embedding`` were added after the
        initial training run; ``output_layer`` was switched to vocab_size in
        round 8). We load with ``strict=False`` and, by default, warm-start
        any missing layer from sensible defaults so the model produces
        sensible outputs without retraining.

        Round-8 specific: ``output_layer`` shape changed from
        ``(latent_dim, d_model)`` to ``(vocab_size, d_model)``. If a round-7
        (or earlier) checkpoint is loaded into a round-8 model, the old
        ``output_layer.weight`` shape won't match and the layer will be in
        ``missing``. We warm-start it from a small random init (rows 0..N
        get small random values, the rest stay zero).

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
        # Round-17: ``save_model`` wraps the weights under
        # ``model_state_dict`` (alongside optimizer/scheduler state).
        # Detect that format too so ``from_pretrained`` works on
        # round-17+ checkpoints.
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
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

        # Round-12 audit: description_embedding. The checkpoint may have
        # ``description_embedding.weight`` sized to the OLD description
        # vocab (=event vocab_size=1254) while the new model uses
        # ``len(DescriptionVocab())`` = 837. Three cases:
        #   1. shapes match -- load directly.
        #   2. ckpt larger (1254 -> 837): trim to first 837 rows.
        #      We were never using the extra rows anyway.
        #   3. ckpt smaller (e.g. 837 -> 837, no change): no-op.
        # In all cases we never extend description_embedding -- it's
        # already the correct size for the live DescriptionVocab.
        if "description_embedding.weight" in sd:
            ckpt_desc_size = sd["description_embedding.weight"].shape[0]
            model_desc_size = model.description_embedding.weight.shape[0]
            if ckpt_desc_size > model_desc_size:
                sd["description_embedding.weight"] = (
                    sd["description_embedding.weight"][:model_desc_size].clone()
                )

        # Round-8: handle output_layer shape change. In round 7 and
        # earlier, output_layer was ``Linear(d_model, latent_dim)``
        # so its weight was ``[latent_dim, d_model]``. From round 8
        # onward it's ``Linear(d_model, vocab_size)`` so the weight is
        # ``[vocab_size, d_model]``. If we load a round-7 checkpoint
        # into a round-8 model, the shapes mismatch and the layer
        # ends up in ``missing``. Drop the legacy shape from the
        # state dict so strict=False doesn't try to load it.
        if "output_layer.weight" in sd:
            old_out_shape = sd["output_layer.weight"].shape  # [out_dim, d_model]
            new_out_dim = model.output_layer.weight.shape[0]
            if old_out_shape[0] != new_out_dim:
                del sd["output_layer.weight"]
                if "output_layer.bias" in sd and sd["output_layer.bias"].shape[0] != new_out_dim:
                    del sd["output_layer.bias"]

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
                    elif name == "output_layer.weight":
                        # Round-8: output_layer is now [vocab_size, d_model].
                        # PyTorch nn.Linear's default init gives random small
                        # values which is fine for warm-starting from a
                        # different-shape checkpoint. We don't need to do
                        # anything special -- PyTorch already initialized it.
                        pass

        # Round-6: pre-compute the list of special token IDs that the
        # sampler should mask out (<pad>, <unk>, <bos>, <eos>, <mask>).
        # Avoids a per-step dictionary lookup.
        try:
            from samar.tokenizer import SamarTokenizer
            import os as _os
            tk = SamarTokenizer.load(_os.path.join(_os.path.dirname(__file__), "..", "samar_vocab.pkl"))
            vocab = tk.get_vocab()
            specials = ["<pad>", "<unk>", "<bos>", "<eos>", "<mask>"]
            ids = []
            for tok in specials:
                if tok in vocab.stoi:
                    ids.append(vocab.stoi[tok])
            model._special_token_ids = ids
        except Exception:
            model._special_token_ids = []
        return model.to(device), {"missing": missing, "unexpected": unexpected}
