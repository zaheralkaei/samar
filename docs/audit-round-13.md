# Round 13 Audit — Training-time soundness (post-round-12 audit)

**Date:** 2026-06-27

**TL;DR**: Round 12 fixed the architectural leak (no causal mask) but two
critical data-pipeline bugs slipped through and would have silently
degraded the next training run:

1. **`samar_collate_fn` was destroying the description stream** — every
   description token was silently being mapped to `<unk>` (id 1) because
   the collate treated pre-encoded IDs as strings and re-ran them through
   the string-keyed `stoi` dict. Description conditioning was effectively
   non-functional in all prior training runs.
2. **`__getitem__` returned latent at VAE-source length (154 for
   Hello.xml samples) but tokens padded to `context_size`** — the
   model's per-step latent add fell back to the mean-pool branch for
   any short sample, losing the per-step style signal.

Both are now fixed and verified end-to-end with the real checkpoint.

## Findings & Fixes

### FIX R13-A — Collate destroys description IDs (CRITICAL)

**WHERE:** `samar/dataset.py:samar_collate_fn` (line 245)

**Bug:** `precompute_samar_latents.py` stores `description` in
`latents.pt` as **integer IDs** (DescriptionVocab IDs). The dataset's
`__getitem__` reads these IDs and returns them as a tensor of int64
values.

The collate then called:
```python
desc_ids = [torch.tensor(desc_tok.encode(desc), dtype=torch.long) for desc in desc_lists]
```

But `desc` was already a tensor of int IDs, not strings.
`DescriptionTokenizer.encode()` calls `vocab.encode(seq)` which calls
`vocab.to_i(tok)` for each `tok`. `to_i` does `stoi.get(token, ...)`,
where `stoi` keys are strings. Passing an int → `stoi.get(int, default)`
returns `default`, which is `<unk>` (id 1).

**Evidence:**
- Before fix: `samar_collate_fn(ds[0:2])["description"][0][:5]` →
  `[1, 1, 1, 1, 1]` (all `<unk>`)
- After fix: → `[6, 809, 731, 705, 684]` (real IDs preserved)
- Real precomputed IDs were `[6, 809, 731, 705, 684]` (DescriptionVocab
  IDs for `Bar_1, TimeSignature_4/4, MeanPitch_..., ...`)

**Fix:** Type-aware handling in collate:
- If `desc` is a `torch.Tensor` → already IDs, use directly
- If `desc` is a list of ints → already IDs, convert to tensor
- Otherwise (strings) → encode via `desc_tok.encode(desc)`

**Impact:** All prior training runs (rounds 1-10) had a non-functional
description stream. The model never learned to condition on descriptions
because it always saw `<unk>` repeated. Round 13 fixes this; future
training runs will use real description tokens.

### FIX R13-B — Latent length mismatch with input_ids length (HIGH)

**WHERE:** `samar/dataset.py:SamarLatentDataset.__getitem__` (line 187)

**Bug:** The VAE produces one latent frame per source token (length 154
for Hello.xml, length 255 for normal samples). The dataset pads tokens
to `context_size` but **left the latent at its source length**. When
forward() runs, it compares `latent_emb.size(1)` to `tgt.size(1)`:

```python
if latent_emb.size(1) == tgt.size(1):  # per-step add
elif latent_emb.size(1) == 1:           # broadcast
else:                                   # mean-pool (fallback!)
    tgt = tgt + latent_emb.mean(dim=1, keepdim=True)
```

For Hello.xml samples (154 latent, 511 tgt): **mean-pool branch fires**,
losing per-step style. For 255-token samples (255 latent, 511 tgt): same
problem.

Also: `input_ids` in the trainer is `tokens[:-1]` (length
`context_size-1`, e.g. 511 for context_size=512). The latent should be
padded to match `input_ids` length, not `context_size`. So padding
to `context_size` (256) gives length 256, while input_ids is length
255 — still off by 1.

**Fix:** Pad/truncate latent to `context_size - 1` (matching input_ids
length). Use zero-padding (consistent with round-8 latent-pad
behavior).

**Verification:**
- Before fix: latent shape `(255, 128)`, input_ids shape `(255,)` — match
  but broken for Hello.xml.
- After fix: latent shape `(255, 128)` for normal samples, `(255, 128)` for
  Hello.xml (padded from 154 to 255 with zeros).
- With context_size=512: latent shape `(511, 128)` for normal samples,
  matching input_ids `(511,)`.

### FIX R13-C — `min_keep_len` default too aggressive (MEDIUM)

**WHERE:** `samar/dataset.py:SamarLatentDataset.__init__`

**Bug:** Round 12's default of `min_keep_len = context_size // 2`
filtered all 700 Arabic samples when `context_size=512` (since all are
154 or 255 tokens, both < 256).

**Fix:** Default `min_keep_len = 64` (a permissive minimum that just
rejects truly degenerate samples like 5-token fragments).

**Verification:** With context_size=512, all 700 samples retained. With
context_size=256, also all 700 retained.

## What's NOT changed

### Trainer forward call

`train_samar_transformer.py:172` calls
`self.model(input_ids, latent=latent, description=description)`. The
round-12 forward signature gained `enc_output` (default None) and
`tgt` (default None); both default to None for training. No trainer
change needed.

### Save/load round-trip

`from_pretrained` correctly handles the round-12 `desc_vocab_size` resize:
- New checkpoint (after retraining): description_embedding [837, 256] in
  state_dict, no resize needed.
- Old checkpoint: description_embedding [1254, 256] → trimmed to [837,
  256]. 0 missing / 0 unexpected keys.

### get_config() drift risk

`get_config()` does NOT save `desc_vocab_size`. This is fine as long as
`DescriptionVocab` doesn't change. If new description tokens are added
later, the model will silently resize incorrectly. Document this risk
in pre-retraining checklist.

## Verification cluster (all probes PASS)

End-to-end with real checkpoint, context_size=512:

```
Batch shapes: input_ids=(8, 511), latent=(8, 511, 128), description=(8, 512)
logits shape: (511, 8, 1254)
loss: 8.38 (sane initial loss for random output_layer)
params with grad: 191/191
description <unk> rate: 0/4096 = 0.0%  (was 100% before R13-A)
pos 0..4 diff: 0.0000 (causal)
pos 5..7 diff: >0 (legit)
```

## Pre-retraining checklist (UPDATED)

- [ ] Decide on Fairuz bias strategy (downsample/upweight/document)
- [ ] Decide on MIDI VAE strategy (separate VAE / drop latent for MIDI)
- [ ] Regenerate `latents/latents.pt` after deleting Hello.xml —
  `python -m samar.precompute_samar_latents`
- [ ] **Verify regenerating latents preserves the new fix** — re-run
  `samar_collate_fn` on the new latents and confirm
  `description <unk> rate == 0`
- [ ] **Verify the new latents' latent length matches `context_size - 1`**
  — re-run the round-13 latent-shape probe
- [ ] Retrain Arabic model (50 epochs recommended)
- [ ] Watch for val_loss curve: should now drop steadily because the
  description stream provides useful signal that previously wasn't
  getting through

## Files Changed (round 13)

- `samar/dataset.py`:
  - `samar_collate_fn` description branch now handles tensor/list-of-int/
    list-of-string inputs
  - `__getitem__` pads/truncates latent to `context_size - 1` (was: raw
    passthrough)
  - `min_keep_len` default changed from `context_size // 2` to `64`
- `docs/audit-round-13.md` (this file)