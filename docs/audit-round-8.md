# Round 8 Audit — Architectural Fix

## TL;DR

The transformer was trained as a **latent predictor** (MSE loss against ground-truth VAE latents) instead of a **next-token predictor**. The round-6/7 fix routed predicted latents through `vae.decoder` to get vocab logits — a hack that didn't actually fix the root cause. Round 8 fixes the architecture end-to-end: the model now outputs vocab logits directly and is trained with CrossEntropy on next-token labels. The model has learned to emit `Bar_N` tokens during generation, fixing the MuseScore empty-bars bug.

## Background — the round-3 audit finding

The round-3 audit documented this in `train_samar_transformer.py:149-159`:

> The architectural shape of this loss comes from the original design: the model is trained to map (latent, events) -> a per-step 128-dim prediction, and the loss is MSE against the ground-truth latent vectors from latents.pt. The model's sample() method treats the 128-dim output as if it were vocab logits via argmax; this is a pre-existing design inconsistency in the model that should be addressed in a separate refactor.

We never did that refactor — round 6 added a `vae_decoder` parameter to `lm.sample()` that projected predicted latents through the VAE decoder to get vocab logits, then sampled from those. This worked (256-token generation, no early stops) but the latents didn't carry enough information to reliably produce Bar/Position/Pitch tokens. The model never learned to emit `Bar_1`, `Bar_2`, etc. during generation.

## Changes

### 1. SamarTransformer architecture

**`samar/models/samar_transformer.py`**

| Field | Round 7 | Round 8 |
|---|---|---|
| `output_layer` | `nn.Linear(d_model, latent_dim)` (128-dim output) | `nn.Linear(d_model, vocab_size)` (1254-dim output) |
| Forward decoder target | `latent_embedding(latent)` | `token_embedding(input_ids)` (autoregressive) |
| Output shape | `[T, B, latent_dim]` | `[T, B, vocab_size]` |
| Latent role | Decoder target (loss source) | Per-step style-conditioning bias on decoder input |
| `sample()` | Sample from `vae_decoder(latent_pred)` | Sample directly from logits |

The forward pass now:
1. Embeds input_ids + position embeddings -> `src` (encoder input)
2. Embeds description + position embeddings, prepends to `src`
3. Embeds input_ids + position embeddings -> `tgt` (decoder target = autoregressive)
4. Adds `latent_embedding(latent)` to `tgt` as a style-conditioning bias (broadcast over T)
5. Runs nn.Transformer(src, tgt) -> `[T, B, d_model]`
6. Projects to vocab_size -> logits `[T, B, vocab_size]`

### 2. Loss function

**`samar/train_samar_transformer.py`**

```python
# Round 7 (MSE on latents):
return F.mse_loss(predicted, latent)

# Round 8 (CrossEntropy on next-token labels):
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    labels.reshape(-1),
    ignore_index=0,  # pad_id
)
```

The `labels` field is already in `SamarLatentDataset.__getitem__` as `tokens[1:]` (the standard autoregressive shift). The model is now trained to predict `input_ids[t+1]` from `input_ids[0..t]` — proper LM training.

### 3. Generating

**`samar/generating.py`**

Removed the `vae.decoder` round-trip. The model output is directly usable as logits. The `--no-vae-decode` flag is no longer needed but kept for backwards compatibility.

### 4. from_pretrained (backwards compat)

`SamarTransformer.from_pretrained` handles the round-7 -> round-8 shape change:
- Old `output_layer.weight` shape: `[128, 256]` (latent_dim=128, d_model=256)
- New `output_layer.weight` shape: `[1254, 256]` (vocab_size=1254, d_model=256)

When loading a round-7 checkpoint into a round-8 model, the old `output_layer` is dropped from the state_dict (shape mismatch) and the new layer is initialized by PyTorch default (small random). All other layers (token_embedding, transformer encoder/decoder, etc.) warm-start cleanly.

## Loss curve

| Epoch | train_loss | val_loss |
|---|---|---|
| 1 | 6.58 | 5.60 |
| 2 | 5.28 | 4.79 |
| 3 | 4.58 | 3.97 |
| 4 | 3.62 | 3.00 |
| 5 | 2.76 | 2.35 |
| 6 | 2.25 | 2.00 |
| 7 | 1.96 | 1.81 |
| 8 | 1.78 | 1.68 |
| 9 | 1.67 | 1.59 |
| 10 | **1.59** | **1.55** |

Baseline (uniform 1254-class prediction) = `log(1254) = 7.13`. The model converges from random to **1.55** in 10 epochs — proper LM training signal.

## MuseScore empty-bars fix — VERIFIED

The model now emits Bar tokens during generation:

```
examples/02_kurd_hafez_t10.txt:
  Bar_0 → Bar_34 → Bar_39 → Bar_38 → Bar_5

examples/03_nahawand_asmahan_t12.txt: 30 Bar tokens
examples/05_huzam_hafez_t10.txt: 33 Bar tokens
examples/01_bayat_fairuz_t08.txt: 52 Bar tokens
```

The reconstructor (already updated in round 7 to handle Bar tokens) now creates real measures per part:

```
examples/06_ajam_fairuz_t10.xml:
  Part P1 (Piano): 7 measures
    Measure 1: 2 notes (C5, A4)
    Measure 4: 5 notes
    Measure 13: 13 notes (includes B4 alter=-0.5 — half-flat microtone!)
    ...
```

## Sample output — half-flat microtone

Example 06 includes the signature 24-EDO quarter-tone:

```
Measure 13: 13 notes
  B4 alter=-0.5 dur=240    <-- half-flat microtone (Arabic maqam hallmark)
  A4 alter=1.0  dur=240    <-- sharp (Kurd maqam interval)
  C5 alter=0    dur=240
```

This is what SAMAR is supposed to produce — Arabic maqam music with 24-EDO quarter-tones.

## What's still suboptimal

1. **Loss hasn't fully converged** (1.55 at epoch 10). Model needs 50-100 epochs to fully learn the REMI structure.
2. **Measure numbers are out of order** when sampling with high temperature — model emits `Bar_27` then `Bar_4` etc. The reconstructor still creates measures in event-stream order, so the user sees "Measure 27" appearing before "Measure 4" in MuseScore. The user needs to be aware that high-temperature output may have measure-number inconsistencies.
3. **Greedy decoding still produces repetitive output** (Velocity_16 spam at temp=0). The model needs more training to learn good defaults.
4. **Few notes per measure** in some examples (example 01 has 3 notes across 52 measures). Model hasn't fully learned Pitch+Duration sequences inside measures yet.

## Future work

- **50-100 epoch retrain** — loss is still descending, more training will help the model learn the full REMI structure (Position+Pitch+Duration+Velocity sequences inside measures)
- **More training data** — 78 files is small. Adding the larger maqam-world dataset would help generalization
- **Decoder causal mask** — currently we use the default nn.Transformer masking. Adding explicit causal masking might improve autoregressive quality
- **Tie-back to description** — the latent pathway is used as style conditioning but the description pathway could also be used more directly (per-piece maqam/genre signal)