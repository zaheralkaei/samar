# SAMAR Round 5 — Retrain from zero on the AI laptop

This document records the round-5 retraining of both the VQ-VAE
and the Transformer from scratch on the cleaned 78-file corpus.

## Setup

- **Hardware**: `ai-laptop` (Ubuntu 24.04, Intel Core i7-1165G7
  4-core 8-thread, 15GB RAM, **no GPU** — CPU-only training)
- **Python**: 3.12.3 (system) in `~/projects/samar/.venv/`
- **Torch**: 2.12.1+cpu (from `https://download.pytorch.org/whl/cpu`)
- **Other deps**: numpy, tqdm, tensorboard
- **Data**: 78 MusicXML files (52 `.xml` + 26 `.mxl`), 672
  token chunks @ context_size=256
- **Vocab size**: 1254 tokens (SamarVocab) + 837 tokens (DescriptionVocab)

## What got fixed in round 5 (3 commits before retraining)

1. **`precompute_samar_latents.py` saves descriptions per latent**
   (round-3 deferred item). Each latent in `latents.pt` now has
   `description: [int, ...]` next to `tokens`/`latent`/`file`.
   Encoded via `DescriptionTokenizer` against `DescriptionVocab`.

2. **`SAMARDataset` loads `.mxl` files** (round-4 audit caught
   this gap during retraining — initial run only saw 52 files).
   Was: `glob(*.xml)`. Now: `glob(*.xml) + glob(*.mxl)`.

3. **`generating.py` accepts `--description-source`** for
   description-conditional generation. Defaults to using the
   description stored in the chosen latent sample.

## Training pipeline

```bash
# 1. (Already done) Install deps
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy tqdm tensorboard

# 2. (Already done) Train the VAE (~60 sec on CPU for 10 epochs)
tmux new-session -d -s train-vae ".venv/bin/python -m samar.train_samar_vae > logs/train_vae.log 2>&1; touch logs/train_vae.done"
#   Epoch 1: loss 6.89, acc 14.9%
#   Epoch 10: loss 3.13, acc 25.7%
#   Output: checkpoints/samar_vae.pt (1.95 MB)

# 3. (Already done) Recompute latents with new (round-5) vocab
tmux new-session -d -s precompute ".venv/bin/python -m samar.precompute_samar_latents > logs/precompute.log 2>&1; touch logs/precompute.done"
#   336 batches @ ~800 it/s on CPU
#   Output: latents/latents.pt (89.4 MB, 672 samples)
#   Verified: 0% <unk> in event stream AND description stream

# 4. (RUNNING) Train the Transformer (overnight, ~50 min on CPU)
tmux new-session -d -s train-tx ".venv/bin/python -u -m samar.train_samar_transformer > logs/train_tx.log 2>&1; touch logs/train_tx.done"
#   Expected: 10 epochs @ ~5 min each = ~50 min total
#   Config: 605 train / 67 val, batch=16, context=256
#   Architecture: 6-encoder + 6-decoder, d_model=256, vocab=1254,
#     latent=128, gradient_clip=1.0, warmup=1000 steps, lr=3e-4

# 5. (Pending transformer) Generate MusicXML
python -m samar.generating --latent-index 15 --output-xml generated.xml
```

## Per-epoch transformer loss curve (in progress)

| Epoch | train_loss | val_loss | wall time |
|---|---|---|---|
| 1 | 0.2544 | 0.0603 | ~5 min |

(Subsequent epochs will be added as they complete.)

## Verification

- All 5 smoke tests pass after the round-5 changes
- `samar/precompute_samar_latents.py` produces clean latents (0% `<unk>`)
- `samar/dataset.py` loads all 78 files (was 52)
- `samar/generating.py --help` shows clean argparse

## Files modified

- `samar/precompute_samar_latents.py` (round 5)
- `samar/dataset.py` (round 5)
- `samar/generating.py` (round 5)
- `latents/latents.pt` (regenerated, 89.4 MB)
- `checkpoints/samar_vae.pt` (retrained, 1.95 MB)
- `checkpoints/samar_transformer.pt` (retraining, ~33 MB)
- `logs/*.tfevents.*` (TensorBoard events from training runs)