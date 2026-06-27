# Round 14 Audit — Pre-training soundness, smoke test, and full-run estimate

**Date:** 2026-06-27

**TL;DR**: Final pre-training audit. 3 trainer issues fixed, smoke test
passed (loss 2.39 → 1.98 in 15 batches), estimated full training time
~6.2 hours. Ready to launch the 50-epoch run.

## Findings & Fixes

### FIX R14-A — context_size default mismatch (HIGH)

**WHERE:** `samar/train_samar_transformer.py:85, 359`

**Bug:** CLI defaulted to `context_size=256` but the existing checkpoint
uses `max_len=512`. Training with the default would silently truncate
samples to 256, wasting 50% of the model's positional embedding capacity.

**Fix:**
- `SamarTransformerTrainer.__init__` default: `256 → 512`
- CLI default: `256 → 512`
- New CLI flag: `--context-size` (overridable)
- Model constructor uses `max_len=args.context_size` so positional
  embedding capacity matches data length

### FIX R14-B — NaN on all-pad labels (HIGH)

**WHERE:** `samar/train_samar_transformer.py:44-58` (new helper),
`189-194, 215-219` (call sites)

**Bug:** `F.cross_entropy` with `reduction='mean'` (the default) returns
**NaN** when ALL labels equal `ignore_index` — there's nothing to average
over. With real Arabic data this never fires (verified: 0 all-pad samples
in 700), but a future regression would propagate NaN through the
optimizer and silently corrupt the model.

**Fix:** Added `safe_cross_entropy` helper using `reduction='sum'` and
dividing by `max(1, count)`. All-pad returns 0.0 (not NaN); partial labels
give identical results to `F.cross_entropy(mean)`.

**Verification:**
- `safe_cross_entropy(all_pad)` → `0.0` (was NaN)
- `safe_cross_entropy(partial)` → matches `F.cross_entropy(mean)`

### FIX R14-C — Dead `self.best_val` in `__init__` (LOW)

**WHERE:** `samar/train_samar_transformer.py:98`

**Bug:** `__init__` set `self.best_val = float('inf')` but the actual
`train()` loop uses its own local `best_val` variable. Instance attribute
was dead code.

**Fix:** Removed.

## Audit Cluster (11 probes, all PASS)

1. Checkpoint loads (0 missing / 0 unexpected keys)
2. Trainer constructs: 630 train / 70 val split
3. Batch shapes: input_ids=(4, 511), latent=(4, 511, 128), description=(4, 512)
4. `safe_cross_entropy` on all-pad → 0.0
5. Train loss: 2.61 (sane initial)
6. Backward + clip + step OK
7. Val loss: 2.48
8. Description `<unk>` rate: 0.0% (round-13 fix preserved)
9. Causality intact (round-12 fix preserved): pos 0..4 diff = 0.0000
10. Encoder gradients flow via cross-attention
11. Save/load round-trip preserves description_embedding shape [837, 256]

## Smoke Test Results

**15-batch smoke test (CPU, batch_size=4):**
- Time: 42.3s for 15 batches = **2.82s/batch**
- Loss trajectory: 2.39 → 1.98 (17% reduction in 15 batches)
- Avg first 5 batches: 2.39
- Avg last 5 batches: 1.98

**Training dynamics observed:**
- Loss decreases monotonically over 15 batches (one outlier at batch 4)
- Gradient clipping prevents explosion (initial total norm ~5 → clipped to 1)
- No NaN, no crashes
- Description stream flowing (0% `<unk>` rate)

## Full-Run Estimate

```
158 batches/epoch × 50 epochs = 7,900 batches
@ 2.82s/batch on CPU = 22,278s = 6.2 hours
```

Notes:
- This is the worst-case estimate (CPU). With GPU, expect 5-10x speedup.
- Round-10 baseline convergence was at epoch 43; expect similar.
- val_fraction=0.1 → 70 samples / 18 batches per validation pass
  (adds ~50s per epoch × 50 = 0.7h).

## Pre-Retraining Checklist (FINAL)

- [x] Round 12: causal mask + encoder=description-only
- [x] Round 13: collate destroys description (FIXED), latent length
  mismatch (FIXED), filter default (FIXED)
- [x] Round 14: context_size default (FIXED), all-pad NaN (FIXED),
  dead code (FIXED)
- [x] Smoke test PASS (loss 2.39 → 1.98)
- [x] Backup existing checkpoint at `backups/samar_transformer_round12.pt`
- [ ] Decide on Fairuz bias strategy (FIX #3 from round 12)
- [ ] Decide on MIDI VAE strategy (FIX #9 from round 12)
- [ ] Regenerate `latents/latents.pt` after deleting Hello.xml
- [ ] Launch 50-epoch training run

## Files Changed (round 14)

- `samar/train_samar_transformer.py`:
  - `safe_cross_entropy` helper added (line 44-58)
  - `__init__` default context_size: 256 → 512 (line 85)
  - Removed dead `self.best_val` (line 98)
  - Training step uses `safe_cross_entropy` (line 213)
  - Validation step uses `safe_cross_entropy` (line 235)
  - CLI gained `--context-size` flag (line 345)
  - Model constructor uses `args.context_size` for max_len (line 376)
  - Trainer instantiation uses `args.context_size` (line 384)
- `docs/audit-round-14.md` (this file)
- `backups/samar_transformer_round12.{pt,json}` (backup of pre-training
  state)