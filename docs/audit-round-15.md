# Round 15 Audit — Pre-10-epoch Arabic Retrain Soundness

**Date:** 2026-06-27

**TL;DR**: All 6 round-12/13/14 fixes are preserved (no regressions). One
new trainer bug found and fixed (duplicate `trainer.train()` call). Two
data-pipeline issues documented but **non-blocking** for the 10-epoch run.
Smoke test PASS. Ready to launch ~2.2-hour 10-epoch Arabic retrain on
`latents/latents.pt`.

## Findings & Fixes

### FIX R15-A — Duplicate `trainer.train()` call (HIGH)

**WHERE:** `samar/train_samar_transformer.py:391` (was lines 391-392
before fix)

**Bug:** The CLI entry point called `trainer.train()` twice in
succession. The user passed `--num-epochs=10` but the model actually
trained for 20 epochs. Worse: the second call re-initializes the
optimizer/scheduler inside `train()` and starts fresh, so effectively
epoch 11..20 used an "epoch 1 LR schedule" (warm-up again), not
continued from epoch 10's state.

**Evidence:**
- Original code:
  ```python
  trainer.train(num_epochs=args.num_epochs)
  trainer.train(num_epochs=args.num_epochs)
  ```
- This was an artifact of the patch tool merging two iterations of
  edits (round-9 added `--num-epochs`, round-14 added `--context-size`)
  without deduplicating the final two lines.

**Fix:** Removed the second `trainer.train()` line.

**Verification:** Re-read of the function entry point — only one
`trainer.train(num_epochs=args.num_epochs)` call remains. Smoke test
of 1 batch ran cleanly without doubling.

## Non-blocking observations (documented, not fixed)

### OBS R15-B — Stale Hello.xml/Hello2.xml samples in `latents.pt`

**WHERE:** `latents/latents.pt` (committed in round-9)

**Observation:** The latents file contains 700 samples, but the first 2
entries are precomputed from `Hello.xml` and `Hello2.xml` — files that
were deleted in round-12 (commit `24d2687`). The deletion removed the
source files but `latents.pt` was not regenerated.

**Impact:** 2 / 700 = 0.3% of training data is from the deleted
placeholder files. Their `Composer_example` description token no longer
appears in the live data, but the precomputed IDs are still in the
latents file. They don't crash training (the IDs are valid in
DescriptionVocab space) but they represent stale content.

**Recommended fix:** After this 10-epoch run completes, regenerate
`latents/latents.pt` via `python -m samar.precompute_samar_latents`.
This drops the 2 stale entries and brings the description space fully
in sync with the live `data/xml/` directory.

**Status:** NOT fixed before the 10-epoch run — the impact is minimal
(0.3% of data, valid tokens) and the user wants to start training
ASAP.

### OBS R15-C — Description max length 1411 exceeds model's max_len 512

**WHERE:** `samar/dataset.py:224-226` (truncation logic) +
`samar/models/samar_transformer.py:33` (max_len parameter)

**Observation:** Some Arabic pieces have very long descriptions
(max=1411 tokens, avg=557). The dataset truncates to
`context_size=512`, which fits inside the model's `pos_embedding`
(512 rows, positions 0..511), so no crash. **But 372 of 700 samples
(53%) have descriptions truncated**, losing description content from
positions 512 onward.

**Impact:** The model still sees 1-512 description tokens per sample,
just not the full description for long pieces. Since descriptions are
per-bar statistics (one Bar_N + a few stats), truncating drops the
last ~150-900 bars of description for long pieces.

**Recommended fix (future):** Increase model `max_len` to 2048 or
pre-truncate descriptions during precompute (e.g. cap at 512 tokens).
For the 10-epoch run: leave as-is — the model was designed for
context_size=512 and changing `max_len` requires retraining the
positional embedding from scratch.

**Status:** NOT fixed before the 10-epoch run — same architecture
constraints as before.

### OBS R15-D — Generating.py foot-gun for SamarVocab IDs (DOCUMENTED)

**WHERE:** `samar/generating.py` + `samar/models/samar_transformer.py:230`

**Observation:** The model's `description_embedding` is sized to 837
(DescriptionVocab size). If anyone constructs a description tensor
using `SamarVocab.stoi[TimeSignature_4/4]` (which is ID 1193), the
forward pass crashes with `IndexError: index out of range`.

**Reality check:** `generating.py:169` uses `desc_tokenizer.encode()`
(DescriptionTokenizer → DescriptionVocab IDs in [0, 836]) so the
normal generation path is safe. The dataset collate also correctly
preserves pre-encoded IDs from `latents.pt` (round-13 fix).

**Recommended fix:** None needed — `generating.py` is already correct.
The foot-gun is only triggered if a user manually constructs a
description tensor using the wrong tokenizer, which is a misuse.

**Status:** Documented in the audit; no code change.

## Probe cluster (6 probes, all PASS)

End-to-end with `backups/samar_transformer_round12.pt`:

```
[Config] vocab=1254, d_model=256, latent=128, max_len=512
[Load] missing=0, unexpected=0
[Load] desc_emb = (837, 256)
[Load] token_emb = (1254, 256)

A: Causality
  pos 0: diff=0.000000
  pos 1: diff=0.000000
  pos 2: diff=0.000000
  pos 3 (legit): diff=1.9600
  Result: PASS

B: Sample() with REAL dataset description
  With description: 30 tokens
  Without description: 30 tokens
  Result: PASS

C: Description <unk> rate (round-13 fix preserved)
  <unk> rate: 0.00% (target: 0.0%)
  Result: PASS

D: Latent length matches input_ids (round-13 fix preserved)
  input_ids: (511,), latent: (511, 128)
  Result: PASS

E: Tokenizer roundtrip
  ['Bar_0', 'Position_0', 'Pitch_24EDO_60', 'Duration_8', 'Instrument_Piano']
  -> [5, 954, 910, 670, 682]
  -> ['Bar_0', 'Position_0', 'Pitch_24EDO_60', 'Duration_8', 'Instrument_Piano']
  Result: PASS

F: Forward with real batch (8 samples)
  input_ids: (8, 511), latent: (8, 511, 128), desc: (8, 512), labels: (8, 511)
  logits: (511, 8, 1254)
  Result: PASS
```

## Smoke test results

**Per-batch timing on local CPU (1 batch_size=8 train batch + 1 val batch):**
- Train: 9.55s (forward + backward + step)
- Val: 4.34s (forward only)
- Initial loss: 7.5 (sane for random init)
- save_model works, produces 35.7MB checkpoint

## Reconstruction sanity check

```
Reconstructor: 11/11 examples clean
(0 multi-voice notes, 0 out-of-range octaves, all parse with strict ET)
```

Round-11 reconstructor fixes still hold.

## 10-epoch run plan

**Command:**
```bash
python -m samar.train_samar_transformer \
    --num-epochs 10 \
    --lr 3e-4 \
    --context-size 512 \
    --latent-path latents/latents.pt
```

**Expected duration:**
- 79 train batches + 9 val batches per epoch
- ~794s/epoch on local CPU (13 min/epoch)
- **10 epochs ≈ 2.2 hours**

**Expected convergence pattern:**
- Round-9 (50 epochs, no causal mask, broken description): breakthrough at epoch 43, final val=0.106
- This run (10 epochs, with causal mask + working description): expect similar or
  slightly faster convergence because description conditioning now provides
  real signal. Hard to predict val_loss at epoch 10; should be < 2.0 by
  epoch 5 and < 1.0 by epoch 10 (rough estimate based on round-9 trajectory).

**Pre-launch checklist:**
- [x] Round-15: duplicate train() call fixed
- [x] All round-12/13/14 fixes verified (probe cluster PASS)
- [x] Reconstructor verified clean on all 11 examples
- [x] Smoke test PASS (1 batch forward+backward+step works)
- [x] Checkpoint restored from round-12 backup (was overwritten by smoke test)
- [x] Latents file exists (700 samples, including 2 stale Hello entries — OBS R15-B)
- [ ] Decide: launch 10-epoch now and regenerate latents after, OR regenerate latents first

**Recommendation:** Launch 10-epoch run NOW with the existing
`latents.pt`. The 2 stale Hello entries are 0.3% of data and don't
crash training. After the run completes, regenerate `latents.pt` to
clean up.

**Files Changed (round 15):**
- `samar/train_samar_transformer.py` — removed duplicate `trainer.train()` call (line 391)
- `docs/audit-round-15.md` (this file)