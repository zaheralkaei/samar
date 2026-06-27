# Round 16 audit — pre-train verification + R16-A fix

## Date: 2026-06-27

## Pre-train audit cluster (13 probes)

Following the `pre-train-retrain-checklist.md` protocol, ran the full
13-probe cluster against the round-15 checkpoint before launching a
fresh retrain. Script: `_verify_round16.py`.

| # | Probe | Result |
|---|-------|--------|
| 1 | Checkpoint load (no missing/unexpected keys, desc_emb sized to DescriptionVocab) | PASS |
| 2 | Causality (causal mask) | PASS (max diff positions 0..T-2 = 0.000000, T-1 diff = 2.76) |
| 3 | sample() end-to-end with real description | PASS (gen shape [1, 30]) |
| 4 | Description <unk> rate via collate | PASS (0.000%) |
| 5 | Latent length matches input_ids | PASS (0/50 mismatches) |
| 6 | Tokenizer roundtrip | PASS |
| 7 | Smoke test (1 batch forward+backward+step, backup-restore) | PASS (train_loss=1.34, val_loss=1.42) |
| 8 | CLI help parses | PASS |
| 9 | Duplicate CLI calls (round-15 foot-gun) | **FAIL** |
| 10 | Post-patch hygiene (py_compile) | PASS |
| 11 | Stale file entries in latents.pt | PASS (0 in first 200) |
| 12 | Description max length vs model max_len | PASS (no truncations) |
| 13 | Token <unk> rate in latents | PASS (0/51100) |

## R16-A: Duplicate trainer.train() call — FOUND AND FIXED

### What

`samar/train_samar_transformer.py` lines 420-421 contained TWO
identical `trainer.train(num_epochs=args.num_epochs)` calls.

### History

- Round 9 (commit `764a4b9`) added the first call when wiring up the
  new `--num-epochs` CLI flag
- Round 14 (commit `5be1370`, "trainer hardening") accidentally added
  a SECOND call on top
- Round 15 commit `59ffe73` message claimed: "R15-A: Remove duplicate
  trainer.train() call (would have trained 20 epochs instead of 10)"
  but the fix did NOT actually land in the file — the duplicate was
  still present when training ran on June 27

### Effect

- Every `--num-epochs N` invocation was actually running **2N epochs**
- The LR schedule reset midway (second call created fresh warmup_steps=1000)
- Round-15 "val=1.70 at epoch 10" was actually the SECOND
  `trainer.train()` call resuming from val=1.70 with fresh warmup —
  those extra epochs were near-no-op
- Three "continue" runs (`training_round15_continue*.log`) each did
  duplicate-call cycles, which is why their loss curves were so flat
  (1.37 → 1.37 → 1.37)

### Fix

Removed the duplicate call. Added verification comment with the
`grep -c 'trainer\.train('` check so the next audit catches it again.

## Round 16 retrain: SUCCESS

After fix, fresh 10-epoch run from scratch:

```
Epoch 1/10  train_loss=6.7268  val_loss=5.7093
Epoch 2/10  train_loss=5.4248  val_loss=4.8700
Epoch 3/10  train_loss=4.7509  val_loss=4.0495
Epoch 4/10  train_loss=3.7938  val_loss=3.0965
Epoch 5/10  train_loss=2.9273  val_loss=2.4903
Epoch 6/10  train_loss=2.3768  val_loss=2.1534
Epoch 7/10  train_loss=2.0665  val_loss=1.9464
Epoch 8/10  train_loss=1.8834  val_loss=1.8229
Epoch 9/10  train_loss=1.7651  val_loss=1.7114
Epoch 10/10 train_loss=1.6827  val_loss=1.6508
```

Wall time: ~83 min on CPU.

## Quality progression round-15 → round-16

| Metric | Round-15 | Round-16 | Change |
|--------|----------|----------|--------|
| Val loss final | 1.70 (with bug) | **1.65** | -0.05 |
| Avg notes/file (temp=1.0) | 64 | **106** | +66% |
| Avg microtones/file | ~10 | ~9 | roughly equal |
| 0 multi-voice notes | yes | yes | (round-11 fix preserved) |
| Strict ET parse | yes | yes | |

The note count jump is the cleanest signal that R16-A was a real fix:
the same architecture, same data, same epoch count, same temperature —
just without the duplicate call that was silently doubling work and
resetting LR schedule midway.

## Files

- `samar/train_samar_transformer.py` — R16-A fix
- `checkpoints/samar_transformer.pt` — round-16 model (35.7MB, val=1.65)
- `checkpoints/samar_transformer_config.json`
- `latents/latents.pt` — unchanged from round-15 (still 698 clean samples)
- `backups/samar_transformer_round15_pre_round16.pt` — round-15 backup
- `backups/samar_transformer_config_round15_pre_round16.json`
- `training_round16.log` — full training log
- `_verify_round16.py` / `_verify_round16.json` — audit trail
- `_smoke_round16.py` — smoke test scratch
- `examples/13-19_*.{xml,txt}` — 7 new round-16 examples
- `examples/README.md` — updated with round-16 row and quality progression

## Lessons learned

1. **Verification comments > assumed fixes.** Round-15's commit message
   said "R15-A: Remove duplicate" but the actual code still had it.
   Verification of post-fix behavior is not optional.
2. **Multi-round CLI-flag patches accumulate duplicate calls.** Every
   time you add a CLI flag in a `__main__` block that touches the same
   function, run `grep -c '<func>(' file.py` to catch duplicate calls.
3. **Silent schedule resets are worse than crashes.** A duplicate call
   that "only" resets the LR schedule looks healthy — val loss still
   decreases, checkpoints still save — but every epoch is being
   trained with two different schedule phases concatenated.
4. **The verification cluster is high-leverage.** 12/13 probes passed
   in <60 seconds; the 1 failure was a critical infrastructure bug
   that had been silently corrupting training for two rounds.

## Next round (round-17) considerations

- Same data, more epochs (20-30) to see if the model breaks through
  the val=1.6 plateau — the round-10 conclusion ("data is the
  bottleneck") suggests this will plateau around val=1.0-1.5
- OR expand the Arabic dataset first (the round-10 conclusion's
  actual recommendation) — adding more files to `data/xml/` then
  regenerating latents
- The duplicate-call probe (`grep -c 'trainer\.train('` in the
  `__main__` block) should be a permanent part of every pre-train
  audit