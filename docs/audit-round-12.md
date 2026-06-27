# Round 12 Audit — Architecture / Training / Post-Training Soundness

**Date:** 2026-06-27

**TL;DR**: The model was non-causal at the architectural level. During training,
position `t` saw the answer at `t+1` via bidirectional self-attention, so the
model learned a peek-cheating distribution that doesn't transfer to inference.
This is the root cause of the bad generation output. Round 12 added a causal
mask, restricted the encoder to description-only (preventing event-side
information leakage via cross-attention), and fixed several downstream issues.

## Findings & Fixes

### FIX #1 — Causal mask on decoder (CRITICAL)

**WHERE:** `samar/models/samar_transformer.py:60-67, 79-200`

**Bug:** `nn.Transformer` does not apply a causal mask by default. Without it,
position `t` could attend to position `t+1` (and beyond), so the model learned
to peek at the answer during training. At inference, the autoregressive loop
only feeds back the prefix, so the model's predictions were off-policy.

**Evidence (probe on real trained checkpoint, before fix):**
- Forward with `input_ids=[Bar_0, Pitch_86, Velocity_16]` (length 3) vs
  `input_ids=[Bar_0]` (length 1): position 0's argmax changes from
  `Instrument_Piano` to `Pitch_24EDO_127`. L1 distance between softmax
  distributions = 1.20.
- Per-position max-diff when only the last token changes: pos 0 = 5.35,
  pos 1 = 2.41, pos 2 = 6.01 — every position depends on the future.

**Fix:** Added `tgt_mask=nn.Transformer.generate_square_subsequent_mask(T, device)`
to the `self.transformer(src, tgt, ...)` call in `forward()`.

**Verification (probe after fix, on fresh small model):**
- Per-position max-diff when only the last token changes: pos 0 = 0.0000,
  pos 1 = 0.0000, pos 2 = 0.0000, pos 3 = 0.5082 (legitimate).
- Causal property restored. Requires retraining to take effect — the
  existing checkpoint was trained peek-cheating.

**FIGARO comparison:** FIGARO uses HuggingFace `EncoderDecoderModel` with
`decoder.is_decoder=True` (HF default = causal mask). SAMAR used raw
`nn.Transformer` (no causal mask by default) and silently dropped the
causal mask somewhere along the way. The fix restores the FIGARO-aligned
behavior.

### FIX #1b — Encoder restricted to description-only (CRITICAL)

**WHERE:** `samar/models/samar_transformer.py:79-200`

**Bug:** Before round 12, the encoder was fed `description + events` and
processed bidirectionally. The encoder output at position 0 therefore
encoded information about future event tokens, which leaked through
cross-attention and let the decoder peek at the answer even with a
decoder-side causal mask.

**Fix:** Restructured `forward()` so the encoder sees description only
(bidirectional over description tokens), and the decoder sees event tokens
(causal self-attention) plus cross-attention to the encoder output. This
matches the FIGARO encoder-decoder split cleanly: the encoder's job is
"describe the piece context"; the decoder's job is "given the context,
generate the next event."

**Verification:** Same causality probes as #1. The encoder-side leak is
no longer possible because the encoder has no event tokens in its input.

### FIX #2 — Delete Hello.xml / Hello2.xml from training set

**WHERE:** `data/xml/Hello.xml`, `data/xml/Hello2.xml`

**Bug:** Two MuseScore placeholder files were in the training set. Both
have `<work-title>Hello</work-title>` / `Untitled score` and
`<creator type="composer">example</creator>` / `Composer / arranger`. The
"example" composer credit produced a `Description_Composer_example` token
that doesn't appear in any real file.

**Fix:** Deleted both files. The Arabic dataset went from 78 files to 76.

**Action required:** Regenerate `latents/latents.pt` via
`python -m samar.precompute_samar_latents` before retraining.

### FIX #3 — Fairuz corpus bias (DOCUMENTED, RETRAINING DECISION)

**WHERE:** `data/xml/` (file naming convention)

**Bug:** 247 of 700 Arabic samples (35.3%) come from filenames starting with
`fairuz_*`. Top-5 first-words: `fairuz` 35.3%, `abdel` 13.9%, `asmahan`
10.6%, `sheikh` 6.7%, `kifak` 6.7%. The model overfits to Fairuz's style
when trained on this corpus.

**Recommended fix (user decision required):**

- **(a) Downsample Fairuz to 50 samples** — caps the dominant singer, gives
  ~150 samples per major singer. Drops total to ~400 samples. Requires
  retraining.
- **(b) Upweight non-Fairuz samples** — repeat non-Fairuz samples 2-3× to
  reach ~700 total. Keeps Fairuz dominant but balances signal.
- **(c) Leave as-is** — document the bias, condition on singer explicitly
  during sampling. Cheapest, least effective.

**Status:** Not auto-applied. Pick (a), (b), or (c) before retraining.

### FIX #4 — description_embedding sized to DescriptionVocab (837) (HIGH)

**WHERE:** `samar/models/samar_transformer.py:32-71`

**Bug:** `description_embedding` was `nn.Embedding(vocab_size=1254, d_model)`,
but `DescriptionTokenizer` only produces IDs in `[0, 837)`. Rows 837-1253
were never reachable, so they stayed at random init forever — 33% of
description_embedding capacity was dead weight.

**Fix:** Added `desc_vocab_size` parameter to `SamarTransformer.__init__()`,
defaulting to `len(DescriptionVocab())` (=837). `from_pretrained` handles
backward compat: if an old checkpoint has description_embedding with 1254
rows, the extra rows are trimmed (they were never trained anyway).

**Verification:** Loaded existing checkpoint with 1254-row desc_emb →
trimmed to 837. No missing/unexpected keys.

### FIX #5 — Cross-vocab ID consistency warning (HIGH)

**WHERE:** `samar/vocab.py:DescriptionVocab.__init__`

**Bug:** `SamarVocab` and `DescriptionVocab` overlap on 716 strings
(Bar_*, Chord_*, TimeSignature_*, Instrument_*) but assign DIFFERENT IDs.
If anyone encodes a description-side token with the event tokenizer (or
vice versa), the wrong token comes out silently.

**Fix:** Added a runtime check in `DescriptionVocab.__init__` that warns
on first construction if any non-special token has a different ID in the
two vocabs. The warning identifies which tokens are at risk.

**Verification:** Warning fires with 107 mismatches and sample examples
(TimeSignature_1/2: 748 vs 1132, etc.).

### FIX #6 — Sampler pre-encodes description once (HIGH)

**WHERE:** `samar/models/samar_transformer.py:201-242`

**Bug:** `sample()` called `self(generated, latent, description)` at every
step, re-running the encoder on the same static description every step.
For a 256-step generation, the encoder ran 256 times instead of once.

**Fix:** Added `_encode_description()` method that runs the encoder once.
Added `enc_output` parameter to `forward()` that bypasses the encoder when
provided. `sample()` now calls `_encode_description()` once at the start
and passes the cached output to all subsequent `forward()` calls.

**Verification:** End-to-end sample() works correctly with both
`description=None` (MIDI path) and `description=tensor` (Arabic path).

### FIX #8 — Filter short samples in SamarLatentDataset (MEDIUM)

**WHERE:** `samar/dataset.py:138-167`

**Bug:** The previous behavior right-padded short samples to `context_size`
with `pad_id`. For a 154-token sample (Hello.xml), 102 of 256 positions
were pad. The loss is correctly masked via `ignore_index=0`, but the
model's positional embeddings at those padding positions never received
gradients and stayed at random init.

**Fix:** Added `min_keep_len` parameter to `SamarLatentDataset.__init__`,
defaulting to `context_size // 2`. Samples shorter than this threshold
are filtered out before training.

**Verification:** With `min_keep_len=128` (default), a 50-token mock
sample is filtered. With `min_keep_len=0`, no filtering.

### FIX #9 — VAE was trained on Arabic but reused for MIDI (DOCUMENTED)

**WHERE:** `samar/precompute_midi_latents.py:82-89`

**Bug:** `samar_vae.pt` was loaded for MIDI encoding, but the VAE was
trained only on Arabic MusicXML. The MIDI encoder maps MIDI tokens into
a latent space that was learned from Arabic data — the resulting latents
don't represent "Bach-ness" but a confused blend of Arabic-trained and
MIDI-projected values. The latent signal during MIDI training is largely
noise.

**Why the MIDI experiment still worked:** Round-10 showed MIDI converged
75x faster than Arabic. This is because the event stream alone (without
latent) carries enough signal for the transformer to learn. The latent is
essentially wasted capacity.

**Recommended fix (user decision required):**

- **(a) Train a separate VAE on MIDI data** — clean fix, requires training
  time.
- **(b) Drop latent conditioning for MIDI** — set `latent=None` in the
  MIDI training pipeline. Faster and probably sufficient given that MIDI
  training converged quickly without good latent signal.

**Status:** Not auto-applied. Pick (a) or (b) before retraining.

### FIX #11 — Updated dead bar_offset comment (LOW)

**WHERE:** `samar/reconstructor.py:163-173`

**Bug:** Round 11 removed a dead `bar_offset` reference but the comment
block still described the old semantics.

**Fix:** Updated comment to mention both round-9 (sequential bar counting)
and round-11 (dead-code removal).

## What's NOT changed

### Decoder input handling

The decoder still receives `input_ids` directly (without explicit shift)
and the `labels = input_ids[1:]` shift is handled by the trainer's
CrossEntropy loss. This is the standard autoregressive LM pattern. The
causal mask ensures that during forward pass, position `t` cannot see
position `t+1` in the decoder self-attention, so logits[t] genuinely
predicts labels[t] = input_ids[t+1].

### Two-stream design

The FIGARO two-stream split (description + events, encoded with separate
vocabs) is correctly enforced in `core.py`, `dataset.py`, `generating.py`,
`precompute_samar_latents.py`, and `precompute_midi_latents.py`. No
changes needed.

### Tokenizer round-trip

The pickle-vocab / live-vocab / checkpoint triple still agrees at 1254
rows. No drift detected.

## Verification cluster (run before retraining)

After all fixes land, run these probes to confirm:

1. **Causality probe** — `(input_ids=[5,7,9,11] vs [5,7,9,13], description=None)` →
   `model(forward)` should give identical outputs at positions 0, 1, 2.
2. **Sample() smoke test** — `model.sample(start, latent, description=desc, max_length=20)`
   should produce 20+ tokens without crashing.
3. **Backward compat** — `SamarTransformer.from_pretrained('checkpoints/samar_transformer.pt')`
   should load with 0 missing / 0 unexpected keys (old 1254-row
   description_embedding is trimmed to 837).
4. **Reconstruction** — Run `reconstruct_musicxml_from_events(events, out.xml)`
   on each example.txt and confirm 0 multi-voice notes, 0 out-of-range
   octaves, parse-clean XML.
5. **Dataset filter** — `SamarLatentDataset('latents/latents.pt', context_size=256)`
   should report filtering N samples shorter than 128 tokens (will be 0
   after deleting Hello.xml + regenerating latents).

## Pre-retraining checklist

Before kicking off a new training run (round-13):

- [ ] Decide on Fairuz bias strategy (FIX #3)
- [ ] Decide on MIDI VAE strategy (FIX #9)
- [ ] Regenerate `latents/latents.pt` after deleting Hello.xml (FIX #2)
  — `python -m samar.precompute_samar_latents`
- [ ] (Optional) Regenerate `latents/midi_latents.pt` if changing VAE
  strategy
- [ ] Retrain Arabic model with the architectural fixes (50 epochs
  recommended, per round-10)
- [ ] Verify val_loss curve is smooth and converges (round-10 saw
  breakthrough at epoch 43; with the fixes, expect slower convergence
  initially because the model can no longer peek-cheat, but a more
  meaningful final loss)

## Files Changed

- `samar/models/samar_transformer.py` — causal mask, encoder=description-only,
  description_embedding size, _encode_description, sample() optimization
- `samar/vocab.py` — cross-vocab ID consistency warning
- `samar/dataset.py` — min_keep_len filter in SamarLatentDataset
- `samar/reconstructor.py` — comment update
- `data/xml/Hello.xml`, `data/xml/Hello2.xml` — DELETED
- `checkpoints/samar_transformer_midi_config.json` — created (round 11)