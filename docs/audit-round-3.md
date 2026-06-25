# SAMAR Audit — 2026-06-25 Round 3

Round 3 ran the 7-probe vocab-checkpoint-triangle cluster plus
six additional probes (transition-state, trainer-forward, VAE,
training-loop, dataset-quality, generating-path, field-read).

Round 1 (`6d15b00`) ran probes A-D only, declared success.
Round 2 (`4ade83d`) ran probes E-H and fixed 5 critical bugs.
Round 3 (this) re-ran the full cluster PLUS the probes that
weren't part of the cluster. Found **5 more bugs that rounds 1 and 2
both missed** because they didn't probe the trainer, the latents
filter, or the JSON config file.

## Critical bugs (this round)

### C1 `train_samar_transformer.py` STILL calls `self.model(input_ids, tgt=latent, ...)` in `training_step` and `validation_step`

**WHAT**: Round 1 (commit `6d15b00`) "fixed" finding #3 by changing
the keyword in the `forward` method (line 49). But the trainer's
**actual code paths** — `training_step` (line 62) and
`validation_step` (line 84) — still call `self.model(input_ids,
tgt=latent, description=description)`.

**WHERE**:
  `samar/train_samar_transformer.py:62` — `training_step`
  `samar/train_samar_transformer.py:84` — `validation_step`

**VERIFIED**:
  ```
  Model.forward signature: ['self', 'input_ids', 'latent', 'description', 'tgt']
  Call sites in trainer:
    L49:   return self.model(input_ids, latent=latent, description=description)  # OK
    L62:   predicted_latent = self.model(input_ids, tgt=latent, description=description)  # WRONG
    L84:   predicted_latent = self.model(input_ids, tgt=latent, description=description)  # WRONG
  ```

**WHY IT MATTERS**: The `SamarTransformer.forward` signature is
`(input_ids, latent=None, description=None, tgt=None)`. Calling
`self.model(input_ids, tgt=latent, ...)` would:
  - Pass the latent vector as `tgt=` (which the model currently
    ignores — the `tgt` arg is unused)
  - Leave `latent=None`, so `self.latent_embedding` is never
    called
  - Pass `description` correctly

Result: the model never sees the latent. Training is functionally
identical to "no latent" — the model is fitting events only and
ignoring the conditioning signal entirely.

**FIX**: Change `tgt=latent` to `latent=latent` on lines 62 and 84.

### C2 `SamarTransformerTrainer.forward` method is over-indented AND dead code

**WHAT**: The `forward` method at lines 44-49 has lines 45-49
over-indented by 4 spaces. Even when syntactically correct, it is
**dead code** — neither `training_step` nor `validation_step` calls
`self.forward(...)`; they call `self.model(...)` directly.

**WHERE**: `samar/train_samar_transformer.py:44-49`

**FIX**: Either delete it (preferred) or fix indentation AND switch
`training_step`/`validation_step` to call it. The current state
preserves the broken-on-purpose from round 1 (round 1 "fixed" the
wrong line).

### C3 `SamarLatentDataset.__getitem__` filter `len >= 256` removes ALL samples

**WHAT**: The filter at line 126 keeps only samples with
`len(tokens) >= context_size` (256). But all 529 precomputed
latents have length 30..255 (most exactly 255). Result:
`len(self.samples) == 0` after filtering.

**VERIFIED**:
  ```
  Total samples:           529
  After filter (>=256):    0
  Unique lengths:          min=30, max=255, mode=255
  Filtered out:            529 / 529
  ```

**WHERE**: `samar/dataset.py:125-127`

**WHY IT MATTERS**: `SamarTransformerTrainer` (line 36) loads
`SamarLatentDataset(latent_path, context_size=256, ...)`. The
dataset is empty. The trainer would iterate over 0 batches per
epoch. **Training cannot start.**

**FIX OPTIONS**:
  - Option A: Change filter to `>` to allow >= 255 (silently truncates)
  - Option B: Pad samples shorter than `context_size` to exactly
    `context_size`
  - Option C: Recompute latents with proper context_size and ensure
    all samples are at least `context_size` long (FIX the upstream
    chunker, not the filter)

The correct architectural fix is C — but for a quick win, A is
shorter and works.

### C4 `checkpoints/samar_transformer_config.json` has `vocab_size=1129` (stale after round-2 fix)

**WHAT**: `generating.py:36-37` loads
`checkpoints/samar_transformer_config.json` and passes it to
`SamarTransformer(**lm_config)`. The JSON has
`"vocab_size": 1129` (round-1 value). But round 2 extended the
live vocab to 1249.

**VERIFIED**:
  ```
  Live SamarVocab size:      1249
  Config vocab_size:         1129
  Checkpoint embedding shape: [1129, 256]
  ```

**WHERE**:
  - JSON file: `checkpoints/samar_transformer_config.json`
  - Reader: `samar/generating.py:34-43`

**WHY IT MATTERS**: `generating.py` creates the model with
`vocab_size=1129`. `from_pretrained` then loads the 1129-row
checkpoint embedding. Any token with ID >= 1129 (e.g. `Pitch_24EDO_158`)
will cause an OOB indexing error during generation. So **the round-2
fix to extend the vocab was effectively a no-op** for inference —
generating can't use the new tokens.

**FIX**: Regenerate `samar_transformer_config.json` with
`vocab_size=1249`. Also update the train script's hardcoded
`vocab_size=1129` (line 164) to match.

### C5 3 XML files fail to parse (`ValueError: invalid literal for int() with base 10: 'X1'`)

**WHAT**: `MusicXMLParser._parse_time_signatures_by_bar` (line 105)
does `int(measure.attrib.get("number", 1))`. Some MusicXML files
have measure elements with `number="X1"` (an editorial / structural
marker, not a numeric bar number).

**VERIFIED**:
  ```
  FAIL abdel_halim_hafez_fouq_alshouk_bayat_1958.xml: invalid literal for int() with base 10: 'X1'
  FAIL fairuz_hela_ya_wasea_bayat_1972.xml: invalid literal for int() with base 10: 'X1'
  FAIL fairuz_jadaka_alghithu_huzam_1960.xml: invalid literal for int() with base 10: 'X1'
  ```

**WHERE**: `samar/core.py:105` (`MusicXMLParser._parse_time_signatures_by_bar`)

**WHY IT MATTERS**: 3 of 52 training files (5.8%) silently dropped
from training. The dataset would log `Failed to process ...` and
continue. The 3 dropped files include 2 Bayat pieces and 1 Huzam
piece — the Huzam is especially costly because only 3 Huzam files
exist in the corpus.

**FIX**: Wrap the `int(...)` in a try/except; fall back to using
the previous bar's number or 1.

## Medium bugs

### M1 `SamarLatentDataset.__getitem__` doesn't return `'description'` key

**WHAT**: Even if `samar_collate_fn` can handle `'description'`,
`SamarLatentDataset.__getitem__` only returns `{'input_ids', 'labels',
'latent'}`. So description-conditioning is plumbed through the data
pipeline but never gets a tensor to condition on.

**WHERE**: `samar/dataset.py:132-140`

**WHY IT MATTERS**: `description` is silently `None` during training.
The trainer accepts `description=None` (the model handles it). So
the model learns to ignore descriptions and only the latent vector
is used. This is consistent with the trainer not actually passing
the latent (C1) — but means description-conditioning is doubly dead.

**FIX**: Add description fields to the latents.pt file (recompute
from XML), then have `__getitem__` return `description=tokens[idx]
encoded against DescriptionVocab`.

### M2 `vocab_size=1129` hardcoded in `train_samar_transformer.py:164`

**WHAT**: Same drift as C4 but on the training side.
`SamarTransformer(d_model=256, ..., vocab_size=1129, latent_dim=128)`
builds a 1129-token model. Even after the fix in C4, the train
script will train a 1129-token model while the live vocab has 1249.

**WHERE**: `samar/train_samar_transformer.py:164`

**FIX**: Change `vocab_size=1129` to `vocab_size=1249`.

### M3 Dataset is 67% Fairuz / 14% Bayat — heavy bias

**WHAT**: Of the 52 XML files in `data/xml/`:
  - **35 files are Fairuz (67%)**
  - 6 are Abdel Halim Hafez (12%)
  - 5 are Asmahan (10%)
  - 4 are Sheikh Imam (8%)
  - 2 are Hello/Hello2 (test files, 4%)

Maqam distribution is similarly skewed:
  - Bayat: 14 files (27%)
  - Nahawand: 10 files (19%)
  - Kurd: 8 files (15%)
  - Ajam: 6 files (12%)
  - Rast: 4 files (8%)
  - Huzam: 3 files (6%)
  - Hijaz: 3 files (6%)
  - Saba: 1 file (2%)

**VERIFIED**: Direct count from `data/xml/` filenames.

**WHERE**: `data/xml/` directory itself.

**WHY IT MATTERS**: A model trained on this corpus will be biased
toward Fairuz's vocal style, Bayat maqam, and 1970s-era Egyptian
arrangements. Generation outside that distribution will be poor.
**Saba, Sikah, Saz, Awj, Iraq maqamat are absent or near-absent.**

**FIX**: Either:
  - Stratified sampling during training (oversample rare maqamat)
  - Augment with additional data (see "Next steps" section)
  - Accept the bias and document it

### M4 Pre-computed latents contain 23.6% `<unk>` (stale)

**WHAT**: The bundled `latents/latents.pt` was generated against
the pre-round-2 tokenization (event stream had 30% `<unk>`). The
top unknown categories are: `Instrument_Part_1`, `MeanPitch_*`,
`NoteDensity_*`, `Pitch_24EDO_144+`.

**VERIFIED**:
  ```
  Latents sample 100: <unk> = 5972/25350 = 23.6%
  ```

**WHERE**: `latents/latents.pt` (binary file, 67M).

**WHY IT MATTERS**: Even after fixing the round-2 bugs, any
training run that loads these latents will fit the model on
poisoned data. The trainer must regenerate them first.

**FIX**: Recompute via `python -m samar.precompute_samar_latents`
AFTER fixing C1 (so VAE input has correct shape) and AFTER fixing
C5 (so all 49 of 52 files contribute).

### M5 `generating.py:20` loads vocab pickle at module import time

**WHAT**: The line `tokenizer = SamarTokenizer.load(...)` runs at
import time. Same for VAE and transformer checkpoint loads
(lines 27-30 and 43-48). Anyone who imports `samar.generating`
for any reason (e.g. just to read a docstring) triggers 50MB+ of
disk I/O.

**VERIFIED**:
  ```
  python -c "import samar.generating"  # takes ~3 seconds, loads ckpt
  ```

**WHERE**: `samar/generating.py:20, 27-30, 43-48`

**FIX**: Wrap the tokenizer/checkpoint loads in
`if __name__ == '__main__':` so they only run on direct execution.

## Low / doc bugs

### L1 README claims "vocab_size=1249" without noting the JSON is stale

**WHERE**: `README.md` — "Known limitations" mentions that the
checkpoint is stale but doesn't mention the JSON config is also
stale.

**FIX**: Update the "Known limitations" section to call out
`samar_transformer_config.json` explicitly.

### L2 README claims "49/52 files parse" but doesn't say which 3

**FIX**: Add the 3 filenames to the README or to the docs.

### L3 `samar_transformer_config.json` is committed to git

**WHY IT MATTERS**: It's a derived artifact (regenerated by the
trainer after every training run). Committing it means:
  - Out-of-date JSON on disk after any retrain without JSON update
  - Couples checkpoint state with vocab state — they're easy to
    desync (see C4)

**FIX**: Add `checkpoints/samar_transformer_config.json` to
`.gitignore`. Re-generate it as part of `train_samar_transformer.py`'s
`save_model()`.

## What's still good (from rounds 1 + 2)

  - Event stream `<unk>` ratio: 0% on real data
  - Description stream `<unk>` ratio: 0% on real data
  - VAE pipeline end-to-end works (forward, encode_latent, compute_loss)
  - All 5 smoke tests pass
  - `from_pretrained` handles vocab-size extension
  - Token embedding extension via zero-padding works

## Verification

```
Probe A: Live=1249, Pickled=1249, Ckpt_state_dict=1129
  → STALE checkpoint (vocab_size=1129, but live vocab is 1249)
  → Fix: regenerate checkpoint after retraining

Probe B: Description <unk> ratio = 0.00%  (passes)

Probe C: DescriptionVocab overlap with SamarVocab: 38 tokens (instrument, bar, chord overlap)
  → Acceptable (matches FIGARO pattern of partial overlap)

Probe D: Description tokens are quantized to bin indices (matches vocab)
  → Passes

Probe E: Event <unk> ratio = 0.00%  (passes after round-2 fix)

Probe F: Description-only tokens NOT emitted in event stream
  → Passes (after round-2 fix)

Probe G: Vocab max pitch 24-EDO 263 (= MIDI 131); data max 188 (= MIDI 94)
  → Vocab has headroom, no overflow

Probe H: Instrument parser fallback works (Part_N → Voice)
  → Passes

NEW probes this round:
  Trainer forward call:     FAIL (lines 62, 84 use tgt=latent)
  SamarLatentDataset filter: FAIL (filter too strict, removes all)
  JSON config vocab_size:   FAIL (1129, should be 1249)
  3 unparseable files:      FAIL (X1 in measure number)
  Data bias:                Bayat-heavy (27%), Saba only 1
  Latent <unk> ratio:       23.6% (stale, must recompute)
```