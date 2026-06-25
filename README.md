# SAMAR — Arabic Music Generation with 24-EDO Support

Controllable symbolic music generation for Arabic maqam music, extended from
the [FIGARO paper](references/5189_figaro_controllable_music_gene.pdf) to
accept 24-tone-equal-octave (24-EDO) pitch alterations — i.e. quarter-tone
microtonality, which is central to Arabic, Turkish, and Persian makam music.

## Repo layout

```
samar/
├── samar/                       # core package (parser, models, training)
│   ├── constants.py             # REMI+ key constants (BAR, POSITION, PITCH, ...)
│   ├── core.py                  # consolidated MusicXML → event representation
│   ├── parser.py                # MusicXML parser (SamarNote, MusicXMLParser)
│   ├── metadata_extractor.py    # key/tempo/time-signature extraction
│   ├── input_representation.py  # REMI+ event builder
│   ├── reconstructor.py         # events → MusicXML (24-EDO aware)
│   ├── chord_recognition_xml.py # chord recognition over MusicXML
│   ├── tokenizer.py             # vocab-aware encode/decode
│   ├── vocab.py                 # REMI+ vocabulary
│   ├── dataset.py               # PyTorch datasets & dataloaders
│   ├── models/
│   │   ├── samar_vae.py         # VQ-VAE for description-token compression
│   │   └── samar_transformer.py # autoregressive decoder over latents + tokens
│   ├── train_samar_vae.py       # train the VAE
│   ├── train_samar_transformer.py  # train the transformer
│   ├── generating.py            # sample (VAE + transformer) → events → MusicXML
│   ├── precompute_samar_latents.py
│   └── tests/                   # smoke tests
│
├── data/
│   ├── xml/                     # 52 training XML files (Fairuz, Sheikh Imam, Asmahan, ...)
│   └── midi/                    # Arabic_Music_Dataset (.mscz + .xml + .pdf per piece)
│                                # samples_rast/ for a single maqam test sample
│
├── checkpoints/                 # trained model weights (.pt, .ckpt)
├── latents/                     # pre-computed VAE latents for training the transformer
├── logs/                        # TensorBoard event files
│
├── figaro/                      # upstream FIGARO paper repo (reference, unmodified)
├── references/                  # papers, thesis, REMI upstream repo
└── archive/                     # superseded iterations (kept for recovery)
    ├── 0_18_05_2025_superseded/
    ├── 0_merged_superseded/
    ├── 000_figaro_annotated_study/   # hand-annotated FIGARO study copy
    ├── SAMAR2_initial/               # original SAMAR2 iteration
    └── XML_intermediate/             # XML-only iteration before consolidation
```

## Quick start

```bash
# from project root
python -c "import sys; sys.path.insert(0, '.'); import samar"
```

## Pipeline overview

The pipeline follows the FIGARO paper's two-stream design (see
`figaro/src/input_representation.py`):

1. **Parse** — `MusicXMLParser` reads MusicXML files (24-EDO alters supported)
   into `SamarNote` objects.
2. **Description stream** — `SAMARInputRepresentation._build_description_tokens()`
   emits per-bar tokens: `Bar_N`, `TimeSignature_N/M`, `MeanPitch_BIN`,
   `MeanVelocity_BIN`, `MeanDuration_BIN`, `NoteDensity_BIN`. Encoded by
   `DescriptionTokenizer` against `DescriptionVocab`.
3. **Event stream** — `SAMARInputRepresentation._build_remi_events()` emits
   per-note tokens: `Position`, `Pitch_24EDO`, `Velocity`, `Duration`,
   `Instrument` (plus a single global `Tempo` token at the start). Encoded
   by `SamarTokenizer` against `SamarVocab`.
4. **Train** — `train_samar_vae.py` learns discrete latents over the
   description stream; `train_samar_transformer.py` learns an
   autoregressive decoder over (latent, description, events).
5. **Generate** — `generating.py` samples event tokens → decodes →
   reconstructs MusicXML via `reconstructor.py`.
6. **Reconstruct** — `reconstructor.py` writes events back to a valid
   MusicXML 4.0 file with `<alter>` elements carrying quarter-tone values.

## Status

This is an active research codebase. Checkpoints and latents are committed
for reproducibility. The `archive/` folder holds earlier iterations and can
be deleted once the project stabilizes.

## Regenerating the vocabulary pickle

`checkpoints/samar_vae.pt` and `samar/samar_vocab.pkl` were both trained with
the legacy flat module layout (where `vocab.py` and `tokenizer.py` lived at
the repo root and the description-tokens bug hadn't been introduced yet).
The current code uses two separate vocabularies -- `SamarVocab` for events
and `DescriptionVocab` for description tokens -- matching the FIGARO paper
(`figaro/src/vocab.py`).

**To regenerate from scratch** (only do this if you've retrained from
scratch and want to commit a fresh vocab):

```python
from samar.tokenizer import SamarTokenizer
SamarTokenizer().save("samar/samar_vocab.pkl")
```

The description vocab is rebuilt from constants on every import, so no
pickle is needed for it.

## Audit

The codebase was audited on 2026-06-25 in two rounds. The first round
fixed 20 mostly-dead-code / drift findings (commit `6d15b00`). The second
round (current) found and fixed 5 critical functional bugs that round 1
missed; details in `docs/audit-2026-06-25.md`.

### Drift vs FIGARO

These are the deliberate and unintentional drifts from the upstream
FIGARO paper (`figaro/src/`):

| # | Decision | FIGARO | SAMAR | Why |
|---|----------|--------|-------|-----|
| 1 | Two-stream split | Description + events, two separate vocabs | Same | Match |
| 2 | Token key names | `'Mean Pitch'` (with space) | `'MeanPitch'` (no space) | Existing pickle uses no-space; would invalidate checkpoint |
| 3 | Per-bar stats in **description** stream | Yes (`get_description`) | Yes (round-2 fix) | Match |
| 4 | Per-bar stats in **event** stream | No | Was yes (removed in round-2) | Was a bug — caused 30% `<unk>` |
| 5 | Pitch vocab range | `range(128)` (MIDI 0..127) | `range(24 * 11)` = 264 quarter-tones | 24-EDO doubling; covers MIDI 0..127 |
| 6 | Instrument fallback | `pretty_midi.program_to_instrument_name(p)` | `'Voice'` when `<part-name>` is missing or starts with `Part_` | Matches existing `Instrument_Voice` token; avoids `<unk>` |
| 7 | `KeySignature_*` tokens | Never emitted | Never emitted (round-2 fix) | Match |
| 8 | `Description_Composer_*` / `Description_Lyricist_*` | Never emitted | Never emitted | Free-text, FIGARO never encodes them |
| 9 | Per-note velocity | From MIDI velocity | Hardcoded 64 | MusicXML doesn't carry per-note velocity; matching FIGARO would require switching to MIDI input |
| 10 | Velocity source | Per-note MIDI velocity | Hardcoded 64 | MusicXML doesn't carry per-note velocity |
| 11 | Description-conditional generation | Available via `description_flavor` | `sample()` accepts `description=` kwarg (not wired in `generating.py`) | Requires retraining with description conditioning |
| 12 | Pitch key name | `'Pitch'` | `'Pitch_24EDO'` | Intentional — the project's core purpose |

### Known limitations

* **Checkpoint `checkpoints/samar_transformer.pt` predates the round-2
  fixes.** It was trained on a corpus where ~25% of tokens were
  `<unk>`. `from_pretrained()` warm-starts the missing
  `description_embedding` / `pos_embedding` layers, but the underlying
  weights are stale. **Retrain to get usable output.**
* **`latents/latents.pt` was precomputed against the pre-round-2
  tokenization.** Recompute via `python -m samar.precompute_samar_latents`
  after retraining.
* **Description-conditional generation is plumbed but not wired.**
  `SamarTransformer.sample()` accepts a `description` kwarg, but
  `generating.py` doesn't pass one. To enable, pick a description
  template and add it to the `sample()` call.
