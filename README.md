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

1. **Parse** — `MusicXMLParser` reads MusicXML files (24-EDO alters supported)
   into `SamarNote` objects.
2. **Tokenize** — `SAMARInputRepresentation` converts notes + metadata
   (key, tempo, time-sig, instrument) into a REMI+ event sequence.
3. **Encode** — `SamarTokenizer` maps events → token IDs.
4. **Train** — `train_samar_vae.py` learns discrete latents over description
   tokens; `train_samar_transformer.py` learns an autoregressive decoder over
   (latent, tokens).
5. **Generate** — `generating.py` samples → decodes → reconstructs MusicXML.
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

The codebase was audited on 2026-06-25. Findings and the rationale for
each design choice live in `docs/audit-2026-06-25.md`. Notable drift vs.
FIGARO:

* Token key names use no spaces (``MeanPitch`` instead of FIGARO's ``Mean
  Pitch``). The existing ``samar_vocab.pkl`` was built with the no-space
  variant; changing to FIGARO's spacing would invalidate the checkpoint.
* Description-tokens never include ``Description_Composer_*`` /
  ``Description_Lyricist_*``. Those are free-text metadata extracted by
  ``core.extract_metadata`` for human inspection but excluded from the
  description token stream because FIGARO never encodes them.
