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
