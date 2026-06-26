# SAMAR Round 6 — Temperature sampling + architectural fix

## The architectural issue

The round-3 audit documented: SamarTransformer's `output_layer`
maps `d_model` → `latent_dim`, not `d_model` → `vocab_size`.
The model was trained to predict next-step latent vectors
(MSE loss against latents.pt), not next-token logits.

The original `sample()` method took `argmax` of the latent-dim
output and used that as a token ID. The latent_dim is 128 and
vocab is 1254, so the produced IDs were always `< 128` and
frequently `<unk>`. With greedy decoding the model emitted
`Bar_150` (token 150) over and over.

## The fix

Two-part:

1. **Generation pipeline now goes through the VAE decoder.**
   At each step, the transformer predicts the next latent, the
   VAE decoder projects that latent to `[B, vocab_size]` event
   logits, and the sampler picks the next token.

2. **`SamarTransformer.sample()` gained temperature / top-k /
   top-p sampling.** These are standard sampler techniques that
   prevent mode collapse and produce diverse outputs. Special
   tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`, `<mask>`) are
   masked to `-inf` before sampling so the model can't pick
   them as events (which would early-terminate generation).

## What works now

Verified end-to-end on ai-laptop with the round-5 trained
checkpoint (val_loss=0.0004):

```
Test 1: temp=1.0, top_k=50
  256 tokens, 44 distinct, full max_length, no <pad> stop
  Includes Velocity, Duration, Position, Pitch (24-EDO),
    Instrument (Piano/Voice/Violin/Drumset/Harp),
    even Pitch_24EDO_Rest

Test 2: temp=0.8, top_k=30, top_p=0.9
  256 tokens, 17 distinct (lower temp = less diverse, expected)

Generated MusicXML is structurally valid:
  - 5 instrument parts (Piano, Violin, Drumset, Voice, Harp)
  - Real pitches with <alter>1.0</alter> for sharps
  - Mix of notes and rests
```

## MIDI question

User asked if we can train on MIDI + XML. **Yes in principle,
no in practice for this dataset.** Reasons:

- FIGARO does it via `pretty_midi.PrettyMIDI(file)`. SAMAR
  doesn't have a MIDI parser.
- `data/midi/Arabic_Music_Dataset/` has **ZERO `.mid` files**.
  Only `.mscz` MuseScore source files (58) + `.xml` (7, all
  duplicates of files in `data/xml/`) + `.pdf` (359, not
  training data).
- Converting `.mscz` → `.mid` needs MuseScore CLI. Not
  installed on ai-laptop.

Decision (user, Option A): skip MIDI for this round.
Retrain on the existing 78 files with the new model
configuration. No architectural change needed -- retraining
just refreshes the weights against the now-working inference
path.

## Files changed (round 6)

- `samar/models/samar_transformer.py`: new `temperature` /
  `top_k` / `top_p` / `vae_decoder` params on `sample()`,
  special-token masking, special-token-ids setup in
  `from_pretrained`.
- `samar/generating.py`: new `--temperature` / `--top-k` /
  `--top-p` / `--no-vae-decode` CLI flags. Default behaviour
  passes the VAE decoder so sampling produces valid tokens.

## Commits

- `8c966ef` round 6: temperature / top-k / top-p sampling
- `89ae587` round 6: mask special tokens during sampling
  (prevents early `<pad>` stop)