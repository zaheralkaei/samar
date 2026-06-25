# SAMAR Audit — 2026-06-25 Round 4

Round 4 was triggered by a dataset expansion: 52 `.xml` files became
78 (52 `.xml` + 26 `.mxl`). The new files revealed:

1. `.mxl` is the standard MusicXML 4.0 compressed format (a zip
   archive containing `META-INF/container.xml` and `score.xml`).
   The parser was only reading `.xml` directly.
2. The new MuseScore exports used 5 instruments that weren't in
   the vocab: `Harp`, `Drumset`, `Acoustic Guitar`, `Classical
   Guitar`, and `Classical Guitar (Tablature)`.
3. The default instrument list was missing a `classical` guitar
   variant that MuseScore tags as "Classical Guitar (Tablature)".

## Critical findings (fixed)

### C1 `.mxl` files were unparseable

**WHAT**: The 26 new `.mxl` files all started with `PK\x03\x04`
(the ZIP magic number). They contain `META-INF/container.xml`
pointing to the actual score via `<rootfile full-path="score.xml">`.
The parser called `ET.parse(path)` directly, which expects raw
XML, so every `.mxl` file failed.

**FIX**: Added `samar/core.py:_parse_xml_root(path)` — a single
chokepoint that handles both formats:

```python
if not path.lower().endswith(".mxl"):
    return ET.parse(path).getroot()

with zipfile.ZipFile(path) as zf:
    container = zf.read("META-INF/container.xml").decode("utf-8")
    match = re.search(r'full-path="([^"]+)"', container)
    return ET.fromstring(zf.read(match.group(1)))
```

Both `MusicXMLParser.__init__` and `extract_metadata` go through
this helper. The `describe.py` CLI's `_iter_xml_files` was also
extended to walk `.mxl` files.

**VERIFIED**: 78/78 files parse (was 52/52).

### C2 5 instruments missing from the vocab

**WHAT**: The default `Tokens.get_instrument_tokens` had 12 hardcoded
names (Violin, Nay, Oud, etc.) but no `Harp`, `Drumset`, or any
Guitar variant. The 26 new `.mxl` files emitted
`Instrument_Harp` (635 occurrences), `Instrument_Drumset` (525),
`Instrument_Acoustic Guitar` (297), `Instrument_Classical Guitar`
(83), `Instrument_Classical Guitar (Tablature)` (85). Total:
**1625 notes tagged as instruments we couldn't encode** (4.2% of
all new-file event-stream tokens).

**FIX**: Extended `Tokens.get_instrument_tokens` to include the 5
missing names. Vocab grew from 1249 -> 1254 tokens. Pickle
regenerated (backup at `samar/samar_vocab_v1249_backup.pkl`).
`DEFAULT_VOCAB_SIZE` in `train_samar_transformer.py` updated to
1254. `samar_transformer_config.json` updated to 1254.

**VERIFIED**:
  - Live vocab: 1254
  - Pickle: 1254
  - JSON config: 1254
  - All 78 files: 0.00% `<unk>` in event stream, 0.00% in description

## What's still good (carried from rounds 1-3)

- All 5 smoke tests pass
- Per-bar time signatures parse correctly
- X1 measure numbers handled (round 3)
- Description <unk> ratio: 0.00% across all 78 files
- Two-vocab split (FIGARO pattern) intact
- `from_pretrained` handles vocab-size extension

## New corpus distribution (78 files)

### Singer distribution
```
fairuz            35 files (45%)
abdel              6 files ( 8%)
asmahan            5 files ( 6%)
sheikh             4 files ( 5%)
Samaii             2 files ( 3%)
mesh               2 files ( 3%)
Kifak, alamouni,   1 file each (~25+ more diverse singers)
...
```
Total: 30 distinct singer names (was 4 before round 4).

### Maqam distribution
```
bayat             16 files (21%)
nahawand          11 files (14%)
kurd               8 files (10%)
ajam               6 files ( 8%)
rast               4 files ( 5%)
huzam              3 files ( 4%)
hijaz              3 files ( 4%)
saba               1 file  ( 1%)
```

### File format
```
.xml: 52 files
.mxl: 26 files  (compressed MusicXML 4.0)
```

### Instrument distribution (top 5)
```
Voice                  18405 occurrences
Piano                  10978
Violin                  1549
Harp                     635  (NEW in round 4)
Drumset                  525  (NEW in round 4)
Acoustic Guitar          297  (NEW in round 4)
Classical Guitar          83  (NEW in round 4)
Classical Guitar (Tab)    85  (NEW in round 4)
```

## Future work (round 5+)

1. **Stale checkpoint needs retraining.** `checkpoints/samar_transformer.pt`
   was trained on the pre-round-2 vocab (1129). After round 2 (1249),
   round 4 (1254), the checkpoint now lags by 125 tokens. `from_pretrained`
   handles the extension via zero-padding, but the weights for new tokens
   are random. **Retrain to get usable output.**

2. **Recompute `latents/latents.pt`** — bundled latents are from
   pre-round-2 tokenization (~24% `<unk>`). After retraining, run
   `python -m samar.precompute_samar_latents` to get clean latents.

3. **Stale pickle backup chain**:
   - `samar_vocab_v1129_backup.pkl` (round 2)
   - `samar_vocab_v1249_backup.pkl` (round 4)
   - `samar_vocab.pkl` (current, 1254 tokens)

4. **Stratified sampling** — Bayat (16 files, 21%) and Fairuz
   (35 files, 45%) still dominate. Either oversample rare
   maqamat or document the bias.

5. **Description-conditional generation** — still plumbed but
   not wired through `generating.py`. (Round 3 audit item,
   deferred.)