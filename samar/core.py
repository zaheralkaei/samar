# -*- coding: utf-8 -*-
"""
SAMAR canonical MusicXML input representation.

This module is the single source of truth for MusicXML -> REMI+ conversion
(``SamarNote``, ``MusicXMLParser``, ``extract_metadata``, ``Event``,
``SAMARInputRepresentation``). Older codebases had two parallel
implementations (one in :mod:`samar.parser` and one in
:mod:`samar.input_representation`); they now re-export from here.

See ``docs/audit-2026-06-25.md`` finding #1 for the rationale.

The two-stream design (description + events) follows the FIGARO paper's
``InputRepresentation.get_description()`` + ``get_event_seq()`` split
(``figaro/src/input_representation.py:409-507``). One description stream per
bar carries ``Bar``, ``TimeSignature``, ``MeanPitch``, ``MeanVelocity``,
``MeanDuration``, ``NoteDensity``; the event stream carries per-note
``Position``, ``Pitch_24EDO``, ``Velocity``, ``Duration``, ``Instrument``.
This is the only correct split -- earlier SAMAR code put the per-bar
statistics in the event stream, where ``SamarVocab`` has no slots for them,
producing ~30% ``<unk>`` on real training data. See the 2026-06-25 round-2
audit (findings A1/A2/A5) for the full rationale.
"""

import xml.etree.ElementTree as ET
import zipfile
import numpy as np
from .constants import (
    BAR_KEY,
    TIME_SIGNATURE_KEY,
    POSITION_KEY,
    PITCH_KEY,
    DURATION_KEY,
    TEMPO_KEY,
    VELOCITY_KEY,
    INSTRUMENT_KEY,
    NOTE_DENSITY_KEY,
    MEAN_PITCH_KEY,
    MEAN_VELOCITY_KEY,
    MEAN_DURATION_KEY,
    DEFAULT_POS_PER_QUARTER,
    DEFAULT_DURATION_BINS,
    DEFAULT_TEMPO_BINS,
    DEFAULT_NOTE_DENSITY_BINS,
    DEFAULT_MEAN_PITCH_BINS,
    DEFAULT_MEAN_VELOCITY_BINS,
    DEFAULT_MEAN_DURATION_BINS,
    DEFAULT_VELOCITY_BINS,
    DEFAULT_RESOLUTION,
    DEFAULT_INSTRUMENT,  # fallback when <part-name> is missing
    # Round-20: structural music token keys
    TIE_KEY,
    DOT_KEY,
    TUPLET_KEY,
    CHORD_KEY,
)


def _parse_xml_root(path):
    """Return an ElementTree root for either ``.xml`` or ``.mxl`` input.

    Plain ``.xml`` files are parsed directly. Compressed ``.mxl``
    files (MusicXML 4.0 standard zip container) are opened with
    :mod:`zipfile`; the score is loaded from the file referenced
    by ``META-INF/container.xml``'s ``<rootfile full-path=...>``
    element (typically ``score.xml`` at the archive root).

    This is the single chokepoint that handles both formats; every
    downstream caller (``MusicXMLParser``, ``extract_metadata``)
    goes through it.
    """
    if not path.lower().endswith(".mxl"):
        return ET.parse(path).getroot()

    with zipfile.ZipFile(path) as zf:
        try:
            container = zf.read("META-INF/container.xml").decode("utf-8")
        except KeyError:
            # No container.xml -- fall back to the first ``.xml`` member.
            xml_members = [n for n in zf.namelist()
                           if n.lower().endswith(".xml")
                           and not n.startswith("META-INF/")]
            if not xml_members:
                raise ValueError(f"{path}: no META-INF/container.xml and no .xml members")
            return ET.fromstring(zf.read(xml_members[0]))

        import re as _re
        match = _re.search(r'full-path="([^"]+)"', container)
        if not match:
            raise ValueError(f"{path}: malformed META-INF/container.xml")
        score_path = match.group(1)
        return ET.fromstring(zf.read(score_path))


class SamarNote:
    """A single note (or rest) in the SAMAR format. Supports 24-EDO alters.

    Round-20: extended with structural attributes (tie, dots, tuplet,
    chord) that were previously dropped by the parser. The tokenizer
    reads these from MusicXML <tie/>, <dot/>, <time-modification/>,
    <chord/> elements and emits corresponding tokens.
    """

    def __init__(self, start_tick, step, alter, octave, duration, instrument,
                 velocity=64, is_rest=False,
                 is_tied_start=False, is_tied_stop=False,
                 dot_count=0, tuplet_num=1, is_chord_member=False):
        self.start_tick = int(start_tick)
        self.step = step
        self.alter = float(alter) if alter is not None else 0.0
        self.octave = int(octave)
        self.duration = int(duration)
        self.instrument = instrument
        self.velocity = int(velocity)
        self.is_rest = is_rest
        # Round-20: structural attributes
        self.is_tied_start = bool(is_tied_start)  # <tie type="start"/>
        self.is_tied_stop = bool(is_tied_stop)    # <tie type="stop"/>
        self.dot_count = int(dot_count) if dot_count is not None else 0  # 0/1/2
        self.tuplet_num = int(tuplet_num) if tuplet_num else 1  # 3/5/7 (normal-notes=2)
        self.is_chord_member = bool(is_chord_member)  # <chord/> present

    def to_24edo_pitch(self):
        """Convert to 24-EDO pitch space (MIDI pitch * 2).

        Returns ``None`` for rests. Quarter-tone alters are preserved
        via the fractional ``alter`` field; e.g. ``alter=-0.5`` lowers the
        note by a quarter-tone before doubling to 24-EDO.
        """
        if self.is_rest:
            return None
        step_map = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        base_pitch = step_map[self.step] + self.alter
        midi_pitch = 12 * (self.octave + 1) + base_pitch
        return int(round(midi_pitch * 2))


class MusicXMLParser:
    """Parse a MusicXML file into ``SamarNote`` objects.

    Honors per-bar time-signature changes (``<time>`` elements inside any
    ``<measure>``) when computing bar lengths, so a piece that switches
    2/4 -> 4/4 mid-score is grouped correctly.
    """

    def __init__(self, path):
        # Supports both ``.xml`` (raw) and ``.mxl`` (MusicXML 4.0
        # compressed zip container) via ``_parse_xml_root``.
        self.tree = ET.ElementTree(_parse_xml_root(path))
        self.root = self.tree.getroot()
        self.time_signatures = self._parse_time_signatures_by_bar()
        self.notes = self.parse_notes()
        self.measures = self._group_notes_by_measure()

    def _parse_time_signatures_by_bar(self):
        """Map ``bar_number -> (beats, beat_type)`` for every bar that defines one.

        Bars without an explicit ``<time>`` element inherit the most recent
        time signature (handled in :meth:`parse_notes`).

        Some MusicXML files have measure ``number`` attributes that are
        editorial markers (e.g. ``X1``, ``X2`` for repeats) rather than
        numeric bar indices. Round-3 audit C4: the bare ``int(...)``
        cast crashed 3 files; we now fall back to the most recent
        numeric bar number.
        """
        bar_time_sigs = {}
        last_numeric_bar = 0
        for part in self.root.findall(".//part"):
            for measure in part.findall("measure"):
                raw_number = measure.attrib.get("number", "1")
                try:
                    bar_number = int(raw_number)
                except ValueError:
                    # Editorial marker (X1, X2, ...) -- inherit the
                    # most recent numeric bar number so the bar stays
                    # tied to its predecessors.
                    bar_number = last_numeric_bar
                else:
                    last_numeric_bar = bar_number
                time_elem = measure.find("attributes/time")
                if time_elem is not None:
                    beats = time_elem.findtext("beats")
                    beat_type = time_elem.findtext("beat-type")
                    if beats and beat_type:
                        bar_time_sigs[bar_number] = (int(beats), int(beat_type))
        return bar_time_sigs

    def parse_notes(self):
        notes = []
        divisions = 1
        first_divisions = self.root.find(".//divisions")
        if first_divisions is not None and first_divisions.text.isdigit():
            divisions = int(first_divisions.text)
        ticks_per_division = DEFAULT_RESOLUTION / divisions

        part_names = {}
        for part in self.root.findall(".//score-part"):
            pid = part.attrib.get("id")
            name = part.findtext("part-name", default=DEFAULT_INSTRUMENT)
            # Many Arabic-Music-Dataset XMLs (MuseScore exports) set
            # ``<part-name>Part_1</part-name>`` as a placeholder instead of
            # a real instrument name. The vocab has ``Instrument_Voice``
            # but no ``Instrument_Part_1`` -- using the placeholder
            # produces ``<unk>`` for every note. Map any name starting
            # with ``Part_`` (case-insensitive) to the default instrument.
            # Audit round-2 finding A4.
            if not name or name.strip().lower().startswith("part_"):
                name = DEFAULT_INSTRUMENT
            part_names[pid] = name

        # Walk parts in score order; tick counter is shared so multi-part
        # scores stay time-aligned.
        for part in self.root.findall(".//part"):
            part_id = part.attrib.get("id")
            instrument = part_names.get(part_id, DEFAULT_INSTRUMENT)
            current_tick = 0
            for measure in part.findall("measure"):
                # Round-3 audit C4: tolerate editorial markers
                # (e.g. ``X1``, ``X2`` for repeats) rather than numeric
                # bar indices. The bare ``int(...)`` cast crashed 3
                # files; we now fall back to the most recent numeric
                # bar number.
                raw_bar = measure.attrib.get("number", "1")
                try:
                    bar_number = int(raw_bar)
                except ValueError:
                    # Editorial marker (X1, X2, ...) -- inherit the
                    # most recent numeric bar number so the bar stays
                    # tied to its predecessors.
                    bar_number = max(self.time_signatures.keys(), default=1)
                # Round-20: cap at MAX_N_BARS-1 to avoid OOV Bar_N tokens
                if bar_number >= 512:
                    bar_number = 511
                # Use this bar's time signature, falling back to 4/4 if
                # the score never declared one (very rare in real XML).
                beats, beat_type = self.time_signatures.get(bar_number, (4, 4))
                ticks_per_bar = int(DEFAULT_RESOLUTION * (4 * beats / beat_type))
                for note in measure.findall("note"):
                    rest = note.find("rest") is not None
                    pitch = note.find("pitch")
                    duration_divs = int(note.findtext("duration", default="1"))
                    tick_duration = duration_divs * ticks_per_division

                    # Round-20: read structural MusicXML attributes.
                    # Each is detected via direct child element (not
                    # findtext) since they have no text content.
                    # <tie type="start"/>, <tie type="stop"/>
                    tie_starts = sum(
                        1 for t in note.findall("tie")
                        if t.get("type") == "start"
                    )
                    tie_stops = sum(
                        1 for t in note.findall("tie")
                        if t.get("type") == "stop"
                    )
                    # <dot/> (one or two children)
                    # Round-20: cap dot_count at 2. The vocab only includes Dot_0/1/2;
                    # a triple-dotted note (Dot_3) is exceedingly rare
                    # (1 occurrence in 50 Arabic files) and would
                    # encode as <unk> in the latents.
                    dot_count = min(len(note.findall("dot")), 2)
                    # <time-modification><actual-notes>3</actual-notes>
                    #                      <normal-notes>2</normal-notes>
                    tm = note.find("time-modification")
                    tuplet_num = 1
                    if tm is not None:
                        actual = tm.findtext("actual-notes")
                        if actual is not None:
                            try:
                                tuplet_num = int(actual)
                            except ValueError:
                                tuplet_num = 1
                    # Round-20: cap tuplet_num at the largest value in
                    # TUPLET_VALUES (12). Anything larger becomes <unk>
                    # in the latents. 9- and 11-tuplets are theoretically
                    # possible but vanishingly rare in our data.
                    if tuplet_num > 12:
                        tuplet_num = 12
                    # <chord/> — note starts at same tick as previous note
                    is_chord_member = note.find("chord") is not None

                    if rest:
                        notes.append(SamarNote(
                            current_tick, "C", 0, 4, tick_duration, instrument,
                            velocity=64, is_rest=True,
                            is_tied_start=(tie_starts > 0),
                            is_tied_stop=(tie_stops > 0),
                            dot_count=dot_count,
                            tuplet_num=tuplet_num,
                            is_chord_member=is_chord_member,
                        ))
                    elif pitch is not None:
                        step = pitch.findtext("step", "C")
                        alter = pitch.findtext("alter", "0")
                        octave = pitch.findtext("octave", "4")
                        notes.append(SamarNote(
                            current_tick, step, alter, octave, tick_duration,
                            instrument, velocity=64,
                            is_tied_start=(tie_starts > 0),
                            is_tied_stop=(tie_stops > 0),
                            dot_count=dot_count,
                            tuplet_num=tuplet_num,
                            is_chord_member=is_chord_member,
                        ))

                    current_tick += tick_duration
        return notes

    def _group_notes_by_measure(self):
        grouped = {}
        for note in self.notes:
            bar_idx = note.start_tick // (DEFAULT_RESOLUTION * 4)
            grouped.setdefault(bar_idx, []).append(note)
        return grouped

    def parse_time_signature(self):
        for ts in self.root.findall(".//time"):
            beats = ts.findtext("beats")
            beat_type = ts.findtext("beat-type")
            if beats and beat_type:
                return int(beats), int(beat_type)
        return 4, 4

    def parse_tempo(self):
        """First ``<sound tempo="...">`` in the score, or None."""
        sound = self.root.find(".//sound[@tempo]")
        if sound is not None:
            tempo = sound.get("tempo")
            if tempo:
                return int(float(tempo))
        metronome = self.root.find(".//metronome/per-minute")
        if metronome is not None:
            return int(float(metronome.text.strip()))
        return None

    def parse_key_signature(self):
        """Number of fifths as a string, or None if missing."""
        for key in self.root.findall(".//key"):
            fifths = key.findtext("fifths")
            if fifths:
                return fifths
        return None


def extract_metadata(xml_path):
    """Pull composer/lyricist credits, time signature, key, tempo out of an XML file.

    The returned dict's keys are split into two groups:
      * ``Description_*`` -- meant for human inspection (composer, lyricist).
        These are NEVER tokenized -- FIGARO never encodes free-text metadata.
      * ``TimeSignature``, ``KeySignature``, ``Tempo``, ``Instruments`` --
        consumed by ``SAMARInputRepresentation._build_remi_events``.
    """
    tree = ET.ElementTree(_parse_xml_root(xml_path))
    root = tree.getroot()
    metadata = {}
    for credit in root.findall("credit"):
        credit_type = credit.find("credit-type")
        credit_words = credit.find("credit-words")
        if credit_type is not None and credit_words is not None:
            key = f"Description_{credit_type.text.capitalize()}"
            metadata[key] = credit_words.text.strip()
    creator = root.find(".//creator[@type='composer']")
    if creator is not None:
        metadata["Description_Composer"] = creator.text.strip()
    for score_part in root.findall(".//score-part"):
        part_name = score_part.find("part-name")
        if part_name is not None:
            metadata.setdefault("Instruments", []).append(part_name.text.strip())
    first_measure = root.find(".//part/measure")
    if first_measure is not None:
        time = first_measure.find("attributes/time")
        if time is not None:
            beats = time.find("beats")
            beat_type = time.find("beat-type")
            if beats is not None and beat_type is not None:
                metadata["TimeSignature"] = f"{beats.text}/{beat_type.text}"
        key = first_measure.find("attributes/key/fifths")
        if key is not None:
            metadata["KeySignature"] = int(key.text)
        sound = first_measure.find(".//sound")
        if sound is not None:
            tempo = sound.attrib.get("tempo")
            if tempo is not None:
                metadata["Tempo"] = int(float(tempo))
    return metadata


class Event:
    """A single symbolic-music event with metadata (kept for backwards compat)."""

    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return f"Event(name={self.name}, time={self.time}, value={self.value}, text={self.text})"


class SAMARInputRepresentation:
    """Convert MusicXML -> a two-stream representation (description + events).

    Mirrors the FIGARO ``InputRepresentation`` design exactly:

      * ``description_tokens`` -- per-bar stream of ``Bar``, ``TimeSignature``,
        ``MeanPitch``, ``MeanVelocity``, ``MeanDuration``, ``NoteDensity``
        tokens. Encoded by :class:`~samar.tokenizer.DescriptionTokenizer`
        against :class:`~samar.vocab.DescriptionVocab`.
      * ``events`` -- per-note stream of ``Position``, ``Pitch_24EDO``,
        ``Velocity``, ``Duration``, ``Instrument`` tokens (plus a single
        global ``Tempo`` token at the start). Encoded by
        :class:`~samar.tokenizer.SamarTokenizer` against
        :class:`~samar.vocab.SamarVocab`.

    See :func:`get_event_sequence` for the order in which they're combined.
    """

    def __init__(self, xml_file):
        self.parser = MusicXMLParser(xml_file)
        self.notes = sorted(self.parser.notes, key=lambda n: n.start_tick)
        self.metadata = extract_metadata(xml_file)
        self.time_sig = self.parser.parse_time_signature()
        self.description_tokens = self._build_description_tokens()
        self.events = self._build_remi_events()

    @staticmethod
    def _safe_mean(values):
        """``np.mean(values)`` or 0 if the input is empty.

        Avoids ``RuntimeWarning: Mean of empty slice`` when a bar contains
        only rests or the file has no pitched notes. Handles both Python
        lists and numpy arrays.
        """
        if len(values) == 0:
            return 0
        return np.mean(values)

    def _bar_index_for_tick(self, tick):
        """Map a MIDI tick to a 1-based bar number for this piece's time signature.

        Round-20: caps the result at MAX_N_BARS - 1 (= 511) to avoid
        out-of-vocabulary Bar_N tokens. Long pieces (>512 bars) are
        extremely rare in our training data; capping is safer than
        growing the vocab by 1000+ tokens for marginal gain.
        """
        beats, beat_type = self.time_sig
        quarters_per_bar = 4 * beats / beat_type
        ticks_per_bar = int(DEFAULT_RESOLUTION * quarters_per_bar)
        if ticks_per_bar <= 0:
            return 1
        bar_num = tick // ticks_per_bar + 1
        if bar_num >= 512:
            bar_num = 511
        return int(bar_num)

    def _notes_by_bar(self):
        """Group notes by 1-based bar number."""
        groups = {}
        for note in self.notes:
            bar_num = self._bar_index_for_tick(note.start_tick)
            groups.setdefault(bar_num, []).append(note)
        return groups

    def _bar_stats(self, bar_notes, positions_per_bar):
        """Quantize per-bar statistics to their bin indices.

        Mirrors the per-bar computation in
        ``figaro/src/input_representation.py:443-498``. Returns a list of
        ``(name, index, raw)`` tuples suitable for emitting as description
        events.
        """
        n_notes = len(bar_notes)
        velocities = np.array([n.velocity for n in bar_notes if not n.is_rest])
        pitches = np.array([n.to_24edo_pitch() for n in bar_notes
                           if not n.is_rest and n.to_24edo_pitch() is not None])
        durations = np.array([n.duration for n in bar_notes])

        # ``_safe_mean`` keeps a rest-only bar from emitting
        # ``RuntimeWarning: Mean of empty slice``. NaN would also work but
        # breaks ``np.argmin`` below.
        mean_velocity = self._safe_mean(velocities) if len(velocities) > 0 else 0
        mean_pitch = self._safe_mean(pitches) if len(pitches) > 0 else 0
        mean_duration = self._safe_mean(durations) if len(durations) > 0 else 0
        note_density = n_notes / positions_per_bar if positions_per_bar > 0 else 0

        v_idx = int(np.argmin(np.abs(DEFAULT_MEAN_VELOCITY_BINS - mean_velocity)))
        p_idx = int(np.argmin(np.abs(DEFAULT_MEAN_PITCH_BINS - mean_pitch)))
        d_idx = int(np.argmin(np.abs(DEFAULT_MEAN_DURATION_BINS - mean_duration)))
        nd_idx = int(np.argmin(np.abs(DEFAULT_NOTE_DENSITY_BINS - note_density)))

        return [
            (NOTE_DENSITY_KEY, nd_idx, note_density),
            (MEAN_VELOCITY_KEY, v_idx, mean_velocity),
            (MEAN_PITCH_KEY, p_idx, mean_pitch),
            (MEAN_DURATION_KEY, d_idx, mean_duration),
        ]

    def _build_description_tokens(self):
        """Per-bar description stream -- one ``Bar`` + stats per bar.

        Matches the FIGARO ``InputRepresentation.get_description()`` design
        (``figaro/src/input_representation.py:409-507``). Each bar emits:

          * ``Bar_N`` -- 1-based bar number
          * ``TimeSignature_N/M`` -- the bar's time signature (only emitted
            when the bar's time sig differs from the previous bar's, to
            avoid redundant tokens; the first bar always emits it)
          * ``NoteDensity_BIN`` -- quantized note-density bin index
          * ``MeanVelocity_BIN`` -- quantized mean-velocity bin index
          * ``MeanPitch_BIN`` -- quantized mean-pitch bin index (in 24-EDO
            space, since we doubled MIDI for quarter-tones)
          * ``MeanDuration_BIN`` -- quantized mean-duration bin index

        Description_Composer_*, KeySignature_*, Tempo_* are deliberately
        excluded -- FIGARO never encodes free-text or unstructured
        metadata in the description stream. Tempo goes in the event
        stream instead.
        """
        tokens = []
        if not self.notes:
            return tokens

        beats, beat_type = self.time_sig
        quarters_per_bar = 4 * beats / beat_type
        ticks_per_bar = int(DEFAULT_RESOLUTION * quarters_per_bar)
        positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)
        if positions_per_bar <= 0:
            positions_per_bar = 1

        notes_by_bar = self._notes_by_bar()
        prev_time_sig = None
        for bar_num in sorted(notes_by_bar.keys()):
            bar_notes = notes_by_bar[bar_num]
            tokens.append(f"{BAR_KEY}_{bar_num}")

            # Emit TimeSignature only when it changes (or on the first bar).
            current_time_sig = (beats, beat_type)
            if current_time_sig != prev_time_sig:
                tokens.append(f"{TIME_SIGNATURE_KEY}_{beats}/{beat_type}")
                prev_time_sig = current_time_sig

            for name, idx, _raw in self._bar_stats(bar_notes, positions_per_bar):
                tokens.append(f"{name}_{idx}")

        return tokens

    def get_description_tokens(self):
        return self.description_tokens

    def get_event_sequence(self):
        """Combined token sequence: per-bar description stream + per-note event stream.

        This is what :meth:`SamarTokenizer.encode` consumes for training
        and what :meth:`SamarTransformer.sample` generates at inference.
        """
        return self.description_tokens + self.events

    def _build_remi_events(self):
        """Per-note event stream.

        Emits, in this order:

          * ``Tempo_BIN`` at the start (if ``<sound tempo>`` was found)
          * Per bar: ``Position``, ``Pitch_24EDO``, ``Velocity``, ``Duration``,
            ``Instrument`` per note.

        KeySignature_* is NOT emitted (matches FIGARO -- KeySignature is in
        the constants but never encoded). Per-bar statistics
        (``MeanPitch``, ``NoteDensity``, ...) belong to the description
        stream, NOT this one -- emitting them here previously caused
        ~30% ``<unk>`` on real training data (audit round-2 finding A2).
        """
        events = []
        if not self.notes:
            return events

        beats, beat_type = self.time_sig
        quarters_per_bar = 4 * beats / beat_type
        ticks_per_bar = int(DEFAULT_RESOLUTION * quarters_per_bar)
        positions_per_bar = int(DEFAULT_POS_PER_QUARTER * quarters_per_bar)

        # Optional global Tempo token (matches FIGARO event stream).
        tempo = self.metadata.get("Tempo")
        if tempo is not None:
            tempo_idx = int(np.argmin(np.abs(DEFAULT_TEMPO_BINS - int(tempo))))
            events.append(f"{TEMPO_KEY}_{tempo_idx}")

        notes_by_bar = self._notes_by_bar()
        # Round-20: tracks when a previous note is tied to the next note.
        # Round-20: tracks when a previous note is tied to the next note.
        # After emitting a note with is_tied_start, the next emitted note
        # (regardless of bar) needs to receive a Tie_Start BEFORE its Pitch
        # so the reconstructor can add the <tie type="stop"/> element to
        # it. We carry this across bar boundaries.
        prev_was_starting_tie = False
        for bar_num in sorted(notes_by_bar.keys()):
            # Round-7: emit explicit Bar_N token at the start of each
            # bar (matches FIGARO's input_representation.get_remi_events
            # at figaro/src/input_representation.py:297-302). Without
            # this, SAMAR's event stream has NO bar-boundary markers
            # and the model never learns to generate them -- which is
            # why round-6 examples jammed all 256 tokens into measure 0
            # and MuseScore rendered empty bars. The round-7
            # reconstructor renumbers Bar_0 -> measure 1 etc. so
            # MuseScore displays 1-indexed measure numbers.
            events.append(f"{BAR_KEY}_{bar_num}")
            for note in notes_by_bar[bar_num]:
                # Chord members don't get their own Position token --
                # they share the previous note's start tick. We emit a
                # Chord_On marker instead.
                if note.is_chord_member:
                    events.append(f"{CHORD_KEY}_On")
                else:
                    rel_tick = note.start_tick % ticks_per_bar
                    position = int(rel_tick / ticks_per_bar * positions_per_bar)
                    events.append(f"{POSITION_KEY}_{position}")

                # Round-20: Tuplet_N BEFORE the note it applies to.
                # Only emitted when this note is in a tuplet group
                # (tuplet_num > 1). The reconstructor emits
                # <time-modification> on the matching note.
                if note.tuplet_num > 1:
                    events.append(f"{TUPLET_KEY}_{note.tuplet_num}")

                # Round-20: Dot_Y BEFORE the note it applies to.
                # Captures <dot/> elements that were previously dropped.
                if note.dot_count > 0:
                    events.append(f"{DOT_KEY}_{note.dot_count}")

                # Round-20: Tie_Start BEFORE Pitch -- signals "this
                # note is the START of a tied run; the NEXT note
                # should get <tie type='stop'/>." Set by the
                # prev_was_starting_tie flag from the previous note.
                if prev_was_starting_tie:
                    events.append(f"{TIE_KEY}_Start")
                    prev_was_starting_tie = False

                if note.is_rest:
                    events.append(f"{PITCH_KEY}_Rest")
                    events.append(f"{VELOCITY_KEY}_0")
                else:
                    pitch_val = note.to_24edo_pitch()
                    events.append(f"{PITCH_KEY}_{pitch_val}")
                    if note.velocity is not None:
                        vel_idx = int(np.argmin(np.abs(DEFAULT_VELOCITY_BINS - note.velocity)))
                        events.append(f"{VELOCITY_KEY}_{vel_idx}")

                duration_pos = int(note.duration / DEFAULT_RESOLUTION * DEFAULT_POS_PER_QUARTER)
                duration_idx = int(np.argmin(np.abs(DEFAULT_DURATION_BINS - duration_pos)))
                events.append(f"{DURATION_KEY}_{duration_idx}")

                if note.instrument:
                    events.append(f"{INSTRUMENT_KEY}_{note.instrument}")

                # Round-20: Tie_Stop AFTER Instrument -- signals "this
                # note is the END of a tied run; should get
                # <tie type='stop'/> in the output XML."
                if note.is_tied_stop:
                    events.append(f"{TIE_KEY}_Stop")

                # Track for next iteration: if THIS note is a tie start,
                # the NEXT note should emit Tie_Start.
                if note.is_tied_start:
                    prev_was_starting_tie = True

        return events
