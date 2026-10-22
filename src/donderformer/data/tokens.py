"""Event tokens for the Transformer: a chart as (time, note) pairs on the 23 ms frame grid.

The shared note format is a ``(n_frames,)`` int array of class ids (``tja.CLASS_NAMES``):
0 none, 1 don, 2 ka, 3 DON, 4 KA, 5 drumroll, 6 balloon, with long notes filling every
frame they cover. Events are the same chart without the empty frames:

    (frame, "don")                    one per onset frame (classes 1-4)
    (start, "ROLL"), (end, "END")     one pair per run of 5s (end = last frame of the run)
    (start, "BALLOON"), (end, "END")  one pair per run of 6s

so ``events_to_classes(classes_to_events(c), len(c))`` gives back ``c`` exactly.

One decoder sequence covers one window of ``n_time`` frames starting at frame ``s``::

    BOS <course> <level>  TIME_k NOTE  ...  SEP  TIME_k NOTE  ...  EOS
                          (past region)          (target region)

``TIME_k`` is the frame ``s + k``. A long note already open at ``s`` is written as starting
at ``TIME_0``; one still open at the end of the window simply has no ``END`` yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from donderformer.data.tja import COURSES

SPECIALS = ("PAD", "BOS", "SEP", "EOS")
LEVELS = tuple(range(1, 11))
NOTES = ("don", "ka", "DON", "KA", "ROLL", "BALLOON", "END")
ONSET_NOTES = NOTES[:4]
LONG_NOTES = ("ROLL", "BALLOON")

# Shared-format class id <-> note name
CLASS_TO_NOTE = {1: "don", 2: "ka", 3: "DON", 4: "KA", 5: "ROLL", 6: "BALLOON"}
NOTE_TO_CLASS = {note: cls for cls, note in CLASS_TO_NOTE.items()}


@dataclass(frozen=True)
class Vocab:
    """Token ids: specials, courses, levels (+ unknown), notes, then one time token per frame."""

    n_time: int = 768

    PAD = 0
    BOS = 1
    SEP = 2
    EOS = 3
    COURSE_BASE = len(SPECIALS)
    LEVEL_BASE = COURSE_BASE + len(COURSES)
    LEVEL_UNK = LEVEL_BASE + len(LEVELS)
    NOTE_BASE = LEVEL_UNK + 1
    TIME_BASE = NOTE_BASE + len(NOTES)

    @property
    def size(self) -> int:
        return self.TIME_BASE + self.n_time

    def course(self, course: str) -> int:
        return self.COURSE_BASE + COURSES.index(course)

    def level(self, level) -> int:
        """Level token; ``None``, ``""`` or out-of-range levels map to ``LEVEL_UNK``."""
        try:
            level = int(float(level))
        except (TypeError, ValueError):
            return self.LEVEL_UNK
        return self.LEVEL_BASE + level - 1 if level in LEVELS else self.LEVEL_UNK

    def note(self, name: str) -> int:
        return self.NOTE_BASE + NOTES.index(name)

    def time(self, k: int) -> int:
        if not 0 <= k < self.n_time:
            raise ValueError(f"time {k} outside 0..{self.n_time - 1}")
        return self.TIME_BASE + k

    def is_time(self, token) -> bool:
        return token >= self.TIME_BASE

    def is_note(self, token) -> bool:
        return self.NOTE_BASE <= token < self.TIME_BASE

    def name(self, token: int) -> str:
        """Readable name of one token, for printing sequences."""
        if token < self.COURSE_BASE:
            return SPECIALS[token]
        if token < self.LEVEL_BASE:
            return COURSES[token - self.COURSE_BASE].upper()
        if token < self.LEVEL_UNK:
            return f"LEVEL_{token - self.LEVEL_BASE + 1}"
        if token == self.LEVEL_UNK:
            return "LEVEL_UNK"
        if token < self.TIME_BASE:
            return NOTES[token - self.NOTE_BASE]
        return f"TIME_{token - self.TIME_BASE}"

    def decode(self, tokens) -> str:
        return " ".join(self.name(int(t)) for t in tokens)


def classes_to_events(classes) -> list[tuple[int, str]]:
    """Shared-format classes -> sorted ``(frame, note)`` events."""
    classes = np.asarray(classes)
    events = []
    frame, n = 0, len(classes)
    while frame < n:
        cls = int(classes[frame])
        if cls in (1, 2, 3, 4):
            events.append((frame, CLASS_TO_NOTE[cls]))
            frame += 1
        elif cls in (5, 6):
            end = frame
            while end + 1 < n and classes[end + 1] == cls:
                end += 1
            events.append((frame, CLASS_TO_NOTE[cls]))
            events.append((end, "END"))
            frame = end + 1
        else:
            frame += 1
    return events


def events_to_classes(events, n_frames: int) -> np.ndarray:
    """Sorted ``(frame, note)`` events -> shared-format classes. An unclosed long note runs to the end."""
    classes = np.zeros(n_frames, dtype=np.int64)
    open_note = None
    for frame, note in events:
        if note == "END":
            if open_note is not None:
                start, cls = open_note
                classes[max(start, 0):min(frame, n_frames - 1) + 1] = cls
                open_note = None
        elif note in LONG_NOTES:
            open_note = (frame, NOTE_TO_CLASS[note])
        elif 0 <= frame < n_frames:
            classes[frame] = NOTE_TO_CLASS[note]
    if open_note is not None:
        start, cls = open_note
        classes[max(start, 0):] = cls
    return classes


def window_events(events, start: int, stop: int) -> list[tuple[int, str]]:
    """Events with ``start <= frame < stop``, shifted to the window (frame - start).

    A long note that began before ``start`` and ends at or after it is kept, starting at 0.
    """
    out = []
    open_note = None    # (frame, note) of a long note whose END we haven't seen yet
    for frame, note in events:
        if frame >= stop:
            if note == "END" and open_note is not None and open_note[0] < start:
                out.insert(0, (0, open_note[1]))      # spans the whole window
            break
        if note in LONG_NOTES:
            open_note = (frame, note)
        if frame < start:
            if note == "END":
                open_note = None
            continue
        if note == "END" and open_note is not None and open_note[0] < start:
            out.append((0, open_note[1]))
        if note == "END":
            open_note = None
        out.append((frame - start, note))
    else:
        if open_note is not None and open_note[0] < start:
            out.insert(0, (0, open_note[1]))
    out.sort(key=lambda e: (e[0], e[1] == "END"))
    return out


def events_to_tokens(vocab: Vocab, events) -> list[int]:
    """Window-relative events -> ``TIME_k NOTE`` pairs."""
    tokens = []
    for frame, note in events:
        tokens += [vocab.time(frame), vocab.note(note)]
    return tokens


def tokens_to_events(vocab: Vocab, tokens) -> list[tuple[int, str]]:
    """``TIME_k NOTE`` pairs -> window-relative events; anything else is skipped."""
    events, time = [], None
    for token in tokens:
        token = int(token)
        if vocab.is_time(token):
            time = token - vocab.TIME_BASE
        elif vocab.is_note(token) and time is not None:
            events.append((time, NOTES[token - vocab.NOTE_BASE]))
            time = None
    return events


def build_sequence(vocab: Vocab, course: str, level, past_events, target_events=None,
                   max_tokens: int | None = None) -> tuple[list[int], int]:
    """Decoder tokens for one window and the index of ``SEP``.

    With ``target_events=None`` the sequence ends at ``SEP`` (a generation prompt). When
    ``max_tokens`` is set, the oldest past events are dropped until the sequence fits.
    """
    head = [vocab.BOS, vocab.course(course), vocab.level(level)]
    past = events_to_tokens(vocab, past_events)
    tail = [vocab.SEP]
    if target_events is not None:
        tail += events_to_tokens(vocab, target_events) + [vocab.EOS]
    if max_tokens is not None:
        overflow = len(head) + len(past) + len(tail) - max_tokens
        if overflow > 0:
            past = past[overflow + overflow % 2:]       # whole (TIME, NOTE) pairs
        if len(head) + len(past) + len(tail) > max_tokens:
            raise ValueError(f"target alone needs {len(head) + len(tail)} tokens > max_tokens={max_tokens}")
    tokens = head + past + tail
    return tokens, len(head) + len(past)
