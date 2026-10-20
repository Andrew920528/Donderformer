"""Turn a per-frame class sequence (what the models generate) back into a playable TJA chart.

Classes follow ``tja.CLASS_NAMES``: 0 none, 1 don, 2 ka, 3 DON, 4 KA, 5 drumroll, 6 denden.
A run of 5s becomes a drumroll ``5 … 8`` and a run of 6s a balloon ``7 … 8``.

Frames are placed on a constant-tempo grid. With the song's real ``bpm``/``offset`` (manifest
columns of the same name) the notes land on its beats; for a new song any tempo works, the grid
is just fine enough (at most half a frame per slot) that every frame keeps its own slot, so
``parse_events`` + ``events_to_labels`` on the result gives back the same frames.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ONSET_CODES = {1: "1", 2: "2", 3: "3", 4: "4"}
LONG_CODES = {5: "5", 6: "7"}
BASE_SLOTS = 48            # per 4/4 measure; divisible by 3 and 16 so common rhythms stay exact


def chart_to_tja(classes, frame_ms: float = 23, bpm: float = 120.0, offset: float = 0.0,
                 title: str = "Generated chart", wave: str = "", course: str = "Oni",
                 level: int | None = None, balloon_hits_per_second: float = 10.0) -> str:
    """TJA text for one chart.

    ``classes``: ``(n_frames,)`` ints, or ``(n_frames, 7)`` one-hot/probabilities (argmax is taken).
    ``offset``: TJA ``OFFSET``; the first measure starts at ``-offset`` seconds. If notes come
    earlier, whole measures are added in front and ``OFFSET`` is adjusted to match.
    """
    classes = np.asarray(classes)
    if classes.ndim == 2:
        classes = classes.argmax(axis=1)
    classes = classes.astype(np.int64)

    measure_s = 4 * 60.0 / bpm
    slots = BASE_SLOTS * math.ceil(measure_s / (BASE_SLOTS * frame_ms / 2000))
    slot_s = measure_s / slots

    # (frame, code) for every symbol: onsets, long-note starts and the 8 that closes them.
    symbols = []
    frame, n = 0, len(classes)
    while frame < n:
        cls = int(classes[frame])
        if cls in ONSET_CODES:
            symbols.append((frame, ONSET_CODES[cls], None))
            frame += 1
        elif cls in LONG_CODES:
            end = frame
            while end + 1 < n and classes[end + 1] == cls:
                end += 1
            symbols.append((frame, LONG_CODES[cls], end))
            frame = end + 1
        else:
            frame += 1
    if not symbols:
        raise ValueError("no notes to write")

    def slot_of(f):            # global slot index of the centre of frame f
        return round(((f + 0.5) * frame_ms / 1000 + offset) / slot_s)

    first = slot_of(symbols[0][0])
    if first < 0:              # notes before the first measure: prepend measures
        extra = math.ceil(-first / slots)
        offset += extra * measure_s

    grid, balloons = {}, []
    for start, code, end in symbols:
        s = slot_of(start)
        grid.setdefault(s, code)
        if end is not None:
            e = max(slot_of(end), s + 1)
            grid.setdefault(e, "8")
            if code == "7":
                duration = (end - start + 1) * frame_ms / 1000
                balloons.append(max(1, round(duration * balloon_hits_per_second)))

    measures = []
    for m in range(max(grid) // slots + 1):
        row = [grid.get(m * slots + i, "0") for i in range(slots)]
        step = math.gcd(slots, *[i for i, c in enumerate(row) if c != "0"]) if any(c != "0" for c in row) else slots
        measures.append("".join(row[::step]) + ",")

    header = [f"TITLE:{title}", f"BPM:{bpm:g}", f"WAVE:{wave}", f"OFFSET:{offset:.6f}", "",
              f"COURSE:{course}"]
    if level is not None:
        header.append(f"LEVEL:{level}")
    if balloons:
        header.append("BALLOON:" + ",".join(map(str, balloons)))
    return "\n".join(header + ["", "#START", *measures, "#END", ""])


def write_tja(path, classes, **kwargs) -> Path:
    """``chart_to_tja`` written to ``path`` (UTF-8). Put the audio named in ``wave`` next to it to play it."""
    path = Path(path)
    path.write_text(chart_to_tja(classes, **kwargs), encoding="utf-8")
    return path
