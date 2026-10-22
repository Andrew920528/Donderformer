"""Turn a per-frame class sequence (what the models generate) back into a playable TJA chart.

Classes follow ``tja.CLASS_NAMES``: 0 none, 1 don, 2 ka, 3 DON, 4 KA, 5 drumroll, 6 denden.
A run of 5s becomes a drumroll ``5 … 8`` and a run of 6s a balloon ``7 … 8``.

Each note is snapped to the simplest musical subdivision of the measure (1/4, 1/8, 1/16, 1/12, …)
that falls inside its 23 ms frame, on a constant tempo given by ``bpm``/``offset``. With the
song's real values (manifest columns of the same name) a human chart's notes go back to their
exact positions, and generated notes sit on the beat grid instead of between grid lines. The
point always stays inside its frame, so ``parse_events`` + ``events_to_labels`` on the result
give back the same frames.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

ONSET_CODES = {1: "1", 2: "2", 3: "3", 4: "4"}
LONG_CODES = {5: "5", 6: "7"}
# Measure subdivisions tried per note, simplest first; SLOTS is a common multiple of all of them.
DIVISIONS = (1, 2, 4, 8, 16, 12, 24, 32, 48, 64, 96, 192, 384, 768, 2304)
SLOTS = 2304
EPS = 1e-6                 # seconds; keeps snapped points strictly inside their frame


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

    def slot_of(f, after=-1):
        """Global slot of the simplest subdivision point inside frame f and past slot ``after``
        (measure 0 starts at -offset); None if there is none."""
        lo = ((f * frame_ms / 1000) + offset + EPS) / measure_s
        hi = (((f + 1) * frame_ms / 1000) + offset - EPS) / measure_s
        for d in DIVISIONS:
            k = max(math.ceil(lo * d), (after * d) // SLOTS + 1)
            if k / d <= hi:
                return k * (SLOTS // d)
        return None

    first_frame = symbols[0][0]
    if first_frame * frame_ms / 1000 + offset < 0:   # notes before the first measure: prepend measures
        extra = math.ceil(-(first_frame * frame_ms / 1000 + offset) / measure_s)
        offset += extra * measure_s

    grid, balloons = {}, []
    for start, code, end in symbols:
        s = slot_of(start)
        if s is None:
            raise ValueError(f"tempo too slow to place frame {start} (bpm={bpm})")
        if end is not None:
            e = slot_of(end, after=s)   # the 8 comes after the start, inside the long note's last frame
            if e is None:               # one-frame roll on the frame's last point: start one slot earlier
                s, e = s - 1, s
            grid[e] = "8"
            if code == "7":
                duration = (end - start + 1) * frame_ms / 1000
                balloons.append(max(1, round(duration * balloon_hits_per_second)))
        grid[s] = code

    measures = []
    for m in range(max(grid) // SLOTS + 1):
        row = [grid.get(m * SLOTS + i, "0") for i in range(SLOTS)]
        step = math.gcd(SLOTS, *[i for i, c in enumerate(row) if c != "0"]) if any(c != "0" for c in row) else SLOTS
        measures.append("".join(row[::step]) + ",")

    header = [f"TITLE:{title}", f"BPM:{bpm:.12g}", f"WAVE:{wave}", f"OFFSET:{offset:.9f}".rstrip("0").rstrip("."), "",
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
