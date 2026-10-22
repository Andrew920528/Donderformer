"""Per-chart statistics for the dataset notebooks: note-type counts and note density.

Density = playable note events / audio duration (s). A long note counts once, at its
start. Audio durations are cached per audio file, so they don't depend on any split.
"""
from __future__ import annotations

import csv
from pathlib import Path

from donderformer.data.audio import load_mono_audio

NOTE_TYPES = {"Don": "1", "Ka": "2", "Big Don": "3A", "Big Ka": "4B", "Drumroll": "56", "Denden": "79"}
DURATION_SR = 8000


def note_counts(rows) -> list[int]:
    """Note events per type in NOTE_TYPES order, summed over ``rows``."""
    return [sum(int(r.get(f"count_{c}") or 0) for r in rows for c in codes) for codes in NOTE_TYPES.values()]


def audio_durations(raw_dir: Path, audios, cache: Path, compute: bool = False) -> dict[str, float]:
    """audio path (relative to ``raw_dir``) -> seconds. Decodes only files missing from ``cache``."""
    durations = {}
    if cache.is_file():
        with cache.open(newline="", encoding="utf-8") as f:
            durations = {r["audio"]: float(r["seconds"]) for r in csv.DictReader(f)}
    missing = sorted(set(audios) - set(durations))
    if compute and missing:
        failures = {}
        for audio in missing:
            try:
                seconds = len(load_mono_audio(raw_dir / audio, sr=DURATION_SR)) / DURATION_SR
                if seconds <= 0:
                    raise ValueError("empty audio")
                durations[audio] = seconds
            except Exception as e:
                failures[audio] = f"{type(e).__name__}: {e}"
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["audio", "seconds"])
            writer.writerows(sorted(durations.items()))
        print(f"decoded {len(missing) - len(failures)} audio files, {len(failures)} failed")
        for audio, error in list(failures.items())[:5]:
            print("  ", audio, error)
    return {a: durations[a] for a in audios if a in durations}


def densities(rows, durations: dict[str, float]) -> dict[str, float]:
    """chart_id -> note events per second, for charts whose audio duration is known."""
    return {r["chart_id"]: sum(note_counts([r])) / durations[r["audio"]] for r in rows if r["audio"] in durations}
