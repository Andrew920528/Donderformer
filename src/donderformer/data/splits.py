"""Reproducible song-level train/validation/test split.

Reads ``splits/manifest_strict.csv`` (from ``scan.py``) and writes
``splits/manifest.csv`` (accepted charts + ``song_id`` + ``split``) and
``splits/split.json`` (song IDs per split). All charts sharing one audio file
land in the same split.
"""
import csv
import hashlib
import json
import random
import re
from pathlib import PurePosixPath

from donderformer.paths import DataPaths

SEED = 42
SPLITS = ("train", "validation", "test")


def song_id(tja: str) -> str:
    """Stable ID from the chart's path relative to raw/, e.g. ``love_and_energy_3fa2c1``.

    It doesn't depend on manifest row order, so re-scanning keeps processed folders valid.
    """
    stem = re.sub(r"[^a-z0-9]+", "_", PurePosixPath(tja).stem.lower()).strip("_")
    digest = hashlib.sha1(tja.encode("utf-8")).hexdigest()[:6]
    return f"{stem}_{digest}"


def make_split(paths: DataPaths, seed: int = SEED, train_pct: int = 80, validation_pct: int = 10):
    with (paths.splits / "manifest_strict.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        accepted = [row for row in reader if row["status"] == "accepted"]

    # Sort before shuffling so the CSV row order does not affect the split.
    songs = sorted({row["audio"] for row in accepted})
    random.Random(seed).shuffle(songs)

    # Test gets the remaining songs after rounding down.
    n_songs = len(songs)
    n_train = n_songs * train_pct // 100
    n_validation = n_songs * validation_pct // 100
    audio_split = {}
    for i, audio in enumerate(songs):
        audio_split[audio] = ("train" if i < n_train
                              else "validation" if i < n_train + n_validation
                              else "test")

    split = {"seed": seed, **{name: [] for name in SPLITS}}
    for row in accepted:
        row["song_id"] = song_id(row["tja"])
        row["split"] = audio_split[row["audio"]]
        split[row["split"]].append(row["song_id"])

    ids = [row["song_id"] for row in accepted]
    assert len(ids) == len(set(ids)), "song_id collision"

    paths.split_file.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8")
    with paths.manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + ["song_id", "split"])
        writer.writeheader()
        writer.writerows(accepted)

    print("Accepted charts:", len(accepted))
    print("Unique songs/audio files:", n_songs)
    for name in SPLITS:
        count = sum(1 for a in songs if audio_split[a] == name)
        percentage = 100 * count / n_songs if n_songs else 0
        print(f"{name}: {count} songs ({percentage:.2f}%), {len(split[name])} charts")
    return split


if __name__ == "__main__":
    make_split(DataPaths())
