"""DataSplitter: choose which charts to keep and split them train/validation/test.

A split is a named folder under ``splits/`` and is never edited in place: change
the filters or ratios and give it a new name.

    splits/<name>/
    ├── config.json     # the DataSplitter parameters
    ├── manifest.csv    # one row per kept chart, with its split
    └── report.json     # what was kept, what was lost and why, distributions
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict, dataclass

from donderformer.data.scan import NOTE_CODES, scan_raw
from donderformer.data.tja import COURSES
from donderformer.paths import DataPaths

PARTS = ("train", "validation", "test")

MANIFEST_FIELDS = [
    "chart_id", "audio_id", "title", "genre", "course", "level", "bpm", "offset",
    "tja", "audio", "playable_notes", "has_measure",
    *[f"count_{c}" for c in NOTE_CODES],
    "split",
]


@dataclass
class DataSplitter:
    """``None`` for a filter means "keep all"."""

    name: str
    courses: tuple[str, ...] | None = None    # e.g. ("oni", "ura"); see tja.COURSES
    genres: tuple[str, ...] | None = None     # top-level ESE folders, e.g. ("01 Pop",)
    bpm_range: tuple[float, float] | None = None
    level_range: tuple[int, int] | None = None
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    seed: int = 42

    def filter_reasons(self, row) -> list[str]:
        """Why a usable chart is excluded by this split's filters."""
        reasons = []
        if self.courses is not None and row["course"] not in self.courses:
            reasons.append("course_not_selected")
        if self.genres is not None and row["genre"] not in self.genres:
            reasons.append("genre_not_selected")
        if self.bpm_range is not None and not (self.bpm_range[0] <= row["bpm"] <= self.bpm_range[1]):
            reasons.append("bpm_out_of_range")
        if self.level_range is not None and (
                row["level"] == "" or not (self.level_range[0] <= row["level"] <= self.level_range[1])):
            reasons.append("level_out_of_range")
        return reasons

    def assign_parts(self, kept) -> dict[str, str]:
        """audio file -> part. Charts sharing an audio file always land in the same part."""
        # Sort before shuffling so scan order does not affect the split.
        songs = sorted({row["audio"] for row in kept})
        random.Random(self.seed).shuffle(songs)

        # Test gets the remaining songs after rounding down.
        n = len(songs)
        n_train = int(n * self.ratios[0] + 1e-9)
        n_validation = int(n * self.ratios[1] + 1e-9)
        return {
            audio: "train" if i < n_train else "validation" if i < n_train + n_validation else "test"
            for i, audio in enumerate(songs)
        }

    def run(self, paths: DataPaths, overwrite: bool = False) -> dict:
        """Scan ``paths.raw``, filter, split, and write ``splits/<name>/``. Returns the report."""
        out = paths.split_dir(self.name)
        if (out / "manifest.csv").exists() and not overwrite:
            raise FileExistsError(
                f"Split {self.name!r} already exists at {out}. Splits are never edited in "
                "place; pick a new name (or pass overwrite=True if nothing uses it yet)."
            )

        rows = scan_raw(paths.raw)
        kept, lost = [], []
        for row in rows:
            reasons = row["reasons"] or self.filter_reasons(row)
            (lost if reasons else kept).append({**row, "reasons": reasons})

        part_of = self.assign_parts(kept)
        for row in kept:
            row["split"] = part_of[row["audio"]]

        ids = [row["chart_id"] for row in kept]
        assert len(ids) == len(set(ids)), "chart_id collision"

        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)

        report = make_report(self, rows, kept, lost)
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print_summary(report)
        return report


def make_report(splitter, rows, kept, lost) -> dict:
    lost_reasons = Counter(reason for row in lost for reason in row["reasons"])
    # Usable charts only: what each filter choice would have to work with.
    usable = [row for row in rows if not row["reasons"]]

    def by(key, items):
        return dict(sorted(Counter(str(row[key]) for row in items).items()))

    return {
        "name": splitter.name,
        "config": asdict(splitter),
        "totals": {
            "scanned_charts": len(rows),
            "usable_charts": len(usable),
            "kept_charts": len(kept),
            "kept_songs": len({row["audio"] for row in kept}),
            "lost_charts": len(lost),
        },
        # A chart can have several reasons, so these can sum to more than lost_charts.
        "lost_reasons": dict(lost_reasons.most_common()),
        "scanned_by_course": {c: n for c, n in sorted(Counter(r["course"] for r in rows).items(),
                                                      key=lambda kv: (COURSES + (kv[0],)).index(kv[0]))},
        "usable_by_course": by("course", usable),
        "kept_by": {
            "split": {part: sum(r["split"] == part for r in kept) for part in PARTS},
            "course": by("course", kept),
            "genre": by("genre", kept),
            "level": by("level", kept),
            "split_course": {part: by("course", [r for r in kept if r["split"] == part]) for part in PARTS},
        },
        "kept_bpm": sorted(row["bpm"] for row in kept),
        "kept": [row["chart_id"] for row in kept],
        "lost": [
            {"chart_id": row["chart_id"], "title": row["title"], "course": row["course"],
             "tja": row["tja"], "reasons": row["reasons"]}
            for row in lost
        ],
    }


def print_summary(report):
    t = report["totals"]
    print(f"Scanned {t['scanned_charts']} charts, {t['usable_charts']} usable, "
          f"kept {t['kept_charts']} charts from {t['kept_songs']} songs")
    print("Kept per split:", report["kept_by"]["split"])
    print("Kept per course:", report["kept_by"]["course"])
    print("Top lost reasons:")
    for reason, count in list(report["lost_reasons"].items())[:10]:
        print(f"  {reason}: {count}")


def manifest_hash(paths: DataPaths, split: str) -> str:
    """Fingerprint of a split's manifest, recorded by feature extraction."""
    return hashlib.sha256((paths.split_dir(split) / "manifest.csv").read_bytes()).hexdigest()


def read_manifest(paths: DataPaths, split: str, part: str | None = None) -> list[dict]:
    """Rows of ``splits/<split>/manifest.csv``, optionally only one part."""
    with (paths.split_dir(split) / "manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if part is None or r["split"] == part]
