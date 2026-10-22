"""DataSplitter: scan raw charts, choose which to keep and split them train/validation/test.

Scanning gives one metadata row per chart (TJA file x course), with paths
relative to ``DataPaths.raw`` so results are the same on every machine.

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
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from donderformer.data.tja import COURSES, course_sections, read_header, validate_chart
from donderformer.paths import DataPaths

NOTE_CODES = "1234AB56789"


def short_id(path: str) -> str:
    """``<slug of file stem>_<6-char hash of path>``, stable across machines and re-scans."""
    stem = re.sub(r"[^a-z0-9]+", "_", PurePosixPath(path).stem.lower()).strip("_")
    return f"{stem}_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:6]}"


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

    @staticmethod
    def scan_tja(tja: Path, raw_dir: Path) -> list[dict]:
        """One row per course section in ``tja``. ``reasons`` is empty when the chart is usable."""
        rel_tja = tja.relative_to(raw_dir).as_posix()
        parts = PurePosixPath(rel_tja).parts
        base = {
            "title": tja.stem,
            # ESE layout is <genre>/<song>/<file>.tja
            "genre": parts[0] if len(parts) >= 3 else "",
            "tja": rel_tja,
            "tja_id": short_id(rel_tja),
            "audio": "",
            "audio_id": "",
            "bpm": "",
            "offset": "",
        }
        base["chart_id"] = base["tja_id"]   # replaced by <tja_id>_<course> per course below

        try:
            text = tja.read_text(encoding="utf-8-sig", errors="replace")
        except Exception as e:
            return [{**base, "course": "", "level": "", "reasons": [f"read_error:{type(e).__name__}"]}]

        base["title"] = read_header(text, "TITLE") or tja.stem
        file_reasons = []

        wave = read_header(text, "WAVE")
        if wave is None:
            file_reasons.append("missing_wave")
        else:
            audio = tja.parent / wave
            if not audio.is_file():
                file_reasons.append("missing_audio")
            else:
                base["audio"] = audio.relative_to(raw_dir).as_posix()
                base["audio_id"] = short_id(base["audio"])

        bpm_raw = read_header(text, "BPM")
        if bpm_raw is None:
            file_reasons.append("missing_bpm")
        else:
            try:
                base["bpm"] = float(bpm_raw)
                if base["bpm"] <= 0:
                    file_reasons.append("invalid_bpm")
            except ValueError:
                file_reasons.append("invalid_bpm")

        base["offset"] = 0.0
        offset_raw = read_header(text, "OFFSET")
        if offset_raw is not None:
            try:
                base["offset"] = float(offset_raw)
            except ValueError:
                file_reasons.append("invalid_offset")

        sections = course_sections(text)
        if not sections:
            return [{**base, "course": "", "level": "", "reasons": file_reasons + ["no_charts"]}]

        by_course = {}
        for section in sections:
            by_course.setdefault(section.course, []).append(section)

        rows = []
        for course, group in by_course.items():
            reasons = list(file_reasons)
            row = {**base, "course": course, "level": group[0].level if group[0].level is not None else "",
                   "chart_id": f"{base['tja_id']}_{course}"}

            if len(group) > 1:
                # e.g. STYLE:Double charts with separate P1/P2 sections
                reasons.append("multiple_sections")
                playable_notes, has_measure, counts = 0, False, {}
            else:
                chart_reasons, counts, playable_notes, has_measure = validate_chart(group[0].lines)
                reasons.extend(chart_reasons)

            row.update({
                "playable_notes": playable_notes,
                "has_measure": has_measure,
                **{f"count_{c}": counts.get(c, 0) for c in NOTE_CODES},
                "reasons": sorted(set(reasons)),
            })
            rows.append(row)

        return rows

    @staticmethod
    def scan(raw_dir: Path) -> list[dict]:
        """Scan every ``.tja`` under ``raw_dir`` into one metadata row per chart (TJA file x course)."""
        tjas = sorted(raw_dir.rglob("*.tja"))
        if not tjas:
            raise FileNotFoundError(f"No .tja files under {raw_dir}. Wrong data root?")
        return [row for tja in tjas for row in DataSplitter.scan_tja(tja, raw_dir)]

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

        rows = self.scan(paths.raw)
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
