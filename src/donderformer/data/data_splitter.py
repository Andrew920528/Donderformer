"""DataSplitter: scan raw charts, choose which to keep and split them train/validation/test.

Scanning gives one metadata row per chart (TJA file x course), with paths
relative to ``DataPaths.raw`` so results are the same on every machine.

A split is a named folder under ``splits/`` and is never edited in place: change
the filters or ratios and give it a new name.

Charts are split by group, so every chart of a group lands in the same part. A group is
one audio file (``group_by="audio"``, the original behaviour) or one song folder
(``group_by="song"``, which also keeps cover versions of a song together). Groups can be
stratified by star level, or copied from an existing split with ``song_parts_from`` so
that, e.g., an all-course split shares the Oni split's validation and test songs.

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

# Filter order for the funnel in report.json: a lost chart is counted at the first stage
# that drops it. Reasons not listed fall under "unparseable".
FUNNEL_STAGES = [
    ("missing_or_invalid_file", ("read_error", "missing_wave", "missing_audio", "missing_bpm",
                                 "invalid_bpm", "invalid_offset", "no_charts")),
    ("multiple_sections", ("multiple_sections",)),
    ("bpm_change", ("bpm_change",)),
    ("delay", ("delay",)),
    ("branching", ("branching",)),
    ("unparseable", None),
    ("course_not_selected", ("course_not_selected",)),
    ("genre_not_selected", ("genre_not_selected",)),
    ("bpm_out_of_range", ("bpm_out_of_range",)),
    ("level_out_of_range", ("level_out_of_range",)),
]

# Star-level strata: every level up to LOW_LEVEL_MAX shares one stratum (they're rare in Oni).
LOW_LEVEL_MAX = 6

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
    group_by: str = "audio"                   # "audio" (one audio file) or "song" (one song folder)
    stratify_by_level: bool = False           # balance star levels across parts (see level_stratum)
    song_parts_from: str | None = None        # copy group -> part from this existing split

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

    def group_key(self, row) -> str:
        """Charts with the same key always land in the same part."""
        if self.group_by == "audio":
            return row["audio"]
        if self.group_by == "song":
            return PurePosixPath(row["tja"]).parent.as_posix()
        raise ValueError(f"group_by must be 'audio' or 'song', not {self.group_by!r}")

    def assign_parts(self, kept, paths: DataPaths | None = None) -> dict[str, str]:
        """group key -> part. Charts sharing a group always land in the same part."""
        if self.song_parts_from is not None:
            return self.inherited_parts(kept, paths)

        # Sort before shuffling so scan order does not affect the split.
        groups = sorted({self.group_key(row) for row in kept})
        rng = random.Random(self.seed)
        if not self.stratify_by_level:
            rng.shuffle(groups)
            # Test gets the remaining groups after rounding down (same as before stratification).
            n = len(groups)
            n_train = int(n * self.ratios[0] + 1e-9)
            n_validation = int(n * self.ratios[1] + 1e-9)
            return {
                g: "train" if i < n_train else "validation" if i < n_train + n_validation else "test"
                for i, g in enumerate(groups)
            }

        # Shuffle within each level stratum, then cut with cumulative rounding: each stratum gets
        # its share of every part, and remainders carry over so the totals still match the ratios.
        strata = group_strata(kept, self.group_key)
        parts, done = {}, 0
        for stratum in sorted(set(strata.values()), key=stratum_order):
            members = [g for g in groups if strata[g] == stratum]
            rng.shuffle(members)
            before, after = part_counts(done, self.ratios), part_counts(done + len(members), self.ratios)
            sequence = [part for part, a, b in zip(PARTS, before, after) for _ in range(b - a)]
            parts.update(zip(members, sequence))
            done += len(members)
        return parts

    def inherited_parts(self, kept, paths: DataPaths) -> dict[str, str]:
        """Parts copied from ``song_parts_from``; groups it doesn't have go to train."""
        source_dir = paths.split_dir(self.song_parts_from)
        if not (source_dir / "manifest.csv").exists():
            raise FileNotFoundError(f"song_parts_from={self.song_parts_from!r}: {source_dir} has no manifest. "
                                    "Make that split first.")
        source_config = json.loads((source_dir / "config.json").read_text())
        if source_config.get("group_by", "audio") != self.group_by:
            raise ValueError(f"{self.song_parts_from} was grouped by {source_config.get('group_by', 'audio')!r}, "
                             f"this split by {self.group_by!r}; they must match")

        source = {}
        for row in read_manifest(paths, self.song_parts_from):
            source.setdefault(self.group_key(row), row["split"])
        return {g: source.get(g, "train") for g in sorted({self.group_key(row) for row in kept})}

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

        part_of = self.assign_parts(kept, paths)
        for row in kept:
            row["split"] = part_of[self.group_key(row)]

        ids = [row["chart_id"] for row in kept]
        assert len(ids) == len(set(ids)), "chart_id collision"
        leaks = check_leaks(kept, self.group_key)

        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        with (out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(kept)

        report = make_report(self, rows, kept, lost)
        report["leak_check"] = leaks
        if self.song_parts_from is not None:
            report["song_parts_from"] = inheritance_report(self, paths, kept)
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
            "kept_groups": len({splitter.group_key(row) for row in kept}),
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
            "split_level": {part: by("level", [r for r in kept if r["split"] == part]) for part in PARTS},
        },
        "funnel": funnel(rows, kept, lost),
        "funnel_by_course": {course: funnel([r for r in rows if r["course"] == course],
                                            [r for r in kept if r["course"] == course],
                                            [r for r in lost if r["course"] == course])
                             for course in sorted({r["course"] for r in rows},
                                                  key=lambda c: (COURSES + (c,)).index(c))},
        "kept_bpm": sorted(row["bpm"] for row in kept),
        "kept": [row["chart_id"] for row in kept],
        "lost": [
            {"chart_id": row["chart_id"], "title": row["title"], "course": row["course"],
             "tja": row["tja"], "reasons": row["reasons"]}
            for row in lost
        ],
    }


def part_counts(n, ratios):
    """(train, validation, test) group counts for the first ``n`` groups; non-decreasing in ``n``."""
    total = sum(ratios)
    validation = int(n * ratios[1] / total + 0.5)
    test = int(n * ratios[2] / total + 0.5)
    return n - validation - test, validation, test


def level_stratum(level) -> str:
    """``"<=6"``, ``"7"`` … ``"10"``, or ``"unknown"``."""
    if level in ("", None):
        return "unknown"
    level = int(float(level))
    return f"<={LOW_LEVEL_MAX}" if level <= LOW_LEVEL_MAX else str(level)


def stratum_order(stratum):
    return (stratum == "unknown", 0 if stratum.startswith("<=") else int(stratum) if stratum.isdigit() else 99)


def group_strata(kept, group_key) -> dict[str, str]:
    """group -> level stratum: its Oni level if it has an Oni chart, else its hardest kept course's level."""
    charts = {}
    for row in kept:
        charts.setdefault(group_key(row), []).append(row)
    strata = {}
    for group, rows in charts.items():
        oni = [r for r in rows if r["course"] == "oni" and r["level"] not in ("", None)]
        if oni:
            strata[group] = level_stratum(oni[0]["level"])
            continue
        known = [r for r in rows if r["course"] in COURSES and r["level"] not in ("", None)]
        hardest = max(known, key=lambda r: COURSES.index(r["course"]), default=None)
        strata[group] = level_stratum(hardest["level"] if hardest else "")
    return strata


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", title.lower())


def check_leaks(kept, group_key) -> dict:
    """Assert no group or audio file spans two parts; list song titles that do (possible covers)."""
    def parts_by(key):
        seen = {}
        for row in kept:
            seen.setdefault(key(row), set()).add(row["split"])
        return {k: sorted(v) for k, v in seen.items() if len(v) > 1}

    crossing_groups = parts_by(group_key)
    assert not crossing_groups, f"groups split across parts: {list(crossing_groups)[:5]}"
    crossing_audio = parts_by(lambda r: r["audio"])
    assert not crossing_audio, f"audio files split across parts: {list(crossing_audio)[:5]}"

    # Same title in two parts is usually two different songs, but worth a look.
    titles = parts_by(lambda r: normalized_title(r["title"]))
    return {"groups_crossing": 0, "audio_crossing": 0,
            "titles_crossing": {title: parts for title, parts in sorted(titles.items()) if title}}


def inheritance_report(splitter, paths, kept) -> dict:
    """What a ``song_parts_from`` split took over from its source, and the leak check against it."""
    source_rows = read_manifest(paths, splitter.song_parts_from)
    source = {splitter.group_key(r): r["split"] for r in source_rows}
    groups = {splitter.group_key(r): r["split"] for r in kept}
    leaked = [g for g, part in groups.items() if g in source and source[g] != part]
    assert not leaked, f"groups whose part differs from {splitter.song_parts_from}: {leaked[:5]}"
    return {
        "split": splitter.song_parts_from,
        "manifest_sha256": manifest_hash(paths, splitter.song_parts_from),
        "inherited_groups": sum(g in source for g in groups),
        "new_groups_to_train": sum(g not in source for g in groups),
        "source_groups_missing_here": sum(g not in groups for g in source),
    }


def funnel(rows, kept, lost) -> list[dict]:
    """Charts remaining after each filter stage, in FUNNEL_STAGES order."""
    def stage_of(reasons):
        for name, matches in FUNNEL_STAGES:
            if matches is not None and any(r.split(":")[0] in matches for r in reasons):
                return name
        return "unparseable"

    dropped = Counter(stage_of(row["reasons"]) for row in lost)
    remaining = len(rows)
    steps = [{"stage": "scanned", "dropped": 0, "remaining": remaining}]
    for name, _ in FUNNEL_STAGES:
        remaining -= dropped[name]
        steps.append({"stage": name, "dropped": dropped[name], "remaining": remaining})
    assert remaining == len(kept), (remaining, len(kept))
    return steps


def print_summary(report):
    t = report["totals"]
    print(f"Scanned {t['scanned_charts']} charts, {t['usable_charts']} usable, "
          f"kept {t['kept_charts']} charts from {t['kept_songs']} audio files ({t['kept_groups']} groups)")
    print("Kept per split:", report["kept_by"]["split"])
    print("Kept per course:", report["kept_by"]["course"])
    print("Funnel:", ", ".join(f"{s['stage']} {s['remaining']}" for s in report["funnel"]))
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
