"""Scan raw ESE folders into one metadata row per chart (TJA file × course).

Paths are relative to ``DataPaths.raw`` so results are the same on every machine.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from donderformer.data.tja import course_sections, read_header, validate_chart

NOTE_CODES = "1234AB56789"


def short_id(path: str) -> str:
    """``<slug of file stem>_<6-char hash of path>``, stable across machines and re-scans."""
    stem = re.sub(r"[^a-z0-9]+", "_", PurePosixPath(path).stem.lower()).strip("_")
    return f"{stem}_{hashlib.sha1(path.encode('utf-8')).hexdigest()[:6]}"


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


def scan_raw(raw_dir: Path) -> list[dict]:
    """Scan every ``.tja`` under ``raw_dir``."""
    tjas = sorted(raw_dir.rglob("*.tja"))
    if not tjas:
        raise FileNotFoundError(f"No .tja files under {raw_dir}. Wrong data root?")
    return [row for tja in tjas for row in scan_tja(tja, raw_dir)]
