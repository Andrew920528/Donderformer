"""TJA chart parsing: header values, course sections, validation and note events."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Canonical course names, easiest first. "Edit" in TJA is the Ura (inner) Oni chart.
COURSES = ("easy", "normal", "hard", "oni", "ura")
COURSE_ALIASES = {
    "easy": "easy", "0": "easy",
    "normal": "normal", "1": "normal",
    "hard": "hard", "2": "hard",
    "oni": "oni", "3": "oni",
    "edit": "ura", "ura": "ura", "4": "ura",
}

CLASS_NAMES = ["none", "don", "ka", "DON", "KA", "drumroll", "denden"]

# Note code -> class index (see CLASS_NAMES).
ONSET = {"1": 1, "2": 2, "3": 3, "A": 3, "4": 4, "B": 4}
LONG_START = {"5": 5, "6": 5, "7": 6, "9": 6}

SUPPORTED_NOTES = set("0123456789AB")
IGNORED_COMMANDS = {
    "#START",
    "#END",
    "#GOGOSTART",
    "#GOGOEND",
    "#BARLINEON",
    "#BARLINEOFF",
    "#SCROLL",
}

DISALLOWED_COMMANDS = {
    "#BPMCHANGE": "bpm_change",
    "#DELAY": "delay",
    "#BRANCHSTART": "branching",
    "#BRANCHEND": "branching",
    "#N": "branching",
    "#E": "branching",
    "#M": "branching",
}


def read_header(text, key):
    """First ``KEY:value`` line in the file, or None."""
    match = re.search(
        rf"(?mi)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$",
        text
    )
    return match.group(1).strip() if match else None


@dataclass
class CourseSection:
    course: str             # canonical name, or the raw COURSE value if unknown
    level: int | None
    lines: list = field(default_factory=list)   # raw lines between #START and #END


def course_sections(text):
    """Every ``#START``..``#END`` block with the COURSE/LEVEL declared before it."""
    sections = []

    course = None
    level = None
    in_chart = False
    chart_lines = []

    for line in text.splitlines():
        stripped = line.split("//", 1)[0].strip()

        if not in_chart and ":" in stripped:
            name, value = (s.strip() for s in stripped.split(":", 1))
            name = name.upper()

            if name == "COURSE":
                course = COURSE_ALIASES.get(value.lower(), value.lower())
                level = None
            elif name == "LEVEL":
                try:
                    level = int(float(value))
                except ValueError:
                    level = None
            continue

        if stripped.upper().startswith("#START"):
            in_chart = True
            chart_lines = []
            continue

        if stripped.upper().startswith("#END") and in_chart:
            sections.append(CourseSection(course or "unspecified", level, chart_lines[:]))
            in_chart = False
            chart_lines = []
            continue

        if in_chart:
            chart_lines.append(line)

    return sections


def validate_chart(lines):
    """Reasons this chart can't be parsed (empty if fine), note counts, playable notes, has #MEASURE."""
    reasons = []
    note_counts = Counter()
    playable_notes = 0
    open_long_note = None
    has_measure = False

    for raw_line in lines:
        line = raw_line.split("//", 1)[0].strip()

        if not line:
            continue

        upper = line.upper()

        if upper.startswith("#MEASURE"):
            has_measure = True

            match = re.match(
                r"(?i)^#MEASURE\s+(\d+)\s*/\s*(\d+)\s*$",
                line
            )

            if not match:
                reasons.append("invalid_measure")
            else:
                numerator = int(match.group(1))
                denominator = int(match.group(2))

                if numerator <= 0 or denominator <= 0:
                    reasons.append("invalid_measure")

            continue

        disallowed_found = False

        for command, reason in DISALLOWED_COMMANDS.items():
            if upper == command or upper.startswith(command + " "):
                reasons.append(reason)
                disallowed_found = True
                break

        if disallowed_found:
            continue

        if upper.startswith("#"):
            command_name = upper.split()[0]

            if command_name in IGNORED_COMMANDS:
                continue

            reasons.append(f"unsupported_command:{command_name}")
            continue

        note_text = line.replace(",", "").replace(" ", "").replace("\t", "")

        if not note_text:
            continue

        for char in note_text:
            c = char.upper()

            if c not in SUPPORTED_NOTES:
                reasons.append(f"unsupported_note:{char}")
                continue

            note_counts[c] += 1

            if c == "0":
                continue

            if c in {"1", "2", "3", "4", "A", "B"}:
                playable_notes += 1

                if open_long_note is not None:
                    reasons.append("note_inside_open_long_note")

                continue

            if c in {"5", "6"}:
                playable_notes += 1

                if open_long_note is not None:
                    reasons.append("nested_long_note")

                open_long_note = "roll"
                continue

            if c in {"7", "9"}:
                playable_notes += 1

                if open_long_note is not None:
                    reasons.append("nested_long_note")

                open_long_note = "denden"
                continue

            if c == "8":
                if open_long_note is None:
                    reasons.append("unmatched_8")
                else:
                    open_long_note = None

    if open_long_note is not None:
        reasons.append("unclosed_long_note")

    if playable_notes == 0:
        reasons.append("empty_chart")

    return sorted(set(reasons)), note_counts, playable_notes, has_measure


FORBIDDEN_COMMANDS = {
    "BPMCHANGE", "DELAY", "BRANCHSTART", "BRANCHEND", "N", "E", "M"
}

ALLOWED_COMMANDS = {
    "MEASURE", "GOGOSTART", "GOGOEND",
    "SCROLL", "BARLINEON", "BARLINEOFF"
}


def parse_events(lines, bpm, offset, where="chart"):
    """Timed note events ``(start_s, class, end_s or None)`` for one course section.

    A second safety gate after ``validate_chart``: accepts ``#MEASURE``, ignores
    visual commands, and raises for timing-changing or unknown commands.
    """
    body = [line.split("//", 1)[0].strip() for line in lines]
    body = [line for line in body if line]

    time = -offset
    beats_per_measure = 4.0
    events = []
    opened = None
    measure_notes = []

    def process_measure():
        nonlocal time, opened, measure_notes

        if not measure_notes:
            time += beats_per_measure * 60.0 / bpm
            return

        measure_duration = beats_per_measure * 60.0 / bpm
        step = measure_duration / len(measure_notes)

        for note in measure_notes:
            if note == "8":
                if opened is None:
                    raise ValueError(f"{where}: unmatched 8")

                events.append((opened[0], opened[1], time))
                opened = None

            elif note in LONG_START:
                if opened is not None:
                    raise ValueError(f"{where}: nested long note")

                opened = (time, LONG_START[note])

            elif note in ONSET:
                if opened is not None:
                    raise ValueError(f"{where}: note inside long note")

                events.append((time, ONSET[note], None))

            elif note != "0":
                raise ValueError(f"{where}: unsupported note {note!r}")

            time += step

    for line in body:
        if line.startswith("#"):
            name, _, argument = line[1:].partition(" ")
            name = name.upper()
            argument = argument.strip()

            if name in FORBIDDEN_COMMANDS:
                raise ValueError(f"{where}: forbidden #{name}")

            if name == "MEASURE":
                num, den = argument.split("/")
                beats_per_measure = 4.0 * float(num) / float(den)

            elif name not in ALLOWED_COMMANDS:
                raise ValueError(f"{where}: unsupported #{name}")

            continue

        for char in line:
            if char == ",":
                process_measure()
                measure_notes = []

            elif not char.isspace():
                measure_notes.append(char.upper())

    if measure_notes:
        process_measure()

    if opened is not None:
        raise ValueError(f"{where}: unclosed long note")

    return events
