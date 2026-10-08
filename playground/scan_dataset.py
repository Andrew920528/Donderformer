# We scan qualified songs from the PACE dataset.
from pathlib import Path
import csv
import re
from collections import Counter

ROOT = Path("ESE")

SUPPORTED_NOTES = set("0123456789AB")
IGNORED_COMMANDS = {
    "#START",
    "#END",
    "#GOGOSTART",
    "#GOGOEND",
    "#BARLINEON",
    "#BARLINEOFF",
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


def get_metadata(text, key):
    match = re.search(
        rf"(?mi)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$",
        text
    )
    return match.group(1).strip() if match else None


def find_oni_sections(text):
    lines = text.splitlines()
    sections = []

    current_course = None
    in_chart = False
    chart_lines = []

    for line in lines:
        stripped = line.strip()

        course_match = re.match(
            r"(?i)^COURSE\s*:\s*(.+?)\s*$",
            stripped
        )

        if course_match and not in_chart:
            current_course = course_match.group(1).strip()

        if stripped.upper().startswith("#START"):
            in_chart = True
            chart_lines = []
            continue

        if stripped.upper().startswith("#END") and in_chart:
            if current_course and current_course.lower() in {"oni", "3"}:
                sections.append(chart_lines[:])

            in_chart = False
            chart_lines = []
            continue

        if in_chart:
            chart_lines.append(line)

    return sections


def validate_oni_chart(lines):
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

            if command_name == "#SCROLL":
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


rows = []

for tja in ROOT.rglob("*.tja"):
    try:
        text = tja.read_text(
            encoding="utf-8-sig",
            errors="replace"
        )
    except Exception as e:
        rows.append({
            "title": tja.stem,
            "tja": str(tja),
            "audio": "",
            "bpm": "",
            "offset": "",
            "oni_sections": 0,
            "playable_notes": 0,
            "has_measure": False,
            "status": "rejected",
            "reason": f"read_error:{type(e).__name__}",
        })
        continue

    title = get_metadata(text, "TITLE") or tja.stem
    wave = get_metadata(text, "WAVE")
    bpm_raw = get_metadata(text, "BPM")
    offset_raw = get_metadata(text, "OFFSET")

    reasons = []

    audio = tja.parent / wave if wave else None

    if wave is None:
        reasons.append("missing_wave")
    elif not audio.is_file():
        reasons.append("missing_audio")

    bpm = None

    if bpm_raw is None:
        reasons.append("missing_bpm")
    else:
        try:
            bpm = float(bpm_raw)

            if bpm <= 0:
                reasons.append("invalid_bpm")

        except ValueError:
            reasons.append("invalid_bpm")

    offset = 0.0

    if offset_raw is not None:
        try:
            offset = float(offset_raw)
        except ValueError:
            reasons.append("invalid_offset")

    oni_sections = find_oni_sections(text)

    if not oni_sections:
        reasons.append("no_oni")

    if len(oni_sections) > 1:
        reasons.append("multiple_oni_sections")

    playable_notes = 0
    has_measure = False
    note_counts = Counter()

    if len(oni_sections) == 1:
        chart_reasons, counts, playable_notes, has_measure = \
            validate_oni_chart(oni_sections[0])

        reasons.extend(chart_reasons)
        note_counts.update(counts)

    reasons = sorted(set(reasons))

    rows.append({
        "title": title,
        "tja": str(tja),
        "audio": str(audio) if audio else "",
        "bpm": bpm if bpm is not None else "",
        "offset": offset,
        "oni_sections": len(oni_sections),
        "playable_notes": playable_notes,
        "has_measure": has_measure,
        "count_1": note_counts["1"],
        "count_2": note_counts["2"],
        "count_3": note_counts["3"],
        "count_4": note_counts["4"],
        "count_A": note_counts["A"],
        "count_B": note_counts["B"],
        "count_5": note_counts["5"],
        "count_6": note_counts["6"],
        "count_7": note_counts["7"],
        "count_8": note_counts["8"],
        "count_9": note_counts["9"],
        "status": "accepted" if not reasons else "rejected",
        "reason": ";".join(reasons),
    })


fieldnames = [
    "title",
    "tja",
    "audio",
    "bpm",
    "offset",
    "oni_sections",
    "playable_notes",
    "has_measure",
    "count_1",
    "count_2",
    "count_3",
    "count_4",
    "count_A",
    "count_B",
    "count_5",
    "count_6",
    "count_7",
    "count_8",
    "count_9",
    "status",
    "reason",
]

with open(
    "manifest_strict.csv",
    "w",
    newline="",
    encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


accepted = [r for r in rows if r["status"] == "accepted"]
rejected = [r for r in rows if r["status"] == "rejected"]

reason_counts = Counter()

for row in rejected:
    for reason in row["reason"].split(";"):
        if reason:
            reason_counts[reason] += 1


print("Total TJA:", len(rows))
print("Strict accepted:", len(accepted))
print("Rejected:", len(rejected))

print("\nTop rejection reasons:")
for reason, count in reason_counts.most_common(20):
    print(f"{reason}: {count}")

print("\nFirst 20 accepted:")
for row in accepted[:20]:
    print(
        row["title"],
        "| BPM:", row["bpm"],
        "| notes:", row["playable_notes"],
        "| measure:", row["has_measure"],
    )

import random

random.seed(42)
selected = random.sample(accepted, 15)

print("\nSelected 15:")
for i, row in enumerate(selected, 1):
    print(
        i,
        row["title"],
        "| BPM:", row["bpm"],
        "| notes:", row["playable_notes"],
        "| measure:", row["has_measure"],
        "|", row["tja"]
    )

with open("selected_15.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected)