from pathlib import Path
import csv
import json
import os
import random

SEED = 42
ROOT = Path(__file__).resolve().parent

with (ROOT / "manifest_strict.csv").open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    accepted = [row for row in reader if row["status"] == "accepted"]

# Sort before shuffling so the CSV row order does not affect the split.
songs = sorted({os.path.normpath(row["audio"]) for row in accepted})
random.Random(SEED).shuffle(songs)

# Test gets the remaining songs after rounding down.
n_songs = len(songs)
n_train = n_songs * 80 // 100
n_validation = n_songs * 10 // 100

split = {
    "seed": SEED,
    "train": songs[:n_train],
    "validation": songs[n_train:n_train + n_validation],
    "test": songs[n_train + n_validation:],
}

train = set(split["train"])
validation = set(split["validation"])
test = set(split["test"])
assert train.isdisjoint(validation)
assert train.isdisjoint(test)
assert validation.isdisjoint(test)

split_by_audio = {}
for name in ("train", "validation", "test"):
    for audio in split[name]:
        split_by_audio[audio] = name

for row in accepted:
    row["split"] = split_by_audio[os.path.normpath(row["audio"])]

with (ROOT / "split.json").open("w", encoding="utf-8") as f:
    json.dump(split, f, indent=2, ensure_ascii=False)

with (ROOT / "manifest_with_split.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames + ["split"])
    writer.writeheader()
    writer.writerows(accepted)

print("Accepted charts:", len(accepted))
print("Unique songs/audio files:", n_songs)
for name in ("train", "validation", "test"):
    count = len(split[name])
    percentage = 100 * count / n_songs if n_songs else 0
    print(f"{name}: {count} songs ({percentage:.2f}%)")
