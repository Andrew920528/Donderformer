"""Copy random song folders from the full dataset into the toy dataset.

Only songs with at least one usable chart (one the parser can handle) are picked.
Folders keep their <genre>/<song>/ layout, and songs already in the toy dataset are skipped.

    uv run python playground/toy_dataset/make_toy_dataset.py            # 10 songs from data/raw
    uv run python playground/toy_dataset/make_toy_dataset.py --n 5 --seed 1 --src /path/to/raw

Afterwards, re-run 01_make_split.ipynb with DataPaths.toy().
"""
import argparse
import random
import shutil
from pathlib import Path

from donderformer.data.data_splitter import DataSplitter
from donderformer.paths import DataPaths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="number of songs to add (default 10)")
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: different every run)")
    parser.add_argument("--src", type=Path, default=DataPaths().raw, help="raw folder to copy from (default data/raw)")
    args = parser.parse_args()

    dst = DataPaths.toy().raw
    # Song folder names already present, whatever their genre nesting.
    existing = {tja.parent.name for tja in dst.rglob("*.tja")}

    # Song folders that contain at least one usable chart.
    usable = {}
    for row in DataSplitter.scan(args.src):
        if not row["reasons"]:
            song_dir = args.src / Path(row["tja"]).parent
            usable[song_dir] = row["title"]
    candidates = sorted(d for d in usable if d.name not in existing)

    if len(candidates) < args.n:
        raise SystemExit(f"Only {len(candidates)} new usable songs under {args.src}, asked for {args.n}")

    picked = random.Random(args.seed).sample(candidates, args.n)
    for song_dir in sorted(picked):
        target = dst / song_dir.relative_to(args.src)
        shutil.copytree(song_dir, target)
        print(f"added {target.relative_to(dst)}  ({usable[song_dir]})")

    print(f"\nToy dataset now has {len(existing) + args.n} songs in {dst}")


if __name__ == "__main__":
    main()
