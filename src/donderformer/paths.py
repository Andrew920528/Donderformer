"""Single place where data locations are resolved.

The data root is chosen in this order:
1. the ``root`` argument,
2. the ``DONDER_DATA`` environment variable,
3. ``<repo>/data``.

Every data root has the same layout, so the toy dataset is just another root::

    data/
    ├── raw/                 # ESE song folders (genre/song/*.tja, audio)
    ├── splits/              # manifest.csv (charts + split), scan_report.json (committed)
    └── processed/<run>/     # preprocessing output + a manifest.csv snapshot of its charts
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOY_DATA_ROOT = REPO_ROOT / "playground" / "toy_dataset"


@dataclass(frozen=True)
class DataPaths:
    root: Path

    def __init__(self, root: str | os.PathLike | None = None):
        if root is None:
            root = os.environ.get("DONDER_DATA", REPO_ROOT / "data")
        object.__setattr__(self, "root", Path(root).expanduser().resolve())

    @classmethod
    def toy(cls) -> "DataPaths":
        return cls(TOY_DATA_ROOT)

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def manifest(self) -> Path:
        return self.splits / "manifest.csv"

    @property
    def scan_report(self) -> Path:
        return self.splits / "scan_report.json"

    def processed(self, run_name: str) -> Path:
        return self.root / "processed" / run_name

    def raw_path(self, relative: str | os.PathLike) -> Path:
        """Resolve a manifest path (stored relative to ``raw/``)."""
        return self.raw / relative


def outputs_dir() -> Path:
    """Checkpoints and generated charts; override with ``DONDER_OUTPUTS``."""
    return Path(os.environ.get("DONDER_OUTPUTS", REPO_ROOT / "outputs")).expanduser().resolve()
