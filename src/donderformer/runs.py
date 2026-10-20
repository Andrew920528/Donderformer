"""Training runs: one folder per run, so a new run never overwrites an old one.

    outputs/runs/<model>/<YYYYMMDD_HHMMSS>[-<alias>]-<split>-<extractor>/
    ├── config.json   # split, extractor, window, hyperparameters, git commit, who/where
    ├── model.pt      # model_state_dict + losses + the same config
    └── eval/         # metrics, generated charts and plots written by the notebook

``outputs/`` is ``<repo>/outputs`` unless ``DONDER_OUTPUTS`` is set (see ``paths.outputs_dir``).
"""
from __future__ import annotations

import getpass
import json
import re
import socket
import subprocess
from datetime import datetime
from pathlib import Path

import torch

from donderformer.paths import REPO_ROOT, outputs_dir

ALIAS_PATTERN = re.compile(r"[A-Za-z0-9_.]+")


def runs_dir(model_name: str) -> Path:
    return outputs_dir() / "runs" / model_name


def run_name(split: str, extractor_id: str, alias: str = "", when: datetime | None = None) -> str:
    """``<YYYYMMDD_HHMMSS>[-<alias>]-<split>-<extractor>``; ``-`` only ever separates fields."""
    if alias and not ALIAS_PATTERN.fullmatch(alias):
        raise ValueError(f"alias {alias!r} may only use letters, digits, '_' and '.'")
    stamp = (when or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return "-".join(part for part in (stamp, alias, split, extractor_id) if part)


def new_run_dir(model_name: str, split: str, extractor_id: str, alias: str = "") -> Path:
    """Create a fresh run folder; fails rather than reuse an existing one."""
    run_dir = runs_dir(model_name) / run_name(split, extractor_id, alias)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def save_run(run_dir: Path, model, config: dict, losses: dict) -> Path:
    """Write ``config.json`` and ``model.pt`` into ``run_dir``; returns the checkpoint path."""
    run_dir = Path(run_dir)
    config = {**config, "run": run_dir.name, "saved_at": datetime.now().isoformat(timespec="seconds"),
              "user": getpass.getuser(), "host": socket.gethostname(), "git": git_info()}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    checkpoint = run_dir / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "losses": losses, "config": config}, checkpoint)
    return checkpoint


def load_run(run_dir: Path, model, map_location="cpu") -> dict:
    """Load ``run_dir/model.pt`` into ``model``; returns the run's config."""
    run_dir = Path(run_dir)
    checkpoint = torch.load(run_dir / "model.pt", map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    config_path = run_dir / "config.json"
    return json.loads(config_path.read_text()) if config_path.exists() else checkpoint.get("config", {})
