"""Chart any audio file with saved runs and package the result like ``ChartExporter`` does.

    outputs/exports/<YYYYMMDD_HHMMSS>-<alias>/
    ├── export.json                                  # runs, songs, seeds and settings
    └── My Song/
        ├── My Song.ogg
        ├── My Song (donderformer_base_oni_v2).tja   # TITLE: My Song (donderformer_base_oni_v2)
        └── My Song (taikonation_bs1_oni_v2).tja

Works for notebook 03 (``taikonation``) and notebook 04 (``donderformer``) runs. Only the run folder
(``config.json``, ``model.pt``) and the training split's normalization are needed, no feature cache.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from donderformer.data.audio import log_mel_features
from donderformer.data.feature_extractor import FeatureExtractor
from donderformer.data.tja_export import write_tja
from donderformer.exporter import NAME_PATTERN, folder_name
from donderformer.paths import REPO_ROOT, DataPaths, outputs_dir

LOW_BANDS = 20                 # mel bands treated as "kick" when choosing the downbeat


@dataclass
class SavedRun:
    run_dir: Path
    model: torch.nn.Module
    config: dict
    extractor: FeatureExtractor
    mean: np.ndarray
    std: np.ndarray

    @property
    def name(self) -> str:
        return self.config["model"]

    @property
    def label(self) -> str:
        """Default chart label, e.g. ``donderformer_base_oni_v2``."""
        parts = (self.config["model"], self.config.get("alias"), self.config["data"]["split"])
        return "_".join(p for p in parts if p)


def load_saved_run(run_dir, paths: DataPaths, device="cpu") -> SavedRun:
    """Model, config, extractor and train normalization of a notebook 03 or 04 run."""
    run_dir = Path(run_dir).expanduser()
    config = json.loads((run_dir / "config.json").read_text())
    data = config["data"]
    extractor = FeatureExtractor(**data["extractor"]) if "extractor" in data else FeatureExtractor(frame_ms=23)
    if extractor.id != data["features"]:
        raise ValueError(f"{run_dir}: config has no data.extractor and features={data['features']!r}; "
                         "add the extractor settings to config.json")
    if config["model"] == "donderformer":
        from donderformer.data.chart_dataset import WindowConfig
        from donderformer.models.donderformer import DonderFormer, DonderFormerConfig
        model = DonderFormer(DonderFormerConfig(**config["model_config"]), WindowConfig(**config["window"]))
    elif config["model"] == "taikonation":
        from donderformer.data.taiko_dataset import Window
        from donderformer.models.taikonation import TaikoNationBaseline
        if "window" not in data:
            raise ValueError(f"{run_dir}: config lacks data.window (a run from before the window redesign)")
        model = TaikoNationBaseline(Window(**data["window"]), dropout=config.get("dropout", 0.2))
    else:
        raise ValueError(f"{run_dir}: unknown model {config['model']!r}")
    checkpoint = torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    mean, std = extractor.load_normalization(paths, data["split"])
    return SavedRun(run_dir, model, config, extractor, mean, std)


def song_mel(run: SavedRun, audio_path) -> np.ndarray:
    """Normalized ``(n_frames, n_mels)`` log-mel of one audio file, as the run was trained on."""
    e = run.extractor
    mel = log_mel_features(Path(audio_path), e.frame_ms, e.n_mels, e.sample_rate)
    return ((mel - run.mean) / (run.std + 1e-8)).astype(np.float32)


def lstm_windows(mel: np.ndarray, window) -> np.ndarray:
    """``(n_frames - targets + 1, n_inputs)`` audio windows with an empty note history.
    ``generate_autoregressive`` replaces the history with its own notes."""
    pad = window.past_frames
    end_pad = max(window.look_forward - (window.targets - 1), 0)
    padded = np.vstack([np.zeros((pad, mel.shape[1]), np.float32), mel,
                        np.zeros((end_pad, mel.shape[1]), np.float32)])
    n = max(len(mel) - (window.targets - 1), 0)
    audio = padded[np.arange(n)[:, None] + np.arange(window.audio_frames)].reshape(n, -1)
    return np.hstack([audio, np.zeros((n, window.history * 7), np.float32)])


def onset_envelope(mel: np.ndarray) -> np.ndarray:
    """Per-frame onset strength: positive spectral flux minus its ~0.5 s local mean, unit std."""
    flux = np.maximum(np.diff(mel, axis=0, prepend=mel[:1]), 0).sum(axis=1)
    env = np.maximum(flux - np.convolve(flux, np.ones(21) / 21, mode="same"), 0)
    return env / (env.std() + 1e-8)


def estimate_bpm_offset(mel: np.ndarray, frame_ms: float = 23, bpm: float | None = None,
                        lo: float = 60.0, hi: float = 240.0, prefer: float = 160.0) -> tuple[float, float]:
    """(bpm, TJA offset) from an unnormalized log-mel, by fitting a constant beat grid to onsets.

    Every BPM in ``[lo, hi)`` (0.05 steps; pass ``bpm`` to keep it fixed) and every phase is
    scored by the mean onset strength on its beats, weighted toward ``prefer`` to settle
    half/double-tempo ties; a round x.0 / x.5 tempo wins when it scores within 3%. The bar
    starts on the beat (mod 4) with the strongest low-band onsets. On the 12 toy songs this
    found 11 BPMs exactly (one at double tempo) and 10 beat grids within 10 ms; bar lines
    landed on the right beat about 60% of the time. Constant tempo only.
    """
    env = onset_envelope(mel)
    fps, idx = 1000.0 / frame_ms, np.arange(len(env))
    candidates = [float(bpm)] if bpm else np.arange(lo, hi, 0.05)
    scored = []
    for b in candidates:
        period = fps * 60.0 / b
        phases = np.arange(0, period, 0.25)
        beats = phases[:, None] + np.arange(int((len(env) - 1) / period))[None, :] * period
        s = np.interp(beats, idx, env).mean(axis=1)
        j = int(s.argmax())
        prior = np.exp(-0.5 * np.log2(b / prefer) ** 2)
        scored.append((s[j] * prior, float(b), float(phases[j])))
    score, best, phase = max(scored)
    rounded = round(best * 2) / 2
    for cand in scored:
        if abs(cand[1] - rounded) < 1e-6 and cand[0] >= 0.97 * score:
            score, best, phase = cand
    period = fps * 60.0 / best
    beats = phase + np.arange(int((len(env) - 1) / period)) * period
    kick = np.interp(beats, idx, onset_envelope(mel[:, :LOW_BANDS]))
    downbeat = int(np.argmax([kick[i::4].mean() for i in range(4)]))
    return round(best, 2), round(-(phase + downbeat * period) / fps, 3)


def generate_song(run: SavedRun, mel: np.ndarray, level=None, course="oni", seed=0) -> np.ndarray:
    """``(n_frames,)`` class ids. DonderFormer uses the run's saved sampling, level and course;
    the CNN-LSTM has no conditioning and ignores them."""
    if run.name == "donderformer":
        from donderformer.models.donderformer import generate_chart
        return generate_chart(run.model, mel, course, level, seed=seed, **run.config["hparams"]["sampling"])
    from donderformer.evaluation import generate_autoregressive
    device = next(run.model.parameters()).device
    return generate_autoregressive(run.model, lstm_windows(mel, run.model.window), device, seed=seed)


def export_songs(songs: list[dict], runs: dict[str, SavedRun], alias: str, version: str = "clean",
                 seed: int = 0, snap_frames: int = 1) -> Path:
    """Chart every song with every run into a new ``outputs/exports/<datetime>-<alias>/`` folder.

    ``songs``: dicts with ``audio`` (path; relative paths are from the repo root) and optional ``title`` (default: file name), ``bpm`` and
    ``offset`` (numbers, or ``"auto"`` to estimate them from the audio; default ``"auto"``), and
    ``level`` (star level for DonderFormer, default unknown).
    ``runs``: chart label -> loaded run; the label goes in each .tja's file name and TITLE.
    ``snap_frames``: notes may move up to this many 23 ms frames onto a simpler beat position.
    """
    from donderformer.evaluation import remove_adjacent_notes

    for name in (alias, *runs):
        if not NAME_PATTERN.fullmatch(name):
            raise ValueError(f"{name!r}: aliases and labels may only use letters, digits, '_' and '.'")
    if version not in ("clean", "raw"):
        raise ValueError("version must be 'clean' or 'raw'")
    out = outputs_dir() / "exports" / f"{datetime.now():%Y%m%d_%H%M%S}-{alias}"
    out.mkdir(parents=True, exist_ok=False)
    used, items = set(), []

    for song in songs:
        audio = Path(song["audio"]).expanduser()
        if not audio.is_absolute():
            audio = REPO_ROOT / audio         # not the notebook's working directory
        title = song.get("title") or audio.stem
        bpm, offset, level = song.get("bpm") or "auto", song.get("offset", "auto"), song.get("level")
        offset = "auto" if offset is None else offset
        if "auto" in (bpm, offset):
            frame_ms = next(iter(runs.values())).extractor.frame_ms
            est_bpm, est_offset = estimate_bpm_offset(log_mel_features(audio, frame_ms), frame_ms,
                                                      bpm=None if bpm == "auto" else float(bpm))
            print(f"{title}: estimated" + (f" bpm {est_bpm}" if bpm == "auto" else "")
                  + (f" offset {est_offset}" if offset == "auto" else ""), flush=True)
            bpm = est_bpm if bpm == "auto" else bpm
            offset = est_offset if offset == "auto" else offset
        bpm, offset = float(bpm), float(offset)
        name = folder_name(title)
        while name in used:
            name += "_"
        used.add(name)
        song_dir = out / name
        song_dir.mkdir()
        shutil.copy2(audio, song_dir / audio.name)

        files, mels = {}, {}
        for label, run in runs.items():
            key = run.extractor
            if key not in mels:
                mels[key] = song_mel(run, audio)
            classes = generate_song(run, mels[key], level=level, seed=seed)
            if version == "clean":
                classes = remove_adjacent_notes(classes)
            tja = write_tja(song_dir / f"{name} ({label}).tja", classes, frame_ms=run.extractor.frame_ms,
                            bpm=bpm, offset=offset, title=f"{title} ({label})", wave=audio.name,
                            course="Oni", level=level, snap_frames=snap_frames)
            files[label] = tja.name
            print(f"{title} ({label}): {int((classes != 0).sum())} note frames", flush=True)
        items.append({"title": title, "audio": str(audio), "folder": name, "bpm": bpm, "offset": offset,
                      "level": level, "charts": files})

    (out / "export.json").write_text(json.dumps({
        "alias": alias, "created": datetime.now().isoformat(timespec="seconds"), "version": version,
        "seed": seed, "snap_frames": snap_frames, "runs": {label: str(run.run_dir) for label, run in runs.items()}, "items": items,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(items)} songs x {len(runs)} charts to {out}")
    return out
