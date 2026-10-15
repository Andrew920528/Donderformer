"""FeatureExtractor: turn a split's charts into cached arrays a model can load.

    features/<extractor.id>/
    ├── config.json
    ├── songs/<audio_id>/
    │   ├── mel.npy                    # (n_frames, n_mels) log-mel, shared by every chart of the song
    │   └── labels/<chart_id>.npy      # (n_frames, 7) one-hot
    ├── normalization/<split>.npz      # per-band mean/std from that split's train songs + manifest hash
    └── failures/<split>.json          # charts that couldn't be labelled, if any

Extraction is deterministic and only computes missing files, so splits share the cache.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from donderformer.data.audio import N_MELS, SR, events_to_labels, log_mel_features
from donderformer.data.data_splitter import manifest_hash, read_manifest
from donderformer.data.tja import course_sections, parse_events
from donderformer.paths import DataPaths


@dataclass(frozen=True)
class FeatureExtractor:
    frame_ms: int = 23
    n_mels: int = N_MELS
    sample_rate: int = SR

    @property
    def id(self) -> str:
        """Folder name; only non-default settings beyond frame length are spelled out."""
        name = f"frame{self.frame_ms}ms"
        if self.n_mels != N_MELS:
            name += f"_mel{self.n_mels}"
        if self.sample_rate != SR:
            name += f"_sr{self.sample_rate}"
        return name

    def dir(self, paths: DataPaths) -> Path:
        return paths.feature_dir(self.id)

    def mel_path(self, paths, audio_id) -> Path:
        return self.dir(paths) / "songs" / audio_id / "mel.npy"

    def labels_path(self, paths, audio_id, chart_id) -> Path:
        return self.dir(paths) / "songs" / audio_id / "labels" / f"{chart_id}.npy"

    def normalization_path(self, paths, split) -> Path:
        return self.dir(paths) / "normalization" / f"{split}.npz"

    def _check_config(self, paths):
        path = self.dir(paths) / "config.json"
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != asdict(self):
                raise ValueError(f"{path} was written with {existing}, not {asdict(self)}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(self), indent=2))

    def extract(self, paths: DataPaths, split: str, overwrite: bool = False) -> dict:
        """Cache mel + labels for every chart in ``split``, then fit its normalization."""
        self._check_config(paths)
        rows = read_manifest(paths, split)
        computed = {"mel": 0, "labels": 0}
        failures = []

        for i, row in enumerate(rows, 1):
            mel_path = self.mel_path(paths, row["audio_id"])
            if overwrite or not mel_path.exists():
                mel = log_mel_features(paths.raw / row["audio"], self.frame_ms, self.n_mels, self.sample_rate)
                if not np.isfinite(mel).all():
                    raise ValueError(f"{row['audio']}: non-finite audio features")
                mel_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(mel_path, mel)
                computed["mel"] += 1
            else:
                mel = np.load(mel_path, mmap_mode="r")

            labels_path = self.labels_path(paths, row["audio_id"], row["chart_id"])
            if overwrite or not labels_path.exists():
                try:
                    labels = self.chart_labels(paths, row, len(mel))
                except ValueError as e:
                    failures.append({"chart_id": row["chart_id"], "tja": row["tja"], "error": str(e)})
                    continue
                labels_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(labels_path, labels)
                computed["labels"] += 1

            if i % 100 == 0:
                print(f"{i}/{len(rows)} charts")

        mean, std = self.fit_normalization(paths, rows)
        norm_path = self.normalization_path(paths, split)
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(norm_path, mean=mean, std=std, manifest_sha256=manifest_hash(paths, split))

        failure_path = self.dir(paths) / "failures" / f"{split}.json"
        if failures:
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False))
        elif failure_path.exists():
            failure_path.unlink()

        summary = {"charts": len(rows), "computed": computed, "failures": len(failures)}
        print(f"{self.id} / {split}: {summary}")
        if failures:
            raise RuntimeError(
                f"{len(failures)} charts couldn't be labelled; see {failure_path}. "
                "Everything else is cached."
            )
        return summary

    def chart_labels(self, paths, row, n_frames):
        text = (paths.raw / row["tja"]).read_text(encoding="utf-8-sig")
        sections = [s for s in course_sections(text) if s.course == row["course"]]
        if len(sections) != 1:
            raise ValueError(f"expected one {row['course']} section, found {len(sections)}")
        events = parse_events(sections[0].lines, float(row["bpm"]), float(row["offset"]), where=row["tja"])
        labels = events_to_labels(events, n_frames, self.frame_ms)
        if not np.all(labels.sum(axis=1) == 1):
            raise ValueError("invalid one-hot labels")
        return labels

    def fit_normalization(self, paths, rows):
        """Per-band mean/std over every frame of the split's train songs (each audio once)."""
        train_audio = sorted({r["audio_id"] for r in rows if r["split"] == "train"})
        if not train_audio:
            raise ValueError("split has no train charts to fit normalization on")
        frames = np.concatenate([np.load(self.mel_path(paths, a)) for a in train_audio], axis=0)
        return frames.mean(axis=0), np.maximum(frames.std(axis=0), 1e-8)

    def load_normalization(self, paths, split):
        path = self.normalization_path(paths, split)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run {self!r}.extract(paths, {split!r}) first")
        with np.load(path) as d:
            if str(d["manifest_sha256"]) != manifest_hash(paths, split):
                raise ValueError(
                    f"splits/{split}/manifest.csv changed after features were extracted. "
                    "Splits should not be edited in place; re-run extract() if this was intended."
                )
            return d["mean"], d["std"]
