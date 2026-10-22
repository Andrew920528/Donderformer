"""TaikoNation-style windows cut on the fly from cached features.

For frame i, with A = audio_frames, H = history, F = look_forward, T = targets
and P = A - 1 - F past audio frames:

    X = [mel i-P .. i+F  (A x 80),  labels i-H .. i-1  (H x 7)]  flattened
    Y = labels i .. i+T-1  (T x 7)

Out-of-range frames are zero-padded.
"""
from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from donderformer.data.feature_extractor import FeatureExtractor
from donderformer.data.data_splitter import read_manifest
from donderformer.paths import DataPaths


N_MELS = 80
N_CLASSES = 7


@dataclass(frozen=True)
class Window:
    """Input layout shared by ``TaikoDataset`` and ``TaikoNationBaseline``.

    The default is the TaikoNation paper's: 16 audio frames ending at frame i,
    the 15 previous notes, and the next 4 frames predicted.
    """
    audio_frames: int = 16   # audio frames per window, ending at i + look_forward
    history: int = 15        # previous note frames, i-H .. i-1
    look_forward: int = 0    # audio frames after frame i
    targets: int = 4         # predicted frames, i .. i+T-1

    def __post_init__(self):
        if min(self.history, self.look_forward) < 0 or self.targets < 1:
            raise ValueError(f"invalid window {self}")
        if self.audio_frames < 4:
            raise ValueError("audio_frames must be >= 4 for the model's two pooling layers")
        if self.history > self.past_frames:
            raise ValueError(f"history={self.history} reaches before the audio "
                             f"(audio_frames - 1 - look_forward = {self.past_frames})")

    @property
    def past_frames(self) -> int:
        """Audio frames before frame i."""
        return self.audio_frames - 1 - self.look_forward

    @property
    def n_inputs(self) -> int:
        return self.audio_frames * N_MELS + self.history * N_CLASSES

    @property
    def name(self) -> str:
        return f"a{self.audio_frames}_h{self.history}_f{self.look_forward}"

    def describe(self) -> str:
        P, H, F, T = self.past_frames, self.history, self.look_forward, self.targets
        span = lambda a, b: "i" + (f"{a:+d}" if a else "") + " .. i" + (f"{b:+d}" if b else "")
        return (f"audio {span(-P, F)}, notes {span(-H, -1) if H else 'none'}, predicts {span(0, T - 1)}"
                + ("" if F >= T - 1 else f" ({T - 1 - F} predicted frames have no audio)"))

    def to_dict(self) -> dict:
        return asdict(self)


class TaikoDataset(Dataset):
    def __init__(self, paths: DataPaths, split: str, extractor: FeatureExtractor, part: str = "train",
                 window: Window = Window()):
        self.paths, self.split, self.extractor, self.part = paths, split, extractor, part
        self.window = window

        rows = read_manifest(paths, split, part)
        failed = extractor.failed_charts(paths, split)
        self.charts = [row for row in rows if row["chart_id"] not in failed]
        self.skipped = len(rows) - len(self.charts)
        if self.skipped:
            print(f"{part}: skipping {self.skipped} charts that couldn't be labelled")
        self.chart_ids = [row["chart_id"] for row in self.charts]
        self.mean, self.std = extractor.load_normalization(paths, split)

        self.lengths = []
        for row in self.charts:
            n_frames = len(np.load(extractor.mel_path(paths, row["audio_id"]), mmap_mode="r"))
            self.lengths.append(max(n_frames - (window.targets - 1), 0))
        self.ends = np.cumsum(self.lengths).tolist()
        self._cached_chart = None
        self._cached_arrays = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        chart_index = bisect.bisect_right(self.ends, index)
        local_index = index - (self.ends[chart_index - 1] if chart_index else 0)
        if self._cached_chart != chart_index:
            self._cached_arrays = self.chart_arrays(chart_index)
            self._cached_chart = chart_index
        X, Y = self._window(*self._cached_arrays, np.array([local_index]))
        return torch.from_numpy(X[0]), torch.from_numpy(Y[0])

    def chart_arrays(self, chart_index):
        """Normalized mel and labels of one chart, zero-padded for windowing; returns the front pad."""
        row = self.charts[chart_index]
        mel = np.load(self.extractor.mel_path(self.paths, row["audio_id"]))
        labels = np.load(self.extractor.labels_path(self.paths, row["audio_id"], row["chart_id"]))
        mel = ((mel - self.mean) / (self.std + 1e-8)).astype(np.float32)
        w = self.window
        pad = w.past_frames
        # audio may reach past the last target frame
        end_pad = max(w.look_forward - (w.targets - 1), 0)
        mel = np.vstack([np.zeros((pad, mel.shape[1]), np.float32), mel,
                         np.zeros((end_pad, mel.shape[1]), np.float32)])
        labels = np.vstack([np.zeros((pad, N_CLASSES), np.float32), labels.astype(np.float32)])
        return mel, labels, pad

    def _window(self, mel, labels, pad, frames):
        w = self.window
        i = frames[:, None] + pad
        audio = mel[i - w.past_frames + np.arange(w.audio_frames)]
        history = labels[i - w.history + np.arange(w.history)]
        targets = labels[i + np.arange(w.targets)]
        X = np.concatenate([audio.reshape(len(frames), -1), history.reshape(len(frames), -1)], axis=1)
        return X, targets

    def chart_windows(self, chart_id):
        """Every window of one chart as float32 ``(X, Y)``, e.g. for evaluating a whole song."""
        chart_index = self.chart_ids.index(chart_id)
        return self._window(*self.chart_arrays(chart_index), np.arange(self.lengths[chart_index]))


class SongBatchSampler(Sampler):
    """Batches never cross chart boundaries; charts (and windows) can be shuffled."""

    def __init__(self, dataset: TaikoDataset, batch_size: int, shuffle: bool = False):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle

    def __iter__(self):
        song_order = np.arange(len(self.dataset.lengths))
        if self.shuffle:
            np.random.shuffle(song_order)
        for s in song_order:
            start = self.dataset.ends[s - 1] if s else 0
            indices = np.arange(start, self.dataset.ends[s])
            if self.shuffle:
                np.random.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                yield indices[offset:offset + self.batch_size].tolist()

    def __len__(self):
        return sum((n + self.batch_size - 1) // self.batch_size
                   for n in self.dataset.lengths)
