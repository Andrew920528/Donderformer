"""TaikoDataset: TaikoNation-style windows cut on the fly from cached features.

For frame ``i`` of a chart, one sample is

    X = [mel frames i-A+1+L .. i+L  (A x n_mels),  label frames i-H .. i-1  (H x 7)]  flattened
    Y = label frames i .. i+T-1  (T x 7)

with A = ``audio_frames``, H = ``history_frames``, T = ``target_frames``,
L = ``audio_lookahead`` and zero padding before the start (and after the end when
L > T-1). With L = T-1 the audio covers every target frame; L = 0 is the original
window, whose audio ends at the first target. The defaults (16, 15, 4, 0) give
TaikoNation's 1385 inputs; (16, 12, 4, 3) is the aligned layout of the original
TaikoNation code (12 past notes + 4 placeholder rows, audio i-12 .. i+3).
Changing them needs no re-extraction.
"""
from __future__ import annotations

import bisect

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from donderformer.data.feature_extractor import FeatureExtractor
from donderformer.data.data_splitter import read_manifest
from donderformer.paths import DataPaths


class TaikoDataset(Dataset):
    def __init__(self, paths: DataPaths, split: str, extractor: FeatureExtractor, part: str = "train",
                 audio_frames: int = 16, history_frames: int = 15, target_frames: int = 4,
                 audio_lookahead: int = 0):
        if audio_lookahead < 0 or audio_lookahead >= audio_frames:
            raise ValueError(f"audio_lookahead must be in [0, audio_frames), got {audio_lookahead}")
        self.paths, self.split, self.extractor, self.part = paths, split, extractor, part
        self.audio_frames, self.history_frames, self.target_frames = audio_frames, history_frames, target_frames
        self.audio_lookahead = audio_lookahead

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
            self.lengths.append(max(n_frames - (target_frames - 1), 0))
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
        pad = max(self.audio_frames - 1 - self.audio_lookahead, self.history_frames, 0)
        # The last window's audio reaches audio_lookahead frames past its anchor, its targets
        # target_frames - 1; pad the mel end for any lookahead beyond the targets.
        end_pad = max(self.audio_lookahead - (self.target_frames - 1), 0)
        mel = np.vstack([np.zeros((pad, mel.shape[1]), np.float32), mel,
                         np.zeros((end_pad, mel.shape[1]), np.float32)])
        labels = np.vstack([np.zeros((pad, 7), np.float32), labels.astype(np.float32)])
        return mel, labels, pad

    def _window(self, mel, labels, pad, frames):
        A, H, T, L = self.audio_frames, self.history_frames, self.target_frames, self.audio_lookahead
        i = frames[:, None] + pad
        audio = mel[i - A + 1 + L + np.arange(A)]
        history = labels[i - H + np.arange(H)]
        targets = labels[i + np.arange(T)]
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
