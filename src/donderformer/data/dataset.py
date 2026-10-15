"""PyTorch dataset over preprocessed TaikoNation-style windows."""
from __future__ import annotations

import bisect
import json

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from donderformer.paths import DataPaths


class TaikoDataset(Dataset):
    """Windows from ``processed/<run_name>/normalized/<song_id>/windows.npz``.

    Songs are loaded one at a time rather than concatenated into RAM; pair it
    with ``SongBatchSampler`` so batches don't reopen large npz files.
    """

    def __init__(self, paths: DataPaths, run_name: str, split: str = "train"):
        self.run_dir = paths.processed(run_name)
        self.window_dir = self.run_dir / "normalized"
        self.split = json.loads((self.run_dir / "split.json").read_text())
        self.song_ids = [str(x) for x in self.split[split]]
        self.lengths = []
        for song_id in self.song_ids:
            with np.load(self.window_dir / song_id / "windows.npz") as d:
                self.lengths.append(len(d["X"]))
        self.ends = np.cumsum(self.lengths).tolist()
        self._cached_song = None
        self._cached_arrays = None

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        song_index = bisect.bisect_right(self.ends, index)
        local_index = index - (self.ends[song_index - 1] if song_index else 0)
        if self._cached_song != song_index:
            self._cached_arrays = self.load_song(self.song_ids[song_index])
            self._cached_song = song_index
        X, Y = self._cached_arrays
        return torch.from_numpy(X[local_index]), torch.from_numpy(Y[local_index])

    def load_song(self, song_id: str):
        """All windows of one song as float32 ``(X, Y)``."""
        with np.load(self.window_dir / song_id / "windows.npz") as d:
            return d["X"].astype(np.float32), d["Y"].astype(np.float32)


class SongBatchSampler(Sampler):
    """Batches never cross song boundaries; songs (and windows) can be shuffled."""

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
