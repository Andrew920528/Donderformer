"""ChartWindowDataset: long audio windows + event-token sequences for the Transformer.

One sample is a window of ``past + target + lookahead`` frames starting at frame ``s``
(``s`` may be negative; frames outside the song are zero mel and carry no notes)::

    frames  s ............ s+past ........... s+past+target ...... s+window
            |  past: notes  |  target: notes   |  lookahead: audio  |
            |  as a prefix  |  to generate     |  only              |

    mel     (window, n_mels) float32, z-scored with the split's train stats
    tokens  BOS course level <past events> SEP <target events> EOS   (see tokens.py)

Training windows start at random frames (``resample(epoch)`` draws a new set, about one
window per ``target`` frames of each chart); evaluation windows tile each chart with
hop ``target``. All mel of the part is loaded into memory once (float16).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from donderformer.data.data_splitter import read_manifest
from donderformer.data.feature_extractor import FeatureExtractor
from donderformer.data.tja import course_sections
from donderformer.data.tokens import (ONSET_NOTES, Vocab, build_sequence, classes_to_events,
                                      window_events)
from donderformer.paths import DataPaths


@dataclass(frozen=True)
class WindowConfig:
    past_frames: int = 512          # 11.8 s of notes given as the prefix
    target_frames: int = 256        # 5.9 s generated per window (also the generation hop)
    lookahead_frames: int = 64      # 1.5 s of audio after the target
    max_tokens: int = 640           # decoder length; the oldest prefix events are trimmed to fit

    @property
    def window_frames(self) -> int:
        return self.past_frames + self.target_frames + self.lookahead_frames

    @property
    def n_time(self) -> int:
        return self.past_frames + self.target_frames

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Augment:
    """Train-only augmentation. Masked mel is set to 0 (the per-band mean after z-scoring)."""
    freq_masks: int = 2
    freq_mask_bands: int = 8
    time_masks: int = 2
    time_mask_frames: int = 10           # past region only, so target audio is never hidden
    level_dropout: float = 0.1           # replace the level token with LEVEL_UNK
    prefix_noise: float = 0.3            # share of windows whose prefix gets corrupted (exposure bias)
    prefix_delete: float = 0.1
    prefix_insert: float = 0.05
    prefix_jitter: float = 0.1


def chart_level(paths: DataPaths, row) -> int | None:
    """Star level from the manifest, else from the chart's own TJA (oni_v1 has no level column)."""
    if row.get("level"):
        return int(float(row["level"]))
    text = (paths.raw / row["tja"]).read_text(encoding="utf-8-sig", errors="replace")
    levels = [s.level for s in course_sections(text) if s.course == row["course"]]
    return levels[0] if len(levels) == 1 else None


class ChartWindowDataset(Dataset):
    def __init__(self, paths: DataPaths, split: str, extractor: FeatureExtractor, part: str = "train",
                 config: WindowConfig = WindowConfig(), augment: Augment | None = None, seed: int = 0):
        self.paths, self.split, self.extractor, self.part = paths, split, extractor, part
        self.config, self.augment, self.seed = config, augment, seed
        self.vocab = Vocab(n_time=config.n_time)
        self.rng = np.random.default_rng(seed)

        rows = read_manifest(paths, split, part)
        failed = extractor.failed_charts(paths, split)
        self.charts = [row for row in rows if row["chart_id"] not in failed]
        self.skipped = len(rows) - len(self.charts)
        if self.skipped:
            print(f"{part}: skipping {self.skipped} charts that couldn't be labelled")
        self.chart_ids = [row["chart_id"] for row in self.charts]
        mean, std = extractor.load_normalization(paths, split)

        self.mel = {}
        for row in self.charts:
            if row["audio_id"] not in self.mel:
                mel = np.load(extractor.mel_path(paths, row["audio_id"]))
                self.mel[row["audio_id"]] = ((mel - mean) / (std + 1e-8)).astype(np.float16)
        self.events, self.levels, self.lengths = [], [], []
        for row in self.charts:
            labels = np.load(extractor.labels_path(paths, row["audio_id"], row["chart_id"]))
            self.events.append(classes_to_events(labels.argmax(axis=1)))
            self.levels.append(chart_level(paths, row))
            self.lengths.append(len(self.mel[row["audio_id"]]))

        self.windows = []
        self.resample(0)

    # ---- which windows make up an epoch

    def starts(self, chart_index, random: bool):
        """Window starts so every frame of the chart is in some target region."""
        n, P, T = self.lengths[chart_index], self.config.past_frames, self.config.target_frames
        count = math.ceil(n / T)
        if random:
            return (self.rng.integers(-P, n - P, size=count)).tolist()
        return [-P + k * T for k in range(count)]

    def resample(self, epoch: int):
        """Training: a fresh random set of window starts; evaluation: the fixed tiling."""
        random = self.augment is not None
        if random:
            self.rng = np.random.default_rng([self.seed, epoch])
        self.windows = [(c, s) for c in range(len(self.charts)) for s in self.starts(c, random)]

    def __len__(self):
        return len(self.windows)

    # ---- one window

    def window_mel(self, audio_id, start):
        """``(window_frames, n_mels)`` float32 starting at frame ``start``, zero outside the song."""
        mel = self.mel[audio_id]
        W = self.config.window_frames
        out = np.zeros((W, mel.shape[1]), np.float32)
        lo, hi = max(start, 0), min(start + W, len(mel))
        if hi > lo:
            out[lo - start:hi - start] = mel[lo:hi]
        return out

    def __getitem__(self, index):
        chart_index, start = self.windows[index]
        return self.item(chart_index, start)

    def item(self, chart_index, start):
        row, cfg = self.charts[chart_index], self.config
        P, T = cfg.past_frames, cfg.target_frames
        mel = self.window_mel(row["audio_id"], start)
        events = self.events[chart_index]
        past = window_events(events, start, start + P)
        target = [(f + P, note) for f, note in window_events(events, start + P, start + P + T)]
        # A long note open across the boundary already appears in past; its continuation at
        # TIME_P in target would read as a second note.
        if target and past and target[0][0] == P and target[0][1] != "END" and _open_at_end(past):
            target = target[1:]
        level = self.levels[chart_index]

        if self.augment is not None:
            mel = self.spec_augment(mel)
            if self.rng.random() < self.augment.level_dropout:
                level = None
            if self.rng.random() < self.augment.prefix_noise:
                past = self.corrupt(past)

        tokens, sep = build_sequence(self.vocab, row["course"], level, past, target, cfg.max_tokens)
        return torch.from_numpy(mel), torch.tensor(tokens, dtype=torch.long), sep

    def spec_augment(self, mel):
        a, mel = self.augment, mel.copy()
        for _ in range(a.freq_masks):
            width = int(self.rng.integers(0, a.freq_mask_bands + 1))
            lo = int(self.rng.integers(0, mel.shape[1] - width + 1))
            mel[:, lo:lo + width] = 0
        for _ in range(a.time_masks):
            width = int(self.rng.integers(0, a.time_mask_frames + 1))
            lo = int(self.rng.integers(0, self.config.past_frames - width + 1))
            mel[lo:lo + width] = 0
        return mel

    def corrupt(self, past):
        """Delete, jitter and insert onset events in the prefix; long notes are left alone."""
        a, P = self.augment, self.config.past_frames
        out, taken = [], {f for f, _ in past}
        for frame, note in past:
            if note in ONSET_NOTES:
                if self.rng.random() < a.prefix_delete:
                    continue
                if self.rng.random() < a.prefix_jitter:
                    moved = frame + int(self.rng.choice([-1, 1]))
                    if 0 <= moved < P and moved not in taken:
                        taken.discard(frame)
                        taken.add(moved)
                        frame = moved
            out.append((frame, note))
        for _ in range(self.rng.binomial(len(past), a.prefix_insert) if past else 0):
            frame = int(self.rng.integers(0, P))
            if frame not in taken and not _inside_long(out, frame):
                taken.add(frame)
                out.append((frame, str(self.rng.choice(["don", "ka"]))))
        out.sort(key=lambda e: (e[0], e[1] == "END"))
        return out


def _open_at_end(events) -> bool:
    open_ = False
    for _, note in events:
        if note in ("ROLL", "BALLOON"):
            open_ = True
        elif note == "END":
            open_ = False
    return open_


def _inside_long(events, frame) -> bool:
    start = None
    for f, note in sorted(events):
        if note in ("ROLL", "BALLOON"):
            start = f
        elif note == "END" and start is not None:
            if start <= frame <= f:
                return True
            start = None
    return start is not None and frame >= start


def collate(batch, pad_id: int = Vocab.PAD):
    """Stack mel; right-pad tokens with PAD. Returns ``mel, tokens, sep`` (sep: index of SEP per row)."""
    mels, seqs, seps = zip(*batch)
    tokens = torch.full((len(seqs), max(len(s) for s in seqs)), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        tokens[i, :len(seq)] = seq
    return torch.stack(mels), tokens, torch.tensor(seps)
