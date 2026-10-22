"""TaikoNation's CNN-LSTM baseline.

Input: one flattened window of A audio frames (A x 80 mel) followed by the previous
H note frames (H x 7 one-hot), with A = context_length + 1 + look_forward and
H = context_length (see ``TaikoDataset``: ``audio_frames``, ``history_frames``,
``audio_lookahead``).
Output: logits for the next 4 frames, shape ``(batch, 4, 7)``.

The LSTM runs over A rows. Row k pairs the audio features with note row k: the first
H rows carry the past notes, the remaining 1 + look_forward rows are all-ones
placeholders for the frames being predicted (and any extra lookahead).

- ``TaikoNationBaseline()`` is the original port: 16 audio frames ending at the first
  target, 15 past notes and 1 placeholder (1385 inputs). Old checkpoints load into it.
- ``TaikoNationBaseline(context_length=12, look_forward=3)`` is the layout of the
  original TaikoNation code: audio i-12 .. i+3, 12 past notes, 4 placeholders (1364 inputs).

``dropout`` is the probability of *dropping* an activation (PyTorch). TaikoNation's
tflearn code used keep_prob=0.8, i.e. dropout=0.2; the first port used 0.8 by mistake.
Dropout has no weights, so it doesn't affect loading a checkpoint.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_MELS = 80
N_CLASSES = 7
TARGET_FRAMES = 4


class TaikoNationBaseline(nn.Module):
    def __init__(self, context_length: int = 15, look_forward: int = 0, dropout: float = 0.2):
        super().__init__()
        self.context_length, self.look_forward = context_length, look_forward
        self.audio_frames = context_length + 1 + look_forward
        if self.audio_frames < 4:
            raise ValueError("need at least 4 audio frames for the two pooling layers")
        self.n_inputs = self.audio_frames * N_MELS + context_length * N_CLASSES

        self.conv1 = nn.Conv1d(N_MELS, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        # One 8-wide row per LSTM step (A rows); 128 -> 128 for the original 16 frames.
        self.audio_fc = nn.Linear(32 * (self.audio_frames // 4), 8 * self.audio_frames)

        self.lstm1 = nn.LSTM(
            input_size=8,
            hidden_size=64,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=64,
            hidden_size=64,
            batch_first=True
        )

        self.output = nn.Linear(64, TARGET_FRAMES * N_CLASSES)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # send the audio through the CNN to find useful patters in the audio features
    def forward(self, x):
        A, H = self.audio_frames, self.context_length
        if x.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs ({A} audio + {H} note frames), got {x.shape[1]}")
        audio = x[:, :A * N_MELS].reshape(-1, A, N_MELS)
        notes = x[:, A * N_MELS:].reshape(-1, H, N_CLASSES)

        audio = audio.transpose(1, 2)

        audio = self.relu(self.conv1(audio))
        audio = self.dropout(audio)
        audio = self.pool1(audio)

        audio = self.relu(self.conv2(audio))
        audio = self.pool2(audio)

        audio = audio.flatten(1)
        audio = self.relu(self.audio_fc(audio))
        audio = audio.reshape(-1, A, 8)

        note_bias = torch.ones(
            notes.shape[0], H, 1,
            device=notes.device,
            dtype=notes.dtype
        )

        note_history = torch.cat([notes, note_bias], dim=2)

        # All-ones rows where the predicted frames (and any extra lookahead) sit.
        current_placeholder = torch.ones(
            notes.shape[0], A - H, 8,
            device=notes.device,
            dtype=notes.dtype
        )

        note_context = torch.cat(
            [note_history, current_placeholder],
            dim=1
        )

        # Combine audio and previous-note information
        combined = audio * note_context

        combined, _ = self.lstm1(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)

        # LSTM learns the temporal pattern
        combined, _ = self.lstm2(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)

        final = combined[:, -1, :]
        logits = self.output(final)

        return logits.reshape(-1, TARGET_FRAMES, N_CLASSES)

    def window_kwargs(self) -> dict:
        """``TaikoDataset`` arguments that produce this model's input layout."""
        return dict(audio_frames=self.audio_frames, history_frames=self.context_length,
                    target_frames=TARGET_FRAMES, audio_lookahead=self.look_forward)
