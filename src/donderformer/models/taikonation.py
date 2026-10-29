"""TaikoNation's CNN-LSTM baseline.

Input: ``(batch, window.n_inputs)``, a flattened ``TaikoDataset`` window.
Output: logits ``(batch, window.targets, 7)``.

The LSTM runs over the A audio rows, one per frame. Rows i-H .. i-1 are multiplied
by that frame's note (one-hot + 1); every other row (older than the history, or
frame i onward) by ones.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from donderformer.data.taiko_dataset import N_CLASSES, N_MELS, Window


class TaikoNationBaseline(nn.Module):
    def __init__(self, window: Window = Window(), dropout: float = 0.2):
        super().__init__()
        self.window = window
        self.audio_frames = window.audio_frames
        self.n_inputs = window.n_inputs

        self.conv1 = nn.Conv1d(N_MELS, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        # (batch, A, 8): one row per LSTM step
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

        self.output = nn.Linear(64, window.targets * N_CLASSES)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    # send the audio through the CNN to find useful patters in the audio features
    def forward(self, x):
        A, H = self.audio_frames, self.window.history
        if x.shape[1] != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} inputs ({A} audio + {H} note frames), got {x.shape[1]}")
        audio = x[:, :A * N_MELS].reshape(-1, A, N_MELS)
        notes = x[:, A * N_MELS:].reshape(x.shape[0], H, N_CLASSES)

        audio = audio.transpose(1, 2)

        audio = self.relu(self.conv1(audio))
        audio = self.dropout(audio)
        audio = self.pool1(audio)

        audio = self.relu(self.conv2(audio))
        audio = self.pool2(audio)

        audio = audio.flatten(1)
        audio = self.relu(self.audio_fc(audio))
        audio = audio.reshape(-1, A, 8)

        ones = lambda rows, width: torch.ones(notes.shape[0], rows, width, device=notes.device, dtype=notes.dtype)
        note_history = torch.cat([notes, ones(H, 1)], dim=2)
        note_context = torch.cat(
            [ones(self.window.past_frames - H, 8), note_history, ones(1 + self.window.look_forward, 8)],
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

        return logits.reshape(-1, self.window.targets, N_CLASSES)
