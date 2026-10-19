"""TaikoNation's CNN-LSTM baseline.

Input: one flattened window of 16 audio frames (16 x 80 mel) followed by the
previous 15 note frames (15 x 7 one-hot), 1385 values in total.
Output: logits for the next 4 frames, shape ``(batch, 4, 7)``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TaikoNationBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        # Split the 1385 inputs; split them back into audio and notes
        self.conv1 = nn.Conv1d(80, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)

        self.audio_fc = nn.Linear(32 * 4, 128)

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

        self.output = nn.Linear(64, 28)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.8)

    # send the audio through the CNN to find useful patters in the audio features
    def forward(self, x):
        audio = x[:, :1280].reshape(-1, 16, 80)
        notes = x[:, 1280:].reshape(-1, 15, 7)

        audio = audio.transpose(1, 2)

        audio = self.relu(self.conv1(audio))
        audio = self.dropout(audio)
        audio = self.pool1(audio)

        audio = self.relu(self.conv2(audio))
        audio = self.pool2(audio)

        audio = audio.flatten(1)
        audio = self.relu(self.audio_fc(audio))
        audio = audio.reshape(-1, 16, 8)

        note_bias = torch.ones(
            notes.shape[0], 15, 1,
            device=notes.device,
            dtype=notes.dtype
        )

        note_history = torch.cat([notes, note_bias], dim=2)

        current_placeholder = torch.ones(
            notes.shape[0], 1, 8,
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

        return logits.reshape(-1, 4, 7)
