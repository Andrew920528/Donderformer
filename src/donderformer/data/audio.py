"""Log-mel features, TaikoNation style: one 80-band frame per fixed-length audio chunk.

A numpy reimplementation of Essentia's Windowing (Blackman-Harris 62 dB),
Spectrum and MelBands, checked against essentia==2.1b6 to about 1e-4.
"""
from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache

import numpy as np

SR = 44100
N_MELS = 80
MEL_LO, MEL_HI = 27.5, 16000.0
LOG_EPS = 1e-16


@lru_cache(maxsize=1)
def ffmpeg_exe():
    """System ffmpeg if installed, else the binary bundled with imageio-ffmpeg (PACE has no ffmpeg)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError("ffmpeg not found; install it or `uv add imageio-ffmpeg`") from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def load_mono_audio(path, sr=SR):
    result = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def make_chunks(audio, frame_ms, sr=SR):
    duration_ms = len(audio) * 1000 / sr
    count = int(duration_ms // frame_ms)

    return [
        audio[
            int(k * frame_ms * sr / 1000):
            int((k + 1) * frame_ms * sr / 1000)
        ]
        for k in range(count)
    ]


def hz_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


@lru_cache(maxsize=None)
def _window(n):
    phase = 2 * np.pi * np.arange(n) / (n - 1)
    window = .44959 - .49364 * np.cos(phase) + .05677 * np.cos(2 * phase)
    return window * 2 / window.sum()


@lru_cache(maxsize=None)
def _filterbank(bins, n_mels, sr):
    edges = np.linspace(hz_to_mel(MEL_LO), hz_to_mel(MEL_HI), n_mels + 2)
    bin_mels = hz_to_mel(np.arange(bins) * sr / 2 / (bins - 1))
    bank = np.zeros((n_mels, bins))
    for i in range(n_mels):
        left, center, right = edges[i:i + 3]
        bank[i] = np.clip(np.minimum((bin_mels - left) / (center - left),
                                     (right - bin_mels) / (right - center)), 0, None)
        bank[i] /= bank[i].sum()
    return bank


def log_mel_features(audio_path, frame_ms, n_mels=N_MELS, sr=SR):
    """``(n_frames, n_mels)`` float32, one row per ``frame_ms`` chunk."""
    output = []
    for chunk in make_chunks(load_mono_audio(audio_path, sr), frame_ms, sr):
        if len(chunk) < 4:
            output.append(np.zeros(n_mels)); continue
        n = len(chunk)
        y = np.roll(chunk * _window(n), -(n // 2))
        if n % 2: y = np.delete(y, n // 2)
        power = np.abs(np.fft.rfft(y)) ** 2
        output.append(_filterbank(len(power), n_mels, sr) @ power)
    return np.log(np.asarray(output) + LOG_EPS).astype(np.float32)


def events_to_labels(events, n_frames, frame_ms):
    """One-hot ``(n_frames, 7)`` int32; rolls and dendens mark every frame they cover."""
    labels = np.zeros(n_frames, dtype=np.int64)
    dropped = 0
    for start, cls, end in events:
        first = int(start * 1000 // frame_ms)
        if not 0 <= first < n_frames: dropped += 1; continue
        if end is None:
            if labels[first]: dropped += 1
            else: labels[first] = cls
        else:
            last = min(int(end * 1000 // frame_ms), n_frames - 1)
            span = labels[first:last + 1]
            span[span == 0] = cls
    if dropped: raise ValueError(f"{dropped} out-of-range or overlapping events")
    return np.eye(7, dtype=np.int32)[labels]
