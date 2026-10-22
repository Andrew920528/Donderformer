"""Evaluation helpers for saved charts and CNN-LSTM inference.

Onset/joint F1 count held-note starts and use greedy one-to-one matching (nearest, earlier frame breaks ties).
OCHuman-style recall allows prediction reuse; paper OCHuman includes empty frames.
The paper pattern metrics use unique eight-frame binary patterns, out of 256.
TaikoNation generation assumes 16 audio frames, 15 history frames and 4 targets.
Sampling generators belong to the caller so notebook sampling order stays explicit.
"""
from __future__ import annotations

import numpy as np
import torch

from donderformer.evaluation_config import CONFIG

def aggregate_overlapping_predictions(probs):
    # probs: (number_of_windows, 4, 7), from ONE song only
    n = len(probs)
    if n == 0:
        return np.empty((0, 7), dtype=np.float32)
    aggregated = np.zeros((n + 3, 7), dtype=np.float32)
    counts = np.zeros(n + 3, dtype=np.float32)
    for j in range(4):
        aggregated[j:j+n] += probs[:, j, :]
        counts[j:j+n] += 1
    return aggregated / counts[:, None]

def sample_chart(aggregated, rng):
    return np.array([rng.choice(7, p=p / p.sum()) for p in aggregated],
                    dtype=np.int64)

def reconstruct_true_sequence(Y):
    # Y: (number_of_windows, 4, 7), for ONE song only
    targets = Y.argmax(axis=2)
    n = len(targets)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    sequence = np.empty(n + 3, dtype=np.int64)
    sequence[:n] = targets[:, 0]
    sequence[n:] = targets[-1, 1:4]
    return sequence


def note_starts(sequence):
    """Keep every discrete hit, but only the first frame of each run of 5 or 6.

    A 5->6 transition starts a new held note. Adjacent discrete hits remain
    separate notes. This matches data.audio.events_to_labels / TJA export.
    """
    sequence = np.asarray(sequence)
    if sequence.ndim != 1 or not np.issubdtype(sequence.dtype, np.integer):
        raise ValueError("Charts must be one-dimensional integer class sequences")
    if np.any((sequence < 0) | (sequence > 6)):
        raise ValueError("Chart classes must be in [0, 6]")
    starts = sequence.copy()
    continuation = np.isin(sequence[1:], [5, 6]) & (sequence[1:] == sequence[:-1])
    starts[1:][continuation] = 0
    return starts


def onset_metrics(true_seq, pred_seq, tolerance=None):
    tolerance = CONFIG["ONSET_TOLERANCE_FRAMES"] if tolerance is None else tolerance
    true_seq, pred_seq = note_starts(true_seq), note_starts(pred_seq)
    true_onsets = np.where(true_seq != 0)[0]
    pred_onsets = np.where(pred_seq != 0)[0]

    matched_true = set()
    matches = 0

    for pred in pred_onsets:
        left = np.searchsorted(true_onsets, pred - tolerance, side="left")
        right = np.searchsorted(true_onsets, pred + tolerance, side="right")
        candidates = [
            t for t in true_onsets[left:right]
            if abs(t - pred) <= tolerance and t not in matched_true
        ]

        if candidates:
            best = min(candidates, key=lambda t: abs(t - pred))
            matched_true.add(best)
            matches += 1

    precision = matches / len(pred_onsets) if len(pred_onsets) else 0
    recall = matches / len(true_onsets) if len(true_onsets) else 0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1, matches

def joint_onset_type_metrics(true_seq, pred_seq, tolerance=None):
    tolerance = CONFIG["ONSET_TOLERANCE_FRAMES"] if tolerance is None else tolerance
    true_seq, pred_seq = note_starts(true_seq), note_starts(pred_seq)
    true_onsets = np.where(true_seq != 0)[0]
    pred_onsets = np.where(pred_seq != 0)[0]

    matched_true = set()
    matches = 0

    for pred in pred_onsets:
        left = np.searchsorted(true_onsets, pred - tolerance, side="left")
        right = np.searchsorted(true_onsets, pred + tolerance, side="right")
        candidates = [
            t for t in true_onsets[left:right]
            if abs(t - pred) <= tolerance
            and true_seq[t] == pred_seq[pred]
            and t not in matched_true
        ]

        if candidates:
            best = min(candidates, key=lambda t: abs(t - pred))
            matched_true.add(best)
            matches += 1

    precision = matches / len(pred_onsets) if len(pred_onsets) else 0
    recall = matches / len(true_onsets) if len(true_onsets) else 0

    if precision + recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1, matches


def score_from_counts(tp, predicted, actual):
    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def ochuman_score(true_seq, pred_seq, tolerance=1):
    true_onsets = np.flatnonzero(true_seq != 0)
    pred_mask = pred_seq != 0

    if len(true_onsets) == 0:
        return 0, 0

    matched = 0
    for t in true_onsets:
        left = max(0, t - tolerance)
        right = min(len(pred_seq), t + tolerance + 1)

        if np.any(pred_mask[left:right]):
            matched += 1

    return matched, len(true_onsets)


def generate_autoregressive(model, X, device, seed=None):
    """
    Autoregressive Taiko chart generation.

    X: preprocessed windows, shape (N, 1385)
    Only X[:, :1280] (audio) is used.
    Ground-truth note history is NEVER used.
    """
    model.eval()
    rng = np.random.default_rng(CONFIG["AR_SEED"] if seed is None else seed)

    n = len(X)
    total_steps = n + 3

    generated = np.zeros(total_steps, dtype=np.int64)

    # Accumulate overlapping predictions
    prob_sums = np.zeros((total_steps, 7), dtype=np.float32)
    counts = np.zeros(total_steps, dtype=np.int32)

    eye = np.eye(7, dtype=np.float32)

    # Keep only audio; discard ground-truth note history
    audio = X[:, :1280].astype(np.float32)

    with torch.inference_mode():
        for t in range(total_steps):

            if t < n:
                # Previous 15 GENERATED notes
                history = np.zeros((15, 7), dtype=np.float32)

                start = max(0, t - 15)
                previous = generated[start:t]

                if len(previous) > 0:
                    history[-len(previous):] = eye[previous]

                # Construct model input
                x = np.concatenate([
                    audio[t],
                    history.reshape(-1)
                ])

                batch = torch.from_numpy(
                    x[None, :]
                ).to(device)

                logits = model(batch)

                probs = torch.softmax(
                    logits, dim=2
                )[0].cpu().numpy()

                # Predictions for t, t+1, t+2, t+3
                for j in range(4):
                    idx = t + j
                    prob_sums[idx] += probs[j]
                    counts[idx] += 1

            # Sample current timestep using predictions
            # available so far (no future-note leakage)
            if counts[t] > 0:
                p = prob_sums[t] / counts[t]
                p = p / p.sum()
                generated[t] = rng.choice(7, p=p)

    return generated


def remove_adjacent_notes(sequence):
    cleaned = sequence.copy()

    for t in range(1, len(sequence)):
        if sequence[t - 1] != 0 and sequence[t] != 0:
            cleaned[t] = 0

    return cleaned


def to_binary_chart(sequence):
    sequence = np.asarray(sequence)

    # Discrete notes
    binary = np.isin(sequence, [1, 2, 3, 4])

    # Drumroll / Denden: only count onset
    held = np.isin(sequence, [5, 6])
    starts = np.r_[True, sequence[1:] != sequence[:-1]]

    binary |= held & starts
    return binary.astype(np.uint8)


def paper_ochuman(human, ai):
    """
    Paper-style approximation:
    Exact binary agreement, with ±1 timestep
    tolerance for human note onsets.
    """
    human = np.asarray(human, dtype=bool)
    ai = np.asarray(ai, dtype=bool)

    assert len(human) == len(ai)

    correct = (human == ai)

    nearby = ai.copy()
    nearby[1:] |= ai[:-1]
    nearby[:-1] |= ai[1:]

    correct |= human & nearby

    return float(correct.mean())


def extract_patterns(binary_chart, window=8):
    if not isinstance(window, int) or window <= 0:
        raise ValueError("Pattern window must be a positive integer")
    chart = np.asarray(binary_chart, dtype=np.uint8)

    return {
        tuple(chart[i:i + window])
        for i in range(len(chart) - window + 1)
    }


def calculate_paper_metrics(human, generated):
    h = to_binary_chart(human)
    a = to_binary_chart(generated)

    dc = float(np.mean(h == a) * 100)
    oc = paper_ochuman(h, a) * 100

    hp = extract_patterns(h)
    ap = extract_patterns(a)

    over = len(ap) / 256 * 100
    hi = len(hp & ap) / len(hp) * 100 if hp else 0.0

    return {
        "DCHuman": dc,
        "OCHuman": oc,
        "Over_PSpace": over,
        "HI_PSpace": hi,
        "human_onsets": int(h.sum()),
        "ai_onsets": int(a.sum())
    }


def calculate_f1_metrics(human, predicted, tolerance=None):
    p, r, f1, onset_matches = onset_metrics(
        human, predicted, tolerance=tolerance
    )

    jp, jr, jf1, joint_matches = joint_onset_type_metrics(
        human, predicted, tolerance=tolerance
    )

    return {
        "onset_precision": float(p),
        "onset_recall": float(r),
        "onset_f1": float(f1),
        "joint_precision": float(jp),
        "joint_recall": float(jr),
        "joint_f1": float(jf1),
        "onset_matches": int(onset_matches),
        "joint_matches": int(joint_matches)
    }



def load_evaluation_run(run_dir, paths, device="cpu"):
    """Restore a saved baseline and its test data without writing or training.

    Sidecar config takes precedence, as in runs.load_run. Older runs may store
    only a feature ID; decode that ID without needing to recreate the cache.
    """
    import json
    import re
    from pathlib import Path

    from donderformer.data.data_splitter import read_manifest
    from donderformer.data.feature_extractor import FeatureExtractor
    from donderformer.data.taiko_dataset import TaikoDataset
    from donderformer.models.taikonation import TaikoNationBaseline

    run_dir = Path(run_dir).expanduser()
    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Saved checkpoint missing: {checkpoint_path}. Set RUN_DIR to an existing "
            "notebook 03 run containing model.pt; copy the run from PACE if needed."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config_path = run_dir / "config.json"
    config = (json.loads(config_path.read_text()) if config_path.is_file()
              else checkpoint.get("config", {}))
    if config.get("model") != "taikonation":
        raise ValueError("Evaluation requires a saved config with model='taikonation'.")
    data = config.get("data", {})
    required = ("split", "features", "audio_frames", "history_frames", "target_frames")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Saved run configuration lacks data fields {missing}; "
                         "restore its original config.json rather than using training defaults.")
    window = {key: data[key] for key in required[2:]}
    if tuple(window.values()) != (16, 15, 4):
        raise ValueError(f"TaikoNation requires window sizes (16, 15, 4), got {window}.")
    if "extractor" in data:
        extractor = FeatureExtractor(**data["extractor"])
    else:
        match = re.fullmatch(r"frame(\d+)ms(?:_mel(\d+))?(?:_sr(\d+))?", data["features"])
        if not match:
            raise ValueError(f"Cannot reconstruct saved extractor {data['features']!r}; "
                             "restore exact extractor settings in config['data']['extractor'].")
        frame, mels, rate = match.groups()
        settings = {"frame_ms": int(frame)}
        if mels: settings["n_mels"] = int(mels)
        if rate: settings["sample_rate"] = int(rate)
        extractor = FeatureExtractor(**settings)
    if extractor.id != data["features"] or extractor.n_mels != 80:
        raise ValueError("Saved extractor settings disagree with the feature ID or baseline's 80 mel bands.")
    if extractor.frame_ms != 23:
        raise ValueError("These evaluation plots and exporter settings require the saved 23 ms frame grid.")
    feature_config = extractor.dir(paths) / "config.json"
    if feature_config.is_file():
        from dataclasses import asdict
        if json.loads(feature_config.read_text()) != asdict(extractor):
            raise ValueError(f"Feature cache config disagrees with saved run: {feature_config}")
    manifest = paths.split_dir(data["split"]) / "manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Original split manifest missing: {manifest}. Restore the saved run's split; do not regenerate it.")
    rows = read_manifest(paths, data["split"], "test")
    failed = extractor.failed_charts(paths, data["split"])
    rows = [row for row in rows if row["chart_id"] not in failed]
    if not rows:
        raise ValueError("Saved split has no usable test charts after applying feature failures.")
    needed = [extractor.normalization_path(paths, data["split"])]
    for row in rows:
        needed.extend([extractor.mel_path(paths, row["audio_id"]),
                       extractor.labels_path(paths, row["audio_id"], row["chart_id"])])
    missing_files = list(dict.fromkeys(p for p in needed if not p.is_file()))
    if missing_files:
        preview = "\n".join(str(p) for p in missing_files[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_files)} evaluation cache files, including:\n{preview}\n"
            "Restore the matching feature cache (mel, labels, train normalization and any "
            "failure list) from PACE, or point DataPaths/DONDER_DATA at it. This notebook does not extract features."
        )
    # TaikoDataset also checks the normalization's manifest hash.
    dataset = TaikoDataset(paths, data["split"], extractor, "test", **window)
    if not any(dataset.lengths):
        raise ValueError("Test charts contain no complete four-target windows to evaluate.")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"{checkpoint_path} lacks model_state_dict; restore a notebook 03 checkpoint.")
    model = TaikoNationBaseline().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config, dataset


def evaluate_saved_charts(charts_dir, context, output_dir=None, *, config=None):
    """Model-independent saved-chart scoring; see chart_evaluation for details."""
    from donderformer.chart_evaluation import evaluate_saved_charts as score
    return score(charts_dir, context, output_dir, config=config)


def predict_teacher_forced(model, X, device, batch_size):
    """Probabilities for one song's windows, in bounded inference batches."""
    if batch_size <= 0:
        raise ValueError('Inference batch size must be positive')
    if not len(X):
        return np.empty((0, 4, 7), dtype=np.float32)
    model.eval()
    batches = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start:start+batch_size].astype(np.float32)).to(device)
            batches.append(torch.softmax(model(batch), dim=2).cpu().numpy())
    return np.concatenate(batches)


def plot_chart_preview(sequences, path, config, max_steps=None):
    """Keep preview filenames/locations while using the shared plot settings."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    colors = {1: 'red', 2: 'blue', 3: 'darkred', 4: 'darkblue', 5: 'orange', 6: 'purple'}
    names = {1: 'Don', 2: 'Ka', 3: 'Big Don', 4: 'Big Ka', 5: 'Drumroll', 6: 'Denden'}
    step_seconds = config['FRAME_MS'] / 1000
    start = int(config['PLOT_START_SECONDS'] / step_seconds)
    fig, axes = plt.subplots(len(sequences), 1, figsize=config['FIGURE_SIZE'], sharex=True, constrained_layout=True)
    for ax, (label, sequence) in zip(np.atleast_1d(axes), sequences.items()):
        starts = note_starts(sequence)
        end = min(len(sequence), int(config['PLOT_END_SECONDS'] / step_seconds))
        if max_steps is not None:
            end = min(end, start+max_steps)
        for kind, color in colors.items():
            positions = np.flatnonzero(starts[start:end] == kind) + start
            ax.scatter(positions*step_seconds, np.ones(len(positions)), color=color, label=names[kind])
        ax.set_title(label); ax.set_yticks([]); ax.grid(axis='x', alpha=config['PLOT_GRID_ALPHA'])
    np.atleast_1d(axes)[0].legend(ncol=len(names))
    np.atleast_1d(axes)[-1].set_xlabel('Time (seconds)')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=config['PLOT_DPI'], bbox_inches='tight')
    plt.show()
    plt.close(fig)
