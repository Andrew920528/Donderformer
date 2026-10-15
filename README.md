# DonderFormer

A study of Transformers for Taiko no Tatsujin chart generation, with a CNN-LSTM baseline reproduced from TaikoNation.

## Layout

```text
Donderformer/
├── src/donderformer/        # importable logic
│   ├── paths.py             # DataPaths: where data lives
│   └── data/                # scan.py, splits.py, dataset.py (TaikoDataset)
├── notebooks/               # one notebook per process
│   ├── 01_scan_and_split.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_train_lstm.ipynb
│   └── reference/taikonation_repro.ipynb
├── data/                    # see data/README.md; only data/splits/ is committed
├── outputs/                 # checkpoints and generated charts (gitignored)
└── playground/              # scratch (gitignored) + toy_dataset/ (committed)
```

## Setup

```bash
cd /path/to/Donderformer
pip install -e ".[train]"
```

This makes `import donderformer` work from any notebook. Re-running it is only needed if `pyproject.toml` changes.

## Where the data lives

`DataPaths()` uses the `DONDER_DATA` environment variable if set, otherwise `<repo>/data`. Every data root has the same layout (`raw/`, `splits/`, `processed/<run>/`), so switching datasets is one line:

```python
from donderformer.paths import DataPaths
paths = DataPaths()         # full dataset
paths = DataPaths.toy()     # playground/toy_dataset, for checking that a notebook runs
```

The full ESE dataset only lives on PACE. Put the genre folders directly in `data/raw/` (no extra `ESE/` level):

```bash
ESE=/path/to/your/ESE
REPO=/path/to/Donderformer
mkdir -p "$REPO/data/raw" && mv "$ESE"/* "$REPO/data/raw/" && rmdir "$ESE"
```

## Pipeline

1. **`01_scan_and_split.ipynb`** scans `raw/` for Oni charts the parser supports, then splits them 80/10/10 by audio file (seed 42). Output goes to `splits/`. The committed `data/splits/` already holds the split used so far, so this only needs re-running when the dataset changes.
2. **`02_preprocess.ipynb`** turns each chart into 23 ms log-mel frames and 7-class labels, normalizes with training-set statistics and builds sliding windows. Output goes to `processed/<RUN_NAME>/`.
3. **`03_train_lstm.ipynb`** trains and evaluates the CNN-LSTM baseline on `processed/<RUN_NAME>/` and saves the checkpoint to `outputs/checkpoints/`.

Processed data and checkpoints move between PACE and a laptop with rsync, never git:

```bash
rsync -avzP yhsu72@login-ice.pace.gatech.edu:/scratch/taiko/Taiko_DL/Donderformer/data/processed/frame23ms_oni/ /Users/xuyanshuo/projects/Taiko_DL/Donderformer/data/processed/frame23ms_oni/
```

Each `windows.npz` contains `X` with shape `(number_of_windows, 1385)` and `Y` with shape `(number_of_windows, 4, 7)`.

## Model Input

For each prediction window:

```text
16 Audio Frames (16 × 80)
        +
15 Previous Note Frames (15 × 7)
        ↓
1385 Input Values
        ↓
CNN-LSTM
        ↓
4 Predicted Timesteps × 7 Note Classes
```

The seven note classes are:

```text
0 → None
1 → Don
2 → Ka
3 → Big Don
4 → Big Ka
5 → Drumroll
6 → Denden
```

## Evaluation

### Onset Evaluation

The basic onset evaluation checks whether the model generates a note at approximately the correct timing.

A generated note is considered a match if it occurs within ±1 timestep (±23 ms) of a ground-truth note.

The note type is not considered for this metric.

### Joint Onset + Type Evaluation

The joint evaluation is stricter.

A prediction is counted as correct only when:

1. The generated note occurs within ±23 ms of the ground-truth note.
2. The generated note type matches the ground-truth note type.

Precision, recall, and F1 score are reported for both evaluations.
