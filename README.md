# DonderFormer

A study of Transformers for Taiko no Tatsujin chart generation, with a CNN-LSTM baseline reproduced from TaikoNation.

## Layout

```text
Donderformer/
├── src/donderformer/        # importable logic
│   ├── paths.py             # DataPaths: where data lives
│   ├── training.py          # train_epoch / evaluate loops
│   ├── runs.py              # one output folder per training run
│   ├── models/
│   │   └── taikonation.py   # TaikoNationBaseline (CNN-LSTM)
│   └── data/
│       ├── tja.py           # TJA parsing and validation
│       ├── tja_export.py    # model output (per-frame classes) -> playable .tja
│       ├── audio.py         # log-mel features, frame labels
│       ├── data_splitter.py      # DataSplitter: scan, filter, split, report
│       ├── feature_extractor.py  # FeatureExtractor
│       └── taiko_dataset.py      # TaikoDataset, SongBatchSampler
├── notebooks/               # one notebook per process
│   ├── 01_make_split.ipynb
│   ├── 02_extract_features.ipynb
│   ├── 03_train_lstm.ipynb
│   ├── 06_exports.ipynb     # a run's generated charts as simulator-ready song folders
│   └── reference/           # earlier notebooks kept as-is for comparison
├── data/                    # see data/README.md; only data/splits/ is committed
├── outputs/                 # checkpoints and generated charts (gitignored)
└── playground/              # scratch (gitignored) + toy_dataset/ (committed)
```

## Setup

The environment is managed with [uv](https://docs.astral.sh/uv/):

```bash
cd /path/to/Donderformer
uv sync
```

This creates `.venv/` with the pinned dependencies (`uv.lock`) and installs `donderformer` in editable mode, so `import donderformer` works everywhere. Run things through it with `uv run`, e.g. `uv run jupyter lab`, or pick the `.venv` interpreter as the notebook kernel in VS Code.

## Where the data lives

`DataPaths()` uses the `DONDER_DATA` environment variable if set, otherwise `<repo>/data`. Every data root has the same layout (`raw/`, `splits/`, `features/`), so switching datasets is one line:

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

1. **`01_make_split.ipynb`** uses a `DataSplitter` (courses, genres, BPM and level ranges, ratios, seed) to pick charts and split them by audio file. It writes `splits/<name>/` with `config.json`, `manifest.csv` and a `report.json` of what was kept, what was lost and why, plus distributions. Splits are never edited in place; a new filter means a new name. `data/splits/oni_v1` is the Oni split used so far.
2. **`02_extract_features.ipynb`** uses a `FeatureExtractor` (frame length) to cache log-mel per audio file and labels per chart under `features/<extractor.id>/`, then fits that split's normalization on its train songs. Cached songs are skipped, so new splits are cheap.
3. **`03_train_lstm.ipynb`** loads `TaikoDataset(paths, split, extractor, part, audio_frames, history_frames, target_frames)`, which cuts windows on the fly, then trains the CNN-LSTM baseline (stage 1, then a lower-learning-rate fine-tune) and saves it to a new run folder (see below). The same notebook then evaluates on the test charts, both teacher-forced (the model sees the human chart's previous notes) and autoregressive (it sees only its own generated notes, as it would on a new song), and writes metrics, generated charts and plots to that run's `eval/` folder. Results of the full-data PACE run are in [docs/results.md](docs/results.md).
4. **`06_exports.ipynb`** turns a run's generated test charts into `<run>/exports/<song title>/` folders, each with the song's audio and a `.tja` (`donderformer.data.tja_export.write_tja`) on its own BPM and OFFSET. Upload the whole `exports/` folder to a TJA player such as OpenTaiko to play them.

### Training runs

Every run of `03_train_lstm.ipynb` gets its own folder, so runs never overwrite each other:

```text
outputs/runs/<model>/<YYYYMMDD_HHMMSS>[-<alias>]-<split>-<extractor>/
├── config.json   # split, extractor, window sizes, hyperparameters, git commit, user, host
├── model.pt      # model_state_dict + losses + config
├── eval/         # metrics, generated charts and plots
└── exports/      # simulator-ready song folders, made by 06_exports
```

Set `ALIAS` in the config cell to label a run (letters, digits, `_` and `.`), e.g. `20261020_194245-bs1_finetune-oni_v1-frame23ms`. To evaluate an earlier run, point `RUN_DIR` at its folder in the cell after "Make predictions on the test song" and call `load_run`. `outputs/` is `<repo>/outputs` (gitignored) unless `DONDER_OUTPUTS` is set. The pre-refactor PACE model lives in `outputs/runs/taikonation/20261015_203543-pace_legacy-oni_v1-frame23ms/` on machines that have copied it.

Feature caches and runs move between PACE and a laptop with rsync, never git:

```bash
rsync -avzP yhsu72@login-ice.pace.gatech.edu:~/scratch/Taiko_DL/Donderformer/data/features/frame23ms/ /Users/xuyanshuo/projects/Taiko_DL/Donderformer/data/features/frame23ms/
```

With the default window sizes each sample has `X` of length 1385 and `Y` of shape `(4, 7)`.

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

### Autoregressive evaluation

Teacher-forced scores are optimistic, because the model's note-history input comes from the human chart. `generate_autoregressive` in `03_train_lstm.ipynb` keeps only the audio part of each window and feeds back the model's own last 15 sampled notes, one 23 ms step at a time. Each test chart is scored raw and "clean" (back-to-back notes removed) with the F1 metrics above plus TaikoNation's DCHuman, OCHuman, Over-PSpace and HI-PSpace.
