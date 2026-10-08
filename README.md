# DonderFormer Data Preprocessing and Baseline

This directory contains the data preprocessing pipeline and CNN-LSTM baseline for the DonderFormer project.

Because the ESE dataset is large, the raw dataset is not stored in this repository. Dataset scanning and preprocessing can be run on Georgia Tech PACE, and the generated processed data can then be downloaded and used locally for model training.

## Pipeline

```text
ESE Dataset
    ↓
Dataset Scanning / Filtering
    ↓
Filtered Dataset Manifest
    ↓
Audio + TJA Preprocessing
    ↓
Model-Ready NumPy Files
    ↓
Download Processed Data to Local Machine
    ↓
CNN-LSTM Baseline Training and Evaluation
```

## 1. Set Up the ESE Dataset on PACE

Place the extracted ESE dataset in your PACE storage.

The dataset should contain the TJA chart files and their corresponding audio files.

Example:

```text
ESE/
├── song_1/
│   ├── song.tja
│   └── song.ogg
├── song_2/
│   ├── song.tja
│   └── song.ogg
└── ...
```

The raw ESE dataset is not included in this repository.

## 2. Run the Dataset Scanner on PACE

Clone the repository on PACE and navigate to the `playground` directory:

```bash
cd <path-to-Donderformer>/playground
```

Run the dataset scanner:

```bash
python scan_dataset.py
```

`scan_dataset.py` scans the ESE dataset and filters out charts that are not currently supported by our preprocessing pipeline.

The scanner generates a manifest containing the charts that can be safely used for preprocessing.

## 3. Run Preprocessing

Open and run:

```text
new_preprocess.ipynb
```

The preprocessing pipeline:

1. Loads the selected songs from the filtered manifest.
2. Converts each audio file into 23 ms frames.
3. Extracts 80-dimensional log-Mel features.
4. Parses the corresponding TJA chart into timed note events.
5. Aligns the chart labels with the 23 ms audio frames.
6. Splits the songs into training, validation, and test sets.
7. Computes per-Mel-band normalization statistics using only the training set.
8. Applies the training-set normalization statistics to the training, validation, and test sets.
9. Constructs model input and output windows.

For each song, the model-ready data is stored in:

```text
windows.npz
```

Each `windows.npz` contains:

```text
X → Model input windows
Y → Ground-truth target sequences
```

The expected shapes are:

```text
X: (number_of_windows, 1385)
Y: (number_of_windows, 4, 7)
```

Each input window contains:

```text
16 audio frames × 80 Mel features = 1280
15 previous note frames × 7 classes = 105

1280 + 105 = 1385 input values
```

Each target contains:

```text
4 timesteps × 7 note classes
```

## 4. Download the Processed Data from PACE

After preprocessing finishes, download the generated output directory from PACE to your local machine.

Run the following from your **local terminal**:

```bash
scp -r <GT_USERNAME>@<PACE_HOST>:<remote-path>/processed_10 ./playground/
```

Replace:

- `<GT_USERNAME>` with your Georgia Tech username.
- `<PACE_HOST>` with the PACE login host.
- `<remote-path>` with the location of the DonderFormer project on PACE.

After downloading, the local directory should look similar to:

```text
playground/
├── baseline_cnn_lstm.ipynb
├── new_preprocess.ipynb
├── scan_dataset.py
└── processed_10/
    ├── split.json
    ├── normalization_stats.npz
    └── normalized/
        ├── <song_id>/
        │   ├── features.npy
        │   └── windows.npz
        └── ...
```

## 5. Run the CNN-LSTM Baseline Locally

Open:

```text
baseline_cnn_lstm.ipynb
```

The notebook loads the processed dataset using:

```python
OUTPUT_DIR = Path("processed_10")
DATA_DIR = OUTPUT_DIR / "normalized"
```

The baseline notebook then:

1. Loads the training, validation, and test splits.
2. Trains the CNN-LSTM model.
3. Runs the trained model on the test set.
4. Produces probabilities for the seven note classes.
5. Aggregates overlapping predictions from the sliding windows.
6. Samples notes from the aggregated probability distributions.
7. Evaluates the generated chart using onset and onset + note-type metrics.

The raw ESE audio and TJA files are not required for this stage because the notebook uses the preprocessed NumPy files.

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

## Files Not Stored in Git

Large datasets, generated outputs, model checkpoints, and temporary system files should not be committed to Git.

The `.gitignore` should include:

```gitignore
.DS_Store
**/.DS_Store

*.zip
*.pt

playground/input/
playground/processed_*/

__pycache__/
*.py[cod]

.ipynb_checkpoints/

.venv/
venv/
env/
```
