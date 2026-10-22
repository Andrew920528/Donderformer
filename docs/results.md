# Results

These are historical results from the original pipeline. Onset/joint F1 counted
held-note frames separately; current evaluation counts held-note starts. Re-score
saved charts before comparing these F1 numbers with new runs. Legacy eight-frame
paper metrics retain their definitions; see [evaluation notes](evaluation.md).

## TaikoNation CNN-LSTM baseline, full Oni split (PACE)

Run by ykim3001 on PACE with the pre-refactor pipeline; the notebook and its outputs are in `notebooks/reference/baseline_cnn_lstm_full_pace.ipynb`.

- Data: 1409 train / 174 validation / 178 test Oni charts (2 train charts dropped because their notes run past the end of the audio). 6.86M train windows.
- Training: 10 epochs, batch 128, lr 1e-5 (val loss 0.441 → 0.407), then 2 fine-tune epochs, batch 128, lr 5e-6 (val loss 0.404).

### Teacher-forced (human note history as input)

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| Onset (±1 frame) | 0.555 | 0.535 | 0.545 |
| Onset + type | 0.440 | 0.424 | 0.432 |

OCHuman-style recall: 55.5% micro, 53.8% macro.

### Autoregressive (model's own note history), mean over 178 test charts

| Metric | Raw | Clean (adjacent notes removed) |
|---|---|---|
| Onset precision | 0.312 | 0.383 |
| Onset recall | 0.285 | 0.239 |
| Onset F1 | 0.290 | 0.289 |
| Joint F1 | 0.105 | 0.112 |
| DCHuman | 81.9% | 82.6% |
| OCHuman | 83.8% | 84.6% |
| Over-PSpace | 42.9% | 20.0% |
| HI-PSpace | 99.5% | 99.2% |

DCHuman and OCHuman are dominated by correctly empty frames (~86% of frames have no note), so the F1 rows are the more informative comparison.
