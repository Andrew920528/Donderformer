# Shared LSTM: full evaluation on oni_v1

The saved-run evaluation passed on all **178 test charts**, after a one-chart gate,
using the shared weights at `weights/lstm-baseline-1` on `origin/refactor`, commit
`a3ea7bc5ca25782702a1d6ad5d2d1f475c33ca82`. No retraining or split regeneration was
performed. This validates the LSTM evaluation workflow; the Transformer comparison
is still pending.

## Inputs and reproducibility

The checkpoint loads strictly into `TaikoNationBaseline`. Its embedded and sidecar
configurations agree: `oni_v1`, `frame23ms`, 16 audio frames, 15 history frames,
and four target frames. Model, feature extraction, and split source are unchanged
since the checkpoint's recorded training commit. Features use 23 ms frames,
80 mel bins, and 44.1 kHz audio.

Notebook 02 cached 1,746 audio files and labelled 1,762 charts. One training chart,
`chala_headchala_older_audio_27bf61_oni`, failed with "15 out-of-range or overlapping
events" and is recorded in the failure list. Usable train/validation/test counts
are **1,410 / 174 / 178**. The committed split manifest was unchanged. Normalization
was computed from training songs only. The original training normalization file
was not shared, so byte-for-byte equivalence cannot be verified.

The imported checkpoint, executed notebook copies, logs, generated charts, and
complete metrics are local, ignored artifacts under:

```text
outputs/runs/taikonation/EVAL_lstm-baseline-1_20261021_221617/
```

`provenance.json` records source commit, model/config hashes, and compatibility.
`results_summary/COMPLETE_RESULTS_SUMMARY.md` contains the full results narrative;
`ALL_METRICS_BY_DIFFICULTY.md` and `all_saved_metric_summaries.csv` contain all saved
means, population SDs, and valid counts. These artifacts are excluded from Git.

## Validation

- Notebook 02: all three code cells passed in 473.9 seconds.
- Notebook 06: all 12 code cells passed in a fresh process in 1,044.5 seconds.
- Notebook 07: both code cells passed in a fresh process in 8.9 seconds, using
  LSTM plus random, with DonderFormer set to None through a runtime-only override.
- All 178 human-self checks returned onset/joint F1, nonempty HI precision, and
  Overall ratio equal to 1 for binary/type windows 8, 16, 32, and 64. These are
  scorer sanity checks, not model performance.
- All 59,096 checked model/random scalar values were finite, with no nulls.
  Expected legacy results, generated NPZ keys, analysis, random-control, and plot
  output paths were present. Random summaries exclude density fields.
- AR generation averaged 4.49 seconds per chart over 177 newly generated charts;
  the first test chart resumed its identically seeded pilot output. AR scoring
  took 18.5 seconds total, or 0.104 seconds/chart. TF scoring took 20.1 seconds;
  random scoring over all three seeds took 57.9 seconds. Peak notebook-06 RSS was
  1,008.4 MiB. The compute allocation supplied four CPUs and 16 GiB, with no GPU.

Source notebooks and production CONFIG were unchanged by execution. Notebook 03
retains its training defaults and model/config saving behavior. Notebook 05 remains
empty: the reusable AR function is in `evaluation.py`, but notebook 06 still owns
the generation loop. Moving that workflow to 05 is separate scope.

## Results and aggregation

All overall cells below have **n=178 valid charts**. Onset matching uses one-to-one
note starts within ±1 frame (23 ms); joint matching also requires the note class.
Clean filters adjacent activity from the same raw sample. See
[evaluation definitions](evaluation.md) for exact protocols.

### Count-aggregated precision / recall / F1

Micro scores aggregate counts across songs before forming ratios. There are
92,908 human starts in the test set. These are fractions, not percentages.

| Mode | Version | Onset P / R / F1 | Joint P / R / F1 |
|---|---|---|---|
| Teacher-forced | Raw | .3402 / .3586 / .3492 | .1719 / .1812 / .1765 |
| Teacher-forced | Clean | .3728 / .3433 / .3575 | .1768 / .1628 / .1695 |
| Autoregressive | Raw | .3251 / .3550 / .3394 | .1484 / .1620 / .1549 |
| Autoregressive | Clean | .3487 / .3447 / .3466 | .1537 / .1519 / .1528 |

Chart-macro onset/joint F1, averaging each chart equally:

| Mode/control | Raw onset / joint F1 | Clean onset / joint F1 |
|---|---|---|
| Teacher-forced | .3400 / .1712 | .3482 / .1646 |
| Autoregressive | .3305 / .1504 | .3377 / .1484 |
| Random | .2775 / .1335 | .2821 / .1304 |

Random averages seeds 0, 1, 2 within each chart before averaging across charts:
534 seed/chart samples, but 178 chart-level observations. Saved SDs are population
SDs across chart-level values. Random has no saved micro summary.

### Legacy protocols

The separate teacher-forced OCHuman-style recall resampling pass reports **.5709
micro / .5550 macro**. It checks nonzero human frames, including held continuations,
for nearby predicted activity and allows reuse of predictions; it differs from
one-to-one note-start recall. AR/clean equivalents of that pass were not saved.

The following paper metrics are chart-macro **percentages**. Paper OCHuman measures
tolerant frame agreement, not recall. Empty frames contribute to agreement. Legacy
HI_PSpace covers patterns in the matching individual human chart, including empty
patterns; it is distinct from pooled multi-window HI below.

| Metric | TF raw | TF clean | AR raw | AR clean | Random raw | Random clean |
|---|---:|---:|---:|---:|---:|---:|
| DCHuman | 82.3411 | 83.2153 | 81.2409 | 82.0776 | 81.7388 | 82.6054 |
| OCHuman | 84.2372 | 85.2114 | 83.4449 | 84.2976 | 83.6444 | 84.5193 |
| Over_PSpace | 49.8815 | 20.1435 | 44.8758 | 20.1391 | 44.4581 | 19.3206 |
| HI_PSpace | 99.4745 | 99.2532 | 99.5880 | 99.2864 | 99.6342 | 99.4107 |

### Multi-window patterns

Windows 8/16/32/64 correspond to 184/368/736/1,472 ms. Each cell is **HI precision
(fraction) / pooled HI coverage (%) / matching-human Overall ratio**. These are
means of chart-level metrics, not ratios of pooled counts. Empty generated
patterns are excluded from HI precision. Coverage and Overall counts retain them.
Human pools use all test charts in the merged group, including the target chart.
Overall ratio above 1 means more unique patterns, not necessarily human patterns.

| Variant/window | TF raw | TF clean | AR raw | AR clean | Random raw | Random clean |
|---|---|---|---|---|---|---|
| Binary 8 | .3364 / 96.77 / 6.786 | .7994 / 94.24 / 2.775 | .3754 / 96.80 / 6.128 | .8010 / 94.33 / 2.770 | .3789 / 96.05 / 5.683 | .8235 / 93.36 / 2.603 |
| Binary 16 | .3043 / 55.51 / 10.424 | .5709 / 62.22 / 6.352 | .3428 / 62.94 / 10.650 | .5834 / 68.01 / 6.833 | .3565 / 53.13 / 7.788 | .6158 / 58.15 / 5.026 |
| Binary 32 | .2089 / 7.39 / 5.248 | .3042 / 9.58 / 4.799 | .2393 / 8.52 / 5.432 | .3166 / 10.48 / 5.133 | .2537 / 7.92 / 4.242 | .3444 / 9.50 / 3.850 |
| Binary 64 | .0351 / .2370 / 1.670 | .0417 / .2775 / 1.668 | .0359 / .2335 / 1.672 | .0415 / .2682 / 1.670 | .0421 / .3508 / 1.620 | .0508 / .4069 / 1.612 |
| Type 8 | .3063 / 67.13 / 6.325 | .5326 / 68.33 / 3.740 | .3404 / 66.23 / 5.601 | .5593 / 66.93 / 3.444 | .3519 / 64.02 / 5.044 | .5774 / 65.09 / 3.145 |
| Type 16 | .3154 / 18.29 / 4.769 | .4713 / 21.20 / 3.778 | .3707 / 21.77 / 4.880 | .5174 / 24.51 / 3.997 | .3632 / 17.23 / 3.618 | .5122 / 19.35 / 2.926 |
| Type 32 | .1581 / 1.703 / 2.145 | .2116 / 2.160 / 2.079 | .1769 / 1.887 / 2.171 | .2168 / 2.242 / 2.133 | .2048 / 2.033 / 1.875 | .2574 / 2.389 / 1.802 |
| Type 64 | .0201 / .1035 / 1.283 | .0222 / .1139 / 1.282 | .0191 / .0946 / 1.283 | .0206 / .1018 / 1.283 | .0194 / .1274 / 1.271 | .0218 / .1382 / 1.268 |

Random exceeds the LSTM's overall 64-frame HI precision for both variants and
versions. These comparisons are descriptive; no model-versus-random p-values were
computed. High short-window coverage does not establish model quality.

### Density and merged Oni star groups

Density uses starts per second between first/last starts. Don/Ka excludes held
classes. Bands are inclusive training-only 10th–90th percentiles. In-band values
below are percentages of charts; random density is excluded because raw random
fixes human start count (its active span can still differ).

| Mode/control | Version | Density (starts/s) | Density in-band | Don/Ka | Don/Ka in-band |
|---|---|---:|---:|---:|---:|
| Teacher-forced | Raw | 4.8908 | 63.48% | 1.2689 | 99.44% |
| Teacher-forced | Clean | 4.2731 | 61.80% | 1.2732 | 100% |
| Autoregressive | Raw | 5.0560 | 54.49% | 1.2671 | 99.44% |
| Autoregressive | Clean | 4.5773 | 64.04% | 1.2712 | 99.44% |
| Random | Raw | Excluded | Excluded | 1.3613 | 100% |
| Random | Clean | Excluded | Excluded | 1.3605 | 100% |

This dataset is Oni-only; difficulty means star level within Oni. Minimum training
count is 30, producing groups 1–4, 5, 6, 7, 8, 9, 10.

| Oni stars | Train/test charts | Training density band (starts/s) | Training Don/Ka band |
|---|---:|---:|---:|
| 1–4 | 71 / 11 | 2.0316–3.6950 | 1.1172–3.7366 |
| 5 | 113 / 7 | 2.5698–4.2405 | 1.0570–2.5678 |
| 6 | 235 / 25 | 2.9868–4.7733 | .9662–2.3944 |
| 7 | 378 / 43 | 3.5699–5.1848 | .9420–2.0919 |
| 8 | 360 / 54 | 4.2210–6.0770 | .9291–1.7929 |
| 9 | 171 / 22 | 4.9335–7.0218 | .8842–1.6761 |
| 10 | 82 / 16 | 6.7336–8.3344 | .8910–1.7183 |

Training valid n equals training count except stars 1–4 Don/Ka, where n=68 because
three charts have no Ka. All test metrics have valid n equal to group test count.
AR density is outside its band for every 1–4-star and 10-star chart, raw and clean.

## Pending comparisons

Notebook 07 saved AR/raw comparisons only. Per-chart HI retention is precision at
64 frames divided by precision at 8 frames; its mean is a mean of chart ratios.
Binary retention is .1048 (SD .0521) for LSTM and .1139 (SD .0944) for random;
type retention is .0586 (SD .0299) and .0523 (SD .0491), respectively, all n=178.

The primary paired Wilcoxon test, LSTM versus DonderFormer binary retention,
reports **`awaiting_donderformer`**: zero pairs, null statistic and p-values. All
360 saved test rows (one primary, 359 secondary) await Transformer results.
Secondary finite p-values will form one Holm-Bonferroni family. Missing tests are
not non-significant results. Random is descriptive. TF/clean notebook-07 model
comparisons and paired TF-versus-AR/raw-versus-clean tests were not performed.
