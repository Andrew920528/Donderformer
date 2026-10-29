# Saved-chart evaluation and model comparison

All provisional choices are in the **one** `CONFIG` dictionary in
`src/donderformer/evaluation_config.py`, each marked `PLACEHOLDER - team to confirm`.
Notebook 06 performs optional saved CNN-LSTM inference, then calls the same
`evaluate_saved_charts` scorer that any other model can use. Notebook 07 compares
its saved results; neither notebook trains, extracts features, or changes splits.
`07_exports.ipynb` remains the simulator exporter.

## Inputs and difficulty

Inference needs a notebook-03 `model.pt`, saved run configuration, matching mel
features and labels, split manifest, train normalization, and any failure list.
Scoring existing charts needs **no checkpoint, mel features, or normalization**:
only generated NPZ files, original cached train/test labels and difficulty metadata.
The independent recipe at the end of notebook 06 shows this path.

`FeatureExtractor.labels_path` locates one-hot labels at
`features/<extractor>/songs/<audio_id>/labels/<chart_id>.npy`. The existing failure
list excludes unlabelled charts. Training references never include validation or
test charts. Human pattern pools include every usable test chart, including those
with no generated output from a particular model.

This dataset contains **Oni charts only**: difficulty means **star level within Oni**. `DIFFICULTY_FIELDS` is therefore `(level,)`.
All 1,763 rows of `oni_v1/manifest.csv` have course `oni` and blank level. Recovering
LEVEL from the matching course section of the linked TJA files succeeds for all
1,763 rows; every test star group has training examples. The existing exporter
uses the same recovery method. Metadata is read, never written back to the split.
Sparse groups use `MIN_CHARTS_PER_BAND_GROUP=30` training charts (provisional). Merge
the lowest-star sparse group with its nearest adjacent group, breaking ties toward
the lower star boundary, until the threshold is reached. Training counts alone
choose merges. If the entire training population remains below threshold, the
single remaining group is flagged below minimum. Bands, human pattern pools,
training type distributions, and all difficulty summaries use the same mapping.
Each analysis writes `difficulty_groups.json` (mapping, original/merged counts,
merge history) and `difficulty_groups.csv`; counts also print during evaluation.

Using the current committed manifest and linked raw TJA metadata, the mapping is
**1,2,3,4 → stars=1-4; 5→stars=5; 6→stars=6; 7→stars=7; 8→stars=8;
9→stars=9; 10→stars=10**:

| Merged Oni stars | Training charts | Test charts |
|---|---:|---:|
| 1–4 | 71 | 11 |
| 5 | 113 | 7 |
| 6 | 235 | 25 |
| 7 | 378 | 43 |
| 8 | 360 | 54 |
| 9 | 171 | 22 |
| 10 | 82 | 16 |
| Total | 1410 | 178 |

These are usable cached-label counts from the successful 178-chart evaluation.
The manifest contains 1,411 training charts; one failed labelling and is recorded
in `features/frame23ms/failures/oni_v1.json`. Its exclusion reduces the six-star
training count from 236 to 235 without changing the merged mapping. The original
metadata-only report at `outputs/analysis/oni_v1/` includes that manifest row.
See [full-run validation](evaluation_full_run.md) for checkpoint provenance,
cache preparation, results, and runtime.

## Definitions

- **Note starts:** classes 1–4 each count as a hit, including adjacent equal hits.
  For classes 5/6 only the first frame of a contiguous same-type run counts. A
  5→6 transition starts a new note. `events_to_labels` stores held notes across
  their covered frames; their duration is not itself a series of onsets.
- **Onset/joint F1:** greedy one-to-one nearest matches within the configured
  frame tolerance; ties select the earlier human note. Joint also requires an
  exact seven-class type match. Both sequences are collapsed to starts first.
  The count-based `score_from_counts` formula itself is unchanged; its callers
  must provide note-start counts. Empty-onset F1 retains the historical value 0.
- **Legacy paper metrics:** DCHuman, paper OCHuman, eight-frame Over_PSpace and
  same-song HI_PSpace retain their definitions, including empty frames and the
  256 possible binary patterns. The separate teacher-forced OCHuman-style recall
  resampling pass also retains its historical frame-based definition. Its
  micro/macro values must not be confused with paper OCHuman.
- **HI precision (primary):** for each binary/type variant and window, divide
  `|non-empty generated unique patterns ∩ human pool|` by
  `|non-empty generated unique patterns|`. The all-zero pattern never contributes
  to numerator or denominator. This is a fraction in [0,1], null if no non-empty
  generated windows exist. `HI_PRIMARY="precision"` selects it for plots and
  retention; `"coverage"` selects the retained coverage definition instead.
  A large pooled reference makes per-chart coverage small and length-dependent;
  precision measures whether the chart's own patterns are human-supported.
- **HI coverage (retained):** `100 * |generated ∩ human pool| / |human pool|`,
  including empty patterns. New `..._hi_coverage` and existing `..._hi` contain
  the same percentage; no old per-chart key is renamed or removed. Pools are
  unions over **all usable test humans in each merged star group**, built once
  per variant/window. Both variants collapse held continuations to starts.
- **Overall P-Space (headline):** `generated unique count / matching-human unique
  count` at every window, including empty patterns. Self-comparison is 1. New
  `..._overall_ratio` and existing `..._ratio` hold this ratio. Dividing by
  `7**64` makes a type-aware percentage nearly uninformative, so normalized
  percentages appear in new summaries only for the legacy eight-frame binary
  case. For compatibility, old normalized `..._overall`/`..._human_overall` values
  remain in per-chart JSON at all windows; long-window percentage rows and old
  alias rows are omitted from current summary/comparison tables. Binary/type
  alphabet sizes are 2/7. None of these changes alter the legacy paper metrics.
- **Density:** number of starts divided by elapsed seconds between first and
  last start (not total song duration). Fewer than two starts gives null.
- **Don/Ka:** `(don + big don) / (ka + big ka)`, excluding held notes. No Ka
  gives null. Reference bands use the configured percentiles of per-chart
  statistics over **training humans of the same merged star group**. In-band flags
  include the endpoints; their mean is the share in-band. Nulls are excluded,
  and valid counts are reported. Sparse groups merge before bands are fitted. A group with no finite values for a statistic has null bounds.
- **Random baseline:** choose the human number of distinct frame positions
  uniformly without replacement. Sample note types from the same difficulty's
  training **start** distribution. Adjacent identical held types cannot encode
  two starts in this frame format, so that type is excluded conditionally on an
  adjacent held start. If no supported alternative exists, fail explicitly.
  This small conditional departure from independent type sampling is necessary
  to preserve the exact start count. Clean applies the existing adjacent-note
  filter and can reduce the count. Seed randomness is keyed by seed and stable
  chart ID, independent of directory order or partial model coverage. Average
  seed metrics within chart before computing chart means and paired tests.
  **Random density is excluded because the control fixes the human note count;
  its active span may still differ.** Random summaries omit density and its
  in-band flag, as do all notebook 07 comparison rows/tests. Don/Ka stays.
  Existing per-chart/seed density fields remain for compatibility, and
  `random_baseline/summary_notes.json` records the exclusion.

Pattern window units: PATTERN_WINDOWS_FRAMES is measured in 23 ms frames
(8 frames ≈ 0.18 s, 64 frames ≈ 1.5 s). The proposal does not fix the units, so frames
are used for now. Measure-based windows (as in Hanzen et al., 2025) would keep windows
comparable across tempos but require beat/measure information; switch later if the
team decides to.

New tables use human-relative Overall ratios and primary HI precision, avoiding
vanishing long-window normalized percentages. Short charts
with no full window, empty reference pools, and undefined denominators give JSON
null, not an invented zero. Reference sets use compact byte strings, but exact
large-window unions can still need substantial RAM on the full dataset.

## Outputs and compatibility

```text
<run>/eval/
  teacher_forced/
    results.json                     # same six aggregate micro F1 keys
    ochuman.npz                      # same historical resampled recall arrays
    ochuman_by_difficulty.json        # new: overall/difficulty recall breakdown
    charts/<chart_id>.npz             # new: first-pass TF samples
    scored/                          # new: generic scoring outputs, as below
  ar_test/
    charts/<chart_id>.npz             # unchanged: human, generated, clean
    results.json                     # same list of song_id/raw/clean records
    summary.csv                      # same version,metric,mean columns
    analysis/
      results.json                   # extended per-chart raw/clean metrics
      summary.csv                    # chart means/SD/valid n, overall/difficulty
      micro_summary.csv              # count-aggregate F1, overall/difficulty
      config.json                    # settings, bands, pool sizes, label hashes
      difficulty_groups.json         # final star mapping, counts and merge history
      difficulty_groups.csv          # merged train/test counts
      random_baseline/               # generated for PRIMARY_MODE
        results.json                 # per-chart seed means
        seed_results.json            # individual seed scores
        summary.csv                  # means/SD, overall/difficulty
        config.json
        summary_notes.json           # density exclusion explanation
        difficulty_groups.json
        difficulty_groups.csv
  plots/
    human_vs_teacher_forced.png       # new preview
    human_vs_autoregressive.png       # preserved filename
    taiko_chart_comparison.png        # preserved filename
```

Pattern keys use `pspace_<binary|type>_w<frames>_...`. New suffixes are
`hi_precision`, `hi_coverage`, `hi_primary`, and `overall_ratio`; existing `hi`,
`overall`, `human_overall`, and `ratio` remain. `hi_precision`/`hi_primary` are
fractions; coverage aliases are percentages. The primary alias expresses either
choice on the fractional scale. Summary `headline` flags mark the configured HI
choice and Overall ratios; legacy normalized binary-eight-frame rows are diagnostic.
Existing `human_onsets`/`ai_onsets` retain the legacy paper counts. New `note_starts`
refers to the predicted sequence; density flags are `note_density_in_band` and
`don_ka_ratio_in_band`. A fractional random-baseline flag is its share of seeds
inside the band. New summary standard deviations use the population definition.

The generic scorer reads saved charts without changing them. If clean is absent,
it computes clean in memory. AR inference resumes existing files. Rescoring
updates metric files; it does not regenerate sequences. Confirm that resumed
sequences belong to the intended checkpoint. Full-sequence validation against
cached human labels detects chart/split mismatches but cannot prove which model
originally generated an otherwise unannotated NPZ.

**F1 behavior change:** held continuations used to count as onsets. They now
count only at their start. In the synthetic example
`human=[0,5,5,5,0,2,0,0]`, `generated=[0,5,0,0,0,2,0,0]`, both onset and joint F1
change from **2/3 to 1**. The legacy paper scores are unchanged. Historical F1
numbers in `docs/results.md` must be recomputed from saved charts before comparison.
A real shared-checkpoint evaluation is now available; see
[full-run results](evaluation_full_run.md). Historical checkpoints/results may
differ, so these new scores are not a controlled before/after comparison with
the older frame-counted results.

## Comparison and inference limits

Set `CONFIG['MODEL_RUNS']` to absolute saved-run directories. The LSTM and its
random baseline are required; DonderFormer may remain None. `PRIMARY_MODE` selects
AR or teacher-forced result folders, and `PRIMARY_VERSION` selects raw or clean.
The latter does not discard the other version from evaluation outputs.

Notebook 07 writes `outputs/comparisons/<mode>_<version>/metrics.csv`,
`wilcoxon.json`, `comparison.json`, and `plots/hi_pspace.png`, plus difficulty
figures. It additionally writes `retention.json` and `primary_test.json`. Re-running the same mode/version replaces these comparison artifacts.
IDs are intersected across available models and random, irrespective of row order.
Excluded IDs and source paths are recorded. The primary test uses the LSTM/Transformer
intersection independently of missing random charts. Per-metric null pairs are omitted;
both valid descriptive counts and paired test counts are reported. Reference
manifest, training/test label hashes, complete difficulty assignments, and metric
settings and merged mappings must agree between models. Different reference pools
are rejected. Comparison requires analysis schema `hi-precision-v2`: older saved
analysis must be rescored from existing charts with the scoring-only recipe, with
no retraining. The old per-chart keys and legacy root result formats are preserved.

The two-sided Wilcoxon implementation drops zero differences, uses average tied
ranks, and computes the conditional sign-permutation distribution by dynamic
programming, with no extra dependency or Monte Carlo setting. All-zero differences
return p=1; no available pairs return null. This handles the tied-rank distribution
explicitly rather than using the untied exact table. See the
[official SciPy signed-rank documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html)
for conventions and assumptions. The pre-declared **single primary test** is the overall paired two-sided Wilcoxon
test of `retention = HI(longest window) / HI(shortest window)`, for
`PRIMARY_TEST_VARIANT="binary"` and the configured `HI_PRIMARY`, mode and version.
Select these settings before inspecting results. Shortest/longest mean numerical
min/max window, not list order. A zero/null shortest value or null longest value
gives null retention. For random, divide the seed-averaged HI values, rather than
averaging per-seed ratios. Retention is not clipped; increases can exceed 1.

All other tests are **secondary**, including per-group retention and the other
variant. Their finite p-values form a **single Holm-Bonferroni family across all
metrics and difficulty groups** in this mode/version comparison. Null tests are
excluded from the family. See the [official Holm correction
convention](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
`metrics.csv` and `wilcoxon.json` keep raw `p_value` and add `p_value_raw`,
`p_value_adjusted`, `test_role`, `correction`, and `significant`. Secondary significance
uses adjusted p-values; the primary is unadjusted (its adjusted field equals raw).
Random rows are descriptive, with no p-value. The table's primary LSTM/Transformer
retention means use their valid primary pairs; other descriptive rows/figures use
the all-model common cohort. Null denominators/pairs remain explicit.

The signed-rank test assumes independent pairs and a symmetric paired-difference
distribution. Multiple charts can share audio, so chart-level inference may
overstate independence. No significance claims can be made before both real
models' matched charts are available.

## Transformer adapter and verification

No Transformer model, vocabulary, or token spec exists in the checkout.
`token_adapter.py` therefore marks an **assumed** alternating absolute-millisecond /
integer-class sequence ending in `'END'`. It validates collisions and ordering,
floors times to frames, and cannot recover held durations or window offsets.
`save_token_chart` pads to the human length and writes the three exporter keys,
refusing to overwrite an existing chart. Replace the assumptions with the team's
actual vocabulary, relative/absolute-time policy, window merge logic and duration
encoding before using real DonderFormer tokens.

Run `../env/bin/python -m unittest discover -s tests -v` from this PACE checkout
(or the equivalent project interpreter elsewhere). Tests include required metric
sanity cases, held-note counting, train/test reference separation, raw/clean NPZ
compatibility, exact random counts and seed means, chart-ID pairing, signed-rank
ties/zeros, both notebook workflows and exporter packaging. Fresh-process notebooks 02,
06, and 07 passed with the shared LSTM checkpoint on all 178 test charts,
including full training/test references, difficulty bands, random controls,
and memory/runtime measurements. The LSTM-versus-Transformer primary test and
secondary comparisons still await compatible Transformer results. Original
training normalization was not supplied, so its reconstructed training-only
statistics cannot be compared byte-for-byte with the training machine’s copy.
