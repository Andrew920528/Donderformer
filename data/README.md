# Data

```text
data/
├── raw/                                # ESE song folders, e.g. "01 Pop/<song>/<song>.tja" (gitignored)
├── splits/<split>/                     # DataSplitter output (committed, never edited in place)
│   ├── config.json                     # filters, ratios, seed
│   ├── manifest.csv                    # one row per kept chart: chart_id, audio_id, course, level, bpm, paths, split
│   └── report.json                     # totals, lost charts and reasons, distributions
└── features/<extractor>/               # FeatureExtractor cache (gitignored), e.g. frame23ms
    ├── config.json
    ├── songs/<audio_id>/
    │   ├── mel.npy                     # (n_frames, 80) log-mel, shared by all charts of the song
    │   └── labels/<chart_id>.npy       # (n_frames, 7) one-hot
    ├── normalization/<split>.npz       # mean/std from the split's train songs + manifest hash
    └── failures/<split>.json           # charts that couldn't be labelled, if any
```

- `audio_id` is `<audio file stem>_<hash of its path>`; `chart_id` is `<tja stem>_<hash of its path>_<course>`. Labels sit under the audio they belong to because a few songs have several `.tja` files for one audio.
- Manifest paths are relative to `raw/`. Code finds this folder through `donderformer.paths.DataPaths`: the `root` argument, then the `DONDER_DATA` environment variable, then `<repo>/data`.
- Labels depend on the frame grid, so they live with the features, not the split.
