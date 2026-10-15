# Data

```text
data/
├── raw/                # ESE song folders, e.g. "01 Pop/<song>/<song>.tja" (gitignored)
├── splits/             # committed
│   ├── manifest_strict.csv   # every scanned chart with accept/reject reason
│   ├── manifest.csv          # accepted charts + song_id + split
│   └── split.json            # song IDs per split (seed 42)
└── processed/<run>/    # preprocessing output for one configuration (gitignored)
    ├── split.json, normalization.npz, summary.json
    ├── raw/<song_id>/        # features_raw.npy, labels.npy, metadata.json
    └── normalized/<song_id>/ # features.npy, windows.npz
```

Manifest paths are relative to `raw/`. Code finds this folder through `donderformer.paths.DataPaths`: the `root` argument, then the `DONDER_DATA` environment variable, then `<repo>/data`.
