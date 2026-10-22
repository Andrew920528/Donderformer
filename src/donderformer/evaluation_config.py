"""One place for evaluation choices. Values are provisional, not fitted to results.

Class IDs, unit conversions, and the legacy paper's 8-frame/256-pattern protocol
are format/metric constants rather than tuning parameters.
"""
CONFIG = {
    "FRAME_MS": 23,  # PLACEHOLDER - team to confirm
    "ONSET_TOLERANCE_FRAMES": 1,  # PLACEHOLDER - team to confirm
    "PATTERN_WINDOWS_FRAMES": [8, 16, 32, 64],  # PLACEHOLDER - team to confirm
    "HI_PRIMARY": "precision",  # PLACEHOLDER - team to confirm; precision | coverage
    "PRIMARY_TEST_VARIANT": "binary",  # PLACEHOLDER - team to confirm; binary | type
    "MIN_CHARTS_PER_BAND_GROUP": 30,  # PLACEHOLDER - team to confirm
    "PERCENTILE_BAND": (10, 90),  # PLACEHOLDER - team to confirm
    "PRIMARY_MODE": "ar",  # PLACEHOLDER - team to confirm
    "PRIMARY_VERSION": "raw",  # PLACEHOLDER - team to confirm
    "RANDOM_BASELINE_SEEDS": [0, 1, 2],  # PLACEHOLDER - team to confirm
    "SIGNIFICANCE_ALPHA": 0.05,  # PLACEHOLDER - team to confirm
    "DIFFICULTY_FIELDS": ("level",),  # PLACEHOLDER - team to confirm
    "TEACHER_FORCED_SEED": 42,  # PLACEHOLDER - team to confirm
    "AR_SEED": 42,  # PLACEHOLDER - team to confirm
    "INFERENCE_BATCH_SIZE": 64,  # PLACEHOLDER - team to confirm
    "PREVIEW_STEPS": 300,  # PLACEHOLDER - team to confirm
    "PLOT_START_SECONDS": 0,  # PLACEHOLDER - team to confirm
    "PLOT_END_SECONDS": 20,  # PLACEHOLDER - team to confirm
    "PLOT_GRID_ALPHA": 0.25,  # PLACEHOLDER - team to confirm
    "PLOT_DPI": 200,  # PLACEHOLDER - team to confirm
    "PROGRESS_EVERY": 10,  # PLACEHOLDER - team to confirm
    "FIGURE_SIZE": (16, 6),  # PLACEHOLDER - team to confirm
    "MODEL_RUNS": {"lstm": "<saved-lstm-run>", "donderformer": None},  # PLACEHOLDER - team to confirm
}


def evaluation_config(overrides=None):
    """Copy defaults and validate overrides without mutating notebook/global state."""
    from copy import deepcopy
    import math

    config = deepcopy(CONFIG)
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown evaluation settings: {sorted(unknown)}")
        config.update(deepcopy(overrides))
    if not math.isfinite(config['FRAME_MS']) or config['FRAME_MS'] <= 0:
        raise ValueError('FRAME_MS must be positive and finite')
    if not isinstance(config['ONSET_TOLERANCE_FRAMES'], int) or config['ONSET_TOLERANCE_FRAMES'] < 0:
        raise ValueError('ONSET_TOLERANCE_FRAMES must be a nonnegative integer')
    windows = config['PATTERN_WINDOWS_FRAMES']
    if not windows or any(not isinstance(w, int) or w <= 0 for w in windows) or len(set(windows)) != len(windows):
        raise ValueError('PATTERN_WINDOWS_FRAMES must contain distinct positive integers')
    if config['HI_PRIMARY'] not in ('precision', 'coverage'):
        raise ValueError('HI_PRIMARY must be precision or coverage')
    if config['PRIMARY_TEST_VARIANT'] not in ('binary', 'type'):
        raise ValueError('PRIMARY_TEST_VARIANT must be binary or type')
    if not isinstance(config['MIN_CHARTS_PER_BAND_GROUP'], int) or config['MIN_CHARTS_PER_BAND_GROUP'] <= 0:
        raise ValueError('MIN_CHARTS_PER_BAND_GROUP must be a positive integer')
    if len(windows) < 2:
        raise ValueError('Retention needs at least two distinct pattern windows')
    low, high = config['PERCENTILE_BAND']
    if not 0 <= low <= high <= 100:
        raise ValueError('PERCENTILE_BAND must be ordered percentiles in [0, 100]')
    if config['PRIMARY_MODE'] not in ('ar', 'teacher_forced'):
        raise ValueError('PRIMARY_MODE must be ar or teacher_forced')
    if config['PRIMARY_VERSION'] not in ('raw', 'clean'):
        raise ValueError('PRIMARY_VERSION must be raw or clean')
    if not config['RANDOM_BASELINE_SEEDS'] or any(not isinstance(s, int) or s < 0 for s in config['RANDOM_BASELINE_SEEDS']):
        raise ValueError('RANDOM_BASELINE_SEEDS must contain nonnegative integers')
    if len(set(config['RANDOM_BASELINE_SEEDS'])) != len(config['RANDOM_BASELINE_SEEDS']):
        raise ValueError('Random baseline seeds must be distinct')
    if not 0 < config['SIGNIFICANCE_ALPHA'] < 1:
        raise ValueError('SIGNIFICANCE_ALPHA must be in (0, 1)')
    if tuple(config['DIFFICULTY_FIELDS']) != ('level',):
        raise ValueError('This Oni-only evaluation groups difficulty by star level; DIFFICULTY_FIELDS must be (level,)')
    for key in ('INFERENCE_BATCH_SIZE', 'PREVIEW_STEPS', 'PROGRESS_EVERY', 'PLOT_DPI'):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('TEACHER_FORCED_SEED', 'AR_SEED'):
        if not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    if not 0 <= config['PLOT_GRID_ALPHA'] <= 1:
        raise ValueError('PLOT_GRID_ALPHA must be in [0, 1]')
    if not 0 <= config['PLOT_START_SECONDS'] < config['PLOT_END_SECONDS']:
        raise ValueError('Plot seconds must define an increasing nonnegative interval')
    if len(config['FIGURE_SIZE']) != 2 or any(not math.isfinite(v) or v <= 0 for v in config['FIGURE_SIZE']):
        raise ValueError('FIGURE_SIZE must have two positive finite dimensions')
    return config
