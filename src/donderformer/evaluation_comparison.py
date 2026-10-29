"""Paired saved-result comparisons with no SciPy dependency."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from donderformer.chart_evaluation import _write_json, ANALYSIS_SCHEMA, DENSITY_METRICS, RANDOM_DENSITY_NOTE, report_metric, metric_headline
from donderformer.evaluation_config import evaluation_config


def wilcoxon_signed_rank(left, right):
    """Two-sided Wilcoxon with exact conditional sign permutations.

    Drop zero differences ('wilcox'); average ranks for exact magnitude ties.
    Dynamic programming enumerates rank-sum probabilities, including ties,
    without drawing Monte Carlo permutations. p=1 for all-zero differences.
    Differences use exact float equality, with no unconfigured rounding.
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
    for signed-rank assumptions, zero conventions, and the ties caveat for the
    usual untied 'exact' distribution. This implementation conditions on ties.
    """
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.ndim != 1 or left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Wilcoxon requires equally sized finite paired vectors')
    differences = right - left
    differences = differences[differences != 0]
    if not len(differences):
        return dict(statistic=0.0, p_value=1.0 if len(left) else None,
                    n_pairs=len(left), n_nonzero=0, status='all_zero' if len(left) else 'no_pairs')
    magnitudes = np.abs(differences)
    order = np.argsort(magnitudes, kind='stable')
    ranks2 = np.empty(len(order), dtype=np.int64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and magnitudes[order[end]] == magnitudes[order[start]]:
            end += 1
        # Twice the mean of ranks start+1,...,end is start+1+end.
        ranks2[order[start:end]] = start + 1 + end
        start = end
    plus = int(ranks2[differences > 0].sum())
    statistic2 = min(plus, int(ranks2.sum()) - plus)
    probabilities = np.array([1.0])
    for rank in ranks2:
        updated = np.zeros(len(probabilities) + rank)
        updated[:len(probabilities)] += probabilities / 2
        updated[rank:] += probabilities / 2
        probabilities = updated
    p_value = min(1.0, float(2 * probabilities[:statistic2+1].sum()))
    return dict(statistic=statistic2 / 2, p_value=p_value, n_pairs=len(left),
                n_nonzero=len(differences), status='ok')


def analysis_dir(run_dir, config):
    run_dir = Path(run_dir).expanduser()
    base = run_dir/'eval'
    return (base/'ar_test'/'analysis' if config['PRIMARY_MODE'] == 'ar'
            else base/'teacher_forced'/'scored'/'analysis')


def load_model_results(model_runs, config=None):
    """Require LSTM results; optional Transformer can be added later.

    Enforce identical reference provenance and scoring settings. Silently mixing
    old F1 definitions or different human pattern pools would invalidate pairing.
    """
    config = evaluation_config(config)
    models, sources, skipped = {}, {}, {}
    reference = None
    reference_config = None
    for model, run_dir in model_runs.items():
        if run_dir is None:
            skipped[model] = 'Run not supplied'
            continue
        folder = analysis_dir(run_dir, config)
        if not (folder/'results.json').is_file():
            if model == 'lstm':
                raise FileNotFoundError(f'LSTM evaluation missing: {folder}/results.json. Run notebook 06 on saved charts first.')
            skipped[model] = f'No scored results at {folder}'
            continue
        meta = json.loads((folder/'config.json').read_text())
        if meta.get('schema') != ANALYSIS_SCHEMA:
            raise ValueError(f'{model}: incompatible analysis schema; rescore saved charts with notebook 06 scoring-only recipe')
        # Model paths and plotting settings can vary; scoring/reference choices cannot.
        keys = ('FRAME_MS', 'ONSET_TOLERANCE_FRAMES', 'PATTERN_WINDOWS_FRAMES', 'PERCENTILE_BAND',
                'DIFFICULTY_FIELDS', 'RANDOM_BASELINE_SEEDS', 'HI_PRIMARY', 'MIN_CHARTS_PER_BAND_GROUP')
        scoring_config = {key: meta['config'][key] for key in keys}
        requested = json.loads(json.dumps({key: config[key] for key in keys}))
        if scoring_config != requested:
            raise ValueError(f'{model}: saved scoring settings differ from comparison CONFIG')
        if reference is None:
            reference, reference_config = meta['provenance'], scoring_config
        elif meta['provenance'] != reference or scoring_config != reference_config:
            raise ValueError('Models were scored with different reference humans or configurations')
        records = json.loads((folder/'results.json').read_text())
        ids = [r['chart_id'] for r in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f'{model}: duplicate chart IDs in results')
        models[model] = records
        sources[model] = str(folder)
    if 'lstm' not in models:
        raise ValueError('Supply an evaluated lstm run in CONFIG["MODEL_RUNS"]')
    random_folder = Path(sources['lstm'])/'random_baseline'
    if not (random_folder/'results.json').is_file():
        raise FileNotFoundError(f'Random baseline missing: {random_folder}. Complete notebook 06 first.')
    random_meta = json.loads((random_folder/'config.json').read_text())
    if random_meta.get('schema') != ANALYSIS_SCHEMA or random_meta['provenance'] != reference:
        raise ValueError('Random baseline reference provenance differs from model scores')
    for key in keys:
        if random_meta['config'][key] != reference_config[key]:
            raise ValueError('Random baseline configuration differs from model scores')
    models['random'] = json.loads((random_folder/'results.json').read_text())
    sources['random'] = str(random_folder)
    return models, dict(sources=sources, skipped_models=skipped, provenance=reference)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values; None hypotheses are not tested."""
    eligible = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    if any(not np.isfinite(p) or not 0 <= p <= 1 for _, p in eligible):
        raise ValueError('P-values must be finite values in [0,1] or None')
    ordered = sorted(eligible, key=lambda item: item[1])
    adjusted = [None] * len(p_values)
    running = 0.0
    for position, (index, p_value) in enumerate(ordered):
        running = max(running, (len(ordered)-position)*p_value)
        adjusted[index] = min(1.0, running)
    return adjusted


def hi_retention(metrics, variant, config):
    """Per-chart longest/shortest HI; undefined when shortest is zero/null.

    For random, metrics are already seed means: divide those means, rather than
    averaging individual seed retention ratios. Coverage's percentage cancels.
    """
    shortest, longest = min(config['PATTERN_WINDOWS_FRAMES']), max(config['PATTERN_WINDOWS_FRAMES'])
    suffix = 'hi_precision' if config['HI_PRIMARY'] == 'precision' else 'hi_coverage'
    short = metrics.get(f'pspace_{variant}_w{shortest}_{suffix}')
    long = metrics.get(f'pspace_{variant}_w{longest}_{suffix}')
    if short is None or long is None or not np.isfinite(short) or not np.isfinite(long) or short <= 0:
        return None
    return float(long / short)


def metric_units(metric):
    if metric.endswith(('_overall_ratio', '_hi_precision')) or metric.startswith('hi_retention_'):
        return 'ratio'
    if metric.endswith(('_hi_coverage', '_overall', '_human_overall')) or metric in ('DCHuman','OCHuman','Over_PSpace','HI_PSpace'):
        return 'percent'
    return ''


def compare_models(models, output_dir, config=None, source_metadata=None):
    """One primary retention test; every other test is in one Holm family.

    Primary: configured HI/variant retention, overall, paired LSTM vs Transformer
    independently of random coverage. All per-group and other metric tests are
    secondary. Descriptive tables/plots retain the common all-model cohort.
    """
    config = evaluation_config(config)
    version = config['PRIMARY_VERSION']
    primary_metric = 'hi_retention_' + config['PRIMARY_TEST_VARIANT']
    maps, retention_records = {}, []
    for name, records in models.items():
        mapping = {}
        for record in records:
            chart = record['chart_id']
            if chart in mapping:
                raise ValueError(f'{name}: duplicate chart IDs')
            row = {**record, version: dict(record[version])}
            for variant in ('binary', 'type'):
                row[version]['hi_retention_'+variant] = hi_retention(row[version], variant, config)
            mapping[chart] = row
            retention_records.append(dict(model=name, chart_id=chart, difficulty=row['difficulty'], version=version,
                                          **{'hi_retention_'+v: row[version]['hi_retention_'+v] for v in ('binary','type')}))
        maps[name] = mapping
    if not maps:
        raise ValueError('No model results supplied')
    common = sorted(set.intersection(*(set(m) for m in maps.values())))
    if not common:
        raise ValueError('No chart IDs are shared by the supplied models')
    for chart in common:
        if len({m[chart]['difficulty'] for m in maps.values()}) != 1:
            raise ValueError(f'{chart}: difficulty metadata differs between models')
    has_pair = {'lstm', 'donderformer'} <= maps.keys()
    primary_cohort = sorted(set(maps['lstm']) & set(maps['donderformer'])) if has_pair else []
    for chart in primary_cohort:
        if maps['lstm'][chart]['difficulty'] != maps['donderformer'][chart]['difficulty']:
            raise ValueError(f'{chart}: difficulty metadata differs between paired models')
    difficulty = {c: next(iter(maps.values()))[c]['difficulty'] for c in common}
    groups = [(None, common)] + [(d, [c for c in common if difficulty[c] == d]) for d in sorted(set(difficulty.values()))]
    rows, tests = [], []
    for group, ids in groups:
        metrics = sorted({metric for model in maps.values() for chart in ids for metric in model[chart][version]
                          if report_metric(metric) and metric not in DENSITY_METRICS})
        for metric in metrics:
            primary = group is None and metric == primary_metric
            cohort = primary_cohort if primary else ids
            valid_pairs = ([c for c in cohort if maps['lstm'][c][version].get(metric) is not None
                            and maps['donderformer'][c][version].get(metric) is not None] if has_pair else [])
            test = wilcoxon_signed_rank([maps['lstm'][c][version][metric] for c in valid_pairs],
                                       [maps['donderformer'][c][version][metric] for c in valid_pairs]) if has_pair else dict(
                                           statistic=None, p_value=None, n_pairs=0, n_nonzero=0, status='awaiting_donderformer')
            test.update(scope='overall' if group is None else 'difficulty', difficulty=group or '', metric=metric,
                        test_role='primary' if primary else 'secondary', p_value_raw=test['p_value'])
            if primary:
                test['paired_chart_ids'] = valid_pairs
            tests.append(test)
            # Primary model means use the primary paired cohort; random remains descriptive.
            for name, model in maps.items():
                descriptive_ids = (valid_pairs if primary and has_pair and name in ('lstm','donderformer') else ids)
                values = [float(model[c][version][metric]) for c in descriptive_ids if model[c][version].get(metric) is not None]
                rows.append(dict(scope=test['scope'], difficulty=group or '', model=name, version=version, metric=metric,
                                 n=len(values), mean=float(np.mean(values)) if values else None,
                                 std=float(np.std(values)) if values else None,
                                 p_value=test['p_value'] if name in ('lstm', 'donderformer') else None,
                                 n_pairs=test['n_pairs'] if name in ('lstm', 'donderformer') else 0,
                                 headline=metric_headline(metric, config), units=metric_units(metric), test_role=test['test_role']))
    secondary = [t for t in tests if t['test_role'] == 'secondary']
    adjusted = holm_adjust([t['p_value'] for t in secondary])
    for test, p_value in zip(secondary, adjusted):
        test['p_value_adjusted'] = p_value
        test['correction'] = 'Holm-Bonferroni'
    for test in tests:
        if test['test_role'] == 'primary':
            test['p_value_adjusted'] = test['p_value']
            test['correction'] = 'none (single pre-declared primary test)'
        p_value = test['p_value_adjusted']
        test['significant'] = p_value < config['SIGNIFICANCE_ALPHA'] if p_value is not None else None
    lookup = {(t['scope'],t['difficulty'],t['metric']):t for t in tests}
    for row in rows:
        test = lookup[(row['scope'],row['difficulty'],row['metric'])]
        is_model_pair = row['model'] in ('lstm','donderformer')
        row.update(p_value_raw=row['p_value'], p_value_adjusted=test['p_value_adjusted'] if is_model_pair else None,
                   correction=test['correction'] if is_model_pair else 'descriptive control',
                   significant=test['significant'] if is_model_pair else None)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scope', 'difficulty', 'model', 'version', 'metric', 'n', 'mean', 'std',
                'p_value', 'n_pairs', 'p_value_raw', 'p_value_adjusted', 'test_role', 'correction', 'significant', 'headline', 'units'])
        writer.writeheader(); writer.writerows(rows)
    primary_test = next(t for t in tests if t['test_role'] == 'primary')
    _write_json(output/'wilcoxon.json', tests)
    _write_json(output/'primary_test.json', primary_test)
    _write_json(output/'retention.json', retention_records)
    _write_json(output/'comparison.json', dict(config=config, common_chart_ids=common,
                excluded_chart_ids={n: sorted(set(m)-set(common)) for n, m in maps.items()},
                sources=source_metadata or {},
                primary_test=dict(metric=primary_metric, hi=config['HI_PRIMARY'], variant=config['PRIMARY_TEST_VARIANT'],
                                  shortest_window=min(config['PATTERN_WINDOWS_FRAMES']), longest_window=max(config['PATTERN_WINDOWS_FRAMES']),
                                  paired_chart_ids=primary_test['paired_chart_ids'] if has_pair else []),
                secondary_family_size=sum(t['p_value'] is not None for t in secondary),
                excluded_metrics=sorted(DENSITY_METRICS), density_note=RANDOM_DENSITY_NOTE,
                inference='One overall retention primary test; all other finite p-values form one Holm-Bonferroni family across metrics and difficulty groups'))
    plot_pattern_comparison(rows, output/'plots', config)
    return rows, tests


def plot_pattern_comparison(rows, plots_dir, config):
    import matplotlib.pyplot as plt

    plots = Path(plots_dir); plots.mkdir(parents=True, exist_ok=True)
    groups = sorted({(r['scope'], r['difficulty']) for r in rows})
    suffix = 'hi_precision' if config['HI_PRIMARY'] == 'precision' else 'hi_coverage'
    for index, (scope, difficulty) in enumerate(groups):
        fig, axes = plt.subplots(1, len(('binary', 'type')), figsize=config['FIGURE_SIZE'], constrained_layout=True)
        selected = [r for r in rows if r['scope'] == scope and r['difficulty'] == difficulty]
        for ax, variant in zip(axes, ('binary', 'type')):
            windows = sorted(config['PATTERN_WINDOWS_FRAMES'])
            for model in sorted({r['model'] for r in selected}):
                mapping = {r['metric']: r for r in selected if r['model'] == model}
                values = [mapping.get(f'pspace_{variant}_w{w}_{suffix}', {}).get('mean') for w in windows]
                ax.plot(windows, [np.nan if v is None else v for v in values], marker='o', label=model)
            units = 'fraction; empty patterns excluded' if config['HI_PRIMARY'] == 'precision' else '% of pooled patterns'
            ax.set(title=f'{variant}: {difficulty or "all star groups"}', xlabel='Pattern window (frames)',
                   ylabel=f'HI {config["HI_PRIMARY"]} ({units})')
            ax.set_xticks(windows); ax.legend(); ax.grid(alpha=config['PLOT_GRID_ALPHA'])
        filename = 'hi_pspace.png' if scope == 'overall' else f'hi_pspace_difficulty_{index}.png'
        fig.savefig(plots/filename, dpi=config['PLOT_DPI'], bbox_inches='tight')
        plt.close(fig)
