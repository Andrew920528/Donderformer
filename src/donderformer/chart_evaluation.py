"""Model-independent scoring of frame charts; no checkpoint or inference required.

Extended metrics live alongside the legacy files in analysis/. Pattern coverage
uses pooled test humans; training humans alone supply difficulty bands and types.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from donderformer.evaluation import (
    calculate_f1_metrics, calculate_paper_metrics, note_starts, remove_adjacent_notes,
)
from donderformer.evaluation_config import evaluation_config
from donderformer.difficulty_groups import merge_star_groups, load_group_metadata, print_group_report, save_group_report

ANALYSIS_SCHEMA = "hi-precision-v2"
DENSITY_METRICS = {"note_density", "note_density_in_band"}
RANDOM_DENSITY_NOTE = "Random density is excluded because the control fixes the human note count; its active span may still differ."

LEGACY_METRICS = (
    'onset_precision', 'onset_recall', 'onset_f1', 'joint_precision',
    'joint_recall', 'joint_f1', 'DCHuman', 'OCHuman', 'Over_PSpace', 'HI_PSpace',
)
VARIANTS = ('binary', 'type')
VERSIONS = ('raw', 'clean')


def difficulty_statistics(sequence, frame_ms):
    """Undefined density (<2 starts) or Don/Ka (no Ka) is None, never infinity."""
    starts = note_starts(sequence)
    positions = np.flatnonzero(starts)
    duration = ((positions[-1] - positions[0]) * frame_ms / 1000
                if len(positions) > 1 else None)
    don = np.count_nonzero(np.isin(starts, [1, 3]))
    ka = np.count_nonzero(np.isin(starts, [2, 4]))
    return {'note_starts': int(len(positions)),
            'note_density': float(len(positions) / duration) if duration else None,
            'don_ka_ratio': float(don / ka) if ka else None}


def pattern_set(sequence, window, variant):
    """Bytes encode exact patterns compactly, including all-zero patterns."""
    starts = note_starts(sequence)
    if variant == 'binary':
        starts = starts != 0
    elif variant != 'type':
        raise ValueError(f'Unknown pattern variant: {variant}')
    if not isinstance(window, int) or window <= 0:
        raise ValueError('Pattern windows must be positive integers')
    if len(starts) < window:
        return set()
    encoded = starts.astype(np.uint8)
    return {row.tobytes() for row in np.lib.stride_tricks.sliding_window_view(encoded, window)}


def pattern_metrics(human, predicted, pools, difficulty, config):
    """Primary HI precision excludes the empty pattern; coverage is retained.

    Existing normalized percentages/aliases remain in per-chart records only.
    New tables headline unique-count ratios, avoiding vanishing long-window
    percentages. HI precision and hi_primary are fractions, coverage is percent.
    """
    result = {}
    for variant in VARIANTS:
        alphabet = 2 if variant == 'binary' else 7  # format constants
        for window in config['PATTERN_WINDOWS_FRAMES']:
            prefix = f'pspace_{variant}_w{window}'
            human_patterns = pattern_set(human, window, variant)
            ai_patterns = pattern_set(predicted, window, variant)
            reference = pools[(difficulty, window, variant)]
            valid = len(predicted) >= window
            nonempty = ai_patterns - {bytes(window)}
            coverage = len(ai_patterns & reference) / len(reference) * 100 if reference and valid else None
            precision = len(nonempty & reference) / len(nonempty) if nonempty else None
            ratio = len(ai_patterns) / len(human_patterns) if human_patterns else None
            result.update({
                prefix+'_overall': len(ai_patterns) / alphabet**window * 100 if valid else None,
                prefix+'_human_overall': len(human_patterns) / alphabet**window * 100 if human_patterns else None,
                prefix+'_ratio': ratio,
                prefix+'_overall_ratio': ratio,
                prefix+'_hi': coverage,  # compatibility: percentage, including empty pattern
                prefix+'_hi_coverage': coverage,
                prefix+'_hi_precision': precision,
                prefix+'_hi_primary': (precision if config['HI_PRIMARY'] == 'precision'
                                      else coverage / 100 if coverage is not None else None),
            })
    return result


@dataclass
class ReferenceContext:
    config: dict
    metadata: dict
    test_humans: dict
    pools: dict
    bands: dict
    type_probabilities: dict
    provenance: dict
    grouping: dict


def build_reference_context(test_humans, train_humans, metadata, config=None, provenance=None):
    """Build pools once from ALL test humans and bands/types from train only.

    Inputs map chart_id to integer frame sequences; metadata supplies difficulty.
    This entry point also supports synthetic tests without feature-cache files.
    """
    config = evaluation_config(config)
    if not test_humans or not train_humans:
        raise ValueError('Both test and training human labels are required for reference metrics')
    if set(test_humans) & set(train_humans):
        raise ValueError('Training and test chart IDs overlap; reference leakage is not allowed')
    metadata, grouping = merge_star_groups(metadata, train_humans, test_humans,
                                           config['MIN_CHARTS_PER_BAND_GROUP'])
    pools, statistics, type_counts = {}, {}, {}
    for chart_id, human in test_humans.items():
        difficulty = metadata[chart_id]['difficulty']
        for window in config['PATTERN_WINDOWS_FRAMES']:
            for variant in VARIANTS:
                pools.setdefault((difficulty, window, variant), set()).update(pattern_set(human, window, variant))
    for chart_id, human in train_humans.items():
        difficulty = metadata[chart_id]['difficulty']
        statistics.setdefault(difficulty, []).append(difficulty_statistics(human, config['FRAME_MS']))
        counts = np.bincount(note_starts(human), minlength=7)[1:]
        type_counts.setdefault(difficulty, np.zeros(6, dtype=np.int64))
        type_counts[difficulty] += counts
    bands = {}
    for difficulty, rows in statistics.items():
        bands[difficulty] = {}
        for metric in ('note_density', 'don_ka_ratio'):
            values = [r[metric] for r in rows if r[metric] is not None]
            bounds = np.percentile(values, config['PERCENTILE_BAND']).tolist() if values else [None, None]
            bands[difficulty][metric] = dict(lower=bounds[0], upper=bounds[1], n=len(values))
    probabilities = {}
    for difficulty, counts in type_counts.items():
        probabilities[difficulty] = counts / counts.sum() if counts.sum() else None
    for difficulty in {metadata[c]['difficulty'] for c in test_humans}:
        if difficulty not in bands:
            raise ValueError(f'No training human charts for test difficulty {difficulty!r}; cannot fit reference bands')
        if probabilities[difficulty] is None:
            raise ValueError(f'No training note starts for difficulty {difficulty!r}; cannot sample random types')
    provenance = dict(provenance or {})
    provenance.update(difficulty_groups=grouping, train_chart_ids=sorted(train_humans), test_chart_ids=sorted(test_humans),
                      human_sha256={c: hashlib.sha256(np.asarray(s, dtype=np.uint8).tobytes()).hexdigest()
                                    for c, s in test_humans.items()},
                      train_label_sha256={c: hashlib.sha256(np.asarray(s, dtype=np.uint8).tobytes()).hexdigest()
                                          for c, s in train_humans.items()},
                      difficulties={c: metadata[c]['difficulty'] for c in sorted(set(test_humans) | set(train_humans))})
    return ReferenceContext(config, metadata, test_humans, pools, bands, probabilities, provenance, grouping)


def load_reference_context(paths, split, extractor, config=None):
    """Read cached labels and original metadata; never extract or modify a split.

    oni_v1's level column is blank. Recover LEVEL from its matching TJA course,
    as ChartExporter does. Sparse Oni star groups merge using training counts.
    """
    from donderformer.data.data_splitter import manifest_hash
    from dataclasses import asdict

    config = evaluation_config(config)
    if extractor.frame_ms != config['FRAME_MS']:
        raise ValueError('FRAME_MS disagrees with the saved feature extractor')
    cache_config = extractor.dir(paths) / 'config.json'
    if cache_config.exists() and json.loads(cache_config.read_text()) != asdict(extractor):
        raise ValueError(f'Feature cache config mismatch: {cache_config}')
    failed = extractor.failed_charts(paths, split)
    metadata, parts, _ = load_group_metadata(paths, split, config, failed)
    sequences = {'train': {}, 'test': {}}
    for part, ids in parts.items():
        for chart_id in ids:
            row = metadata[chart_id]
            label_path = extractor.labels_path(paths, row['audio_id'], chart_id)
            if not label_path.is_file():
                raise FileNotFoundError(f'Human reference labels missing: {label_path}. Restore training AND test labels and any extractor failure list; no extraction is performed.')
            with label_path.open('rb') as stream:
                labels = np.load(stream)
            if labels.ndim != 2 or labels.shape[1] != 7 or not np.isin(labels, [0, 1]).all() or not np.all(labels.sum(axis=1) == 1):
                raise ValueError(f'Invalid seven-class one-hot labels: {label_path}')
            sequences[part][chart_id] = labels.argmax(axis=1)
    context = build_reference_context(sequences['test'], sequences['train'], metadata, config,
                                     dict(split=split, manifest_sha256=manifest_hash(paths, split),
                                          extractor=asdict(extractor), skipped_chart_ids=sorted(failed)))
    context.grouping.update(split=split, manifest_sha256=manifest_hash(paths, split),
                            skipped_chart_ids=sorted(failed), counts_basis='Cached label charts after applying the extractor failure list')
    print_group_report(context.grouping)
    return context


def score_chart(human, predicted, chart_id, context):
    human, predicted = np.asarray(human), np.asarray(predicted)
    if human.shape != predicted.shape or not len(human):
        raise ValueError(f'{chart_id}: human and prediction must have equal, nonzero frame lengths')
    difficulty = context.metadata[chart_id]['difficulty']
    result = {**calculate_f1_metrics(human, predicted, context.config['ONSET_TOLERANCE_FRAMES']),
              **calculate_paper_metrics(human, predicted),
              **pattern_metrics(human, predicted, context.pools, difficulty, context.config),
              **difficulty_statistics(predicted, context.config['FRAME_MS'])}
    for metric in ('note_density', 'don_ka_ratio'):
        band = context.bands[difficulty][metric]
        value = result[metric]
        result[metric+'_in_band'] = (band['lower'] <= value <= band['upper']
                                    if value is not None and band['lower'] is not None else None)
    return result


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def report_metric(metric):
    """Compatibility keys stay in JSON; summaries use explicit current labels."""
    if not metric.startswith('pspace_'):
        return True
    if metric.endswith(('_hi', '_ratio')) and not metric.endswith('_overall_ratio'):
        return False
    if metric.endswith(('_overall', '_human_overall')):
        return metric.startswith('pspace_binary_w8_')
    return not metric.endswith('_hi_primary')  # alias is for configured plots/retention


def metric_headline(metric, config):
    if metric.endswith('_hi_precision'):
        return config['HI_PRIMARY'] == 'precision'
    if metric.endswith('_hi_coverage'):
        return config['HI_PRIMARY'] == 'coverage'
    if metric.startswith('pspace_') and metric.endswith(('_overall', '_human_overall')):
        return False  # legacy binary-eight-frame normalized percentage
    return True


def summarize_records(records, exclude_metrics=(), config=None):
    """Chart means and population SD; booleans/fractional flags give in-band share.

    Missing/undefined values are excluded and n is written for each metric.
    """
    config = evaluation_config(config)
    summaries = []
    difficulties = sorted({r['difficulty'] for r in records})
    for difficulty in [None, *difficulties]:
        selected = records if difficulty is None else [r for r in records if r['difficulty'] == difficulty]
        for version in VERSIONS:
            metrics = sorted({m for r in selected for m in r[version] if report_metric(m) and m not in exclude_metrics})
            for metric in metrics:
                values = [float(r[version][metric]) for r in selected if r[version].get(metric) is not None]
                summaries.append(dict(scope='overall' if difficulty is None else 'difficulty',
                                      difficulty=difficulty or '', version=version, metric=metric, n=len(values),
                                      mean=float(np.mean(values)) if values else None,
                                      std=float(np.std(values)) if values else None,
                                      headline=metric_headline(metric, config)))
    return summaries


def _write_summary(path, records, exclude_metrics=(), config=None):
    rows = summarize_records(records, exclude_metrics, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scope', 'difficulty', 'version', 'metric', 'n', 'mean', 'std', 'headline'])
        writer.writeheader(); writer.writerows(rows)


def evaluate_saved_charts(charts_dir, context, output_dir=None, *, config=None):
    """Score any model's <chart_id>.npz (human, generated; clean optional).

    Write compatible results.json/summary.csv plus extended analysis/ outputs.
    Read-only with respect to chart files. Pools always come from the complete
    context, never just the intersection of model-generated chart directories.
    """
    charts_dir = Path(charts_dir)
    files = sorted(charts_dir.glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No saved chart .npz files in {charts_dir}; generate or restore charts first')
    if config is not None and evaluation_config(config) != context.config:
        raise ValueError('Scoring configuration differs from the reference context; rebuild references')
    records = []
    for path in files:
        chart_id = path.stem
        if chart_id not in context.test_humans:
            raise ValueError(f'{chart_id}: generated chart is not in the reference test split')
        with np.load(path, allow_pickle=False) as arrays:
            if not {'human', 'generated'} <= set(arrays.files):
                raise ValueError(f'{path}: expected human and generated keys')
            human = arrays['human']
            generated = arrays['generated']
            clean = arrays['clean'] if 'clean' in arrays.files else remove_adjacent_notes(generated)
        note_starts(human); note_starts(generated); note_starts(clean)
        if not np.array_equal(human, context.test_humans[chart_id]):
            raise ValueError(f'{chart_id}: saved human chart disagrees with the original test labels')
        records.append(dict(song_id=chart_id, chart_id=chart_id, difficulty=context.metadata[chart_id]['difficulty'],
                            raw=score_chart(human, generated, chart_id, context),
                            clean=score_chart(human, clean, chart_id, context)))
    output = Path(output_dir) if output_dir is not None else charts_dir.parent
    # Preserve the original list/song_id/raw/clean schema and ten-metric means.
    legacy = [dict(song_id=r['song_id'], **{version: {k: v for k, v in r[version].items()
                    if k in (*LEGACY_METRICS, 'onset_matches', 'joint_matches', 'human_onsets', 'ai_onsets')}
                    for version in VERSIONS}) for r in records]
    _write_json(output/'results.json', legacy)
    with (output/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['version', 'metric', 'mean'])
        writer.writeheader()
        writer.writerows(dict(version=version, metric=m, mean=float(np.mean([r[version][m] for r in records])))
                         for version in VERSIONS for m in LEGACY_METRICS)
    analysis = output/'analysis'
    _write_json(analysis/'results.json', records)
    _write_json(analysis/'config.json', dict(schema=ANALYSIS_SCHEMA, config=context.config, provenance=context.provenance,
                                           evaluated_chart_ids=[r['chart_id'] for r in records],
                                           pool_sizes=[dict(difficulty=d, window=w, variant=v, count=len(p))
                                                       for (d, w, v), p in context.pools.items()],
                                           bands=context.bands,
                                           training_type_probabilities={d:p.tolist() for d,p in context.type_probabilities.items()}))
    save_group_report(analysis, context.grouping)
    print_group_report(context.grouping)
    _write_summary(analysis/'summary.csv', records, config=context.config)
    write_micro_summary(analysis/'micro_summary.csv', records)
    return records


def random_chart(human, probabilities, seed, chart_id):
    """Uniform frame subset, training types; deterministic across directory order.

    Adjacent identical held labels would merge. For that one collision case,
    resample the type from the training distribution conditional on another
    class. This preserves the exact count of representable note starts. If no
    other class has mass, fail rather than silently reduce the note count.
    """
    human = note_starts(human)
    count = np.count_nonzero(human)
    digest = hashlib.sha256(chart_id.encode('utf-8')).digest()
    entropy = [int(seed), *np.frombuffer(digest, dtype='<u4').tolist()]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    positions = np.sort(rng.choice(len(human), size=count, replace=False))
    result = np.zeros_like(human)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (6,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or probabilities.sum() <= 0:
        raise ValueError('Random baseline needs a finite nonnegative six-note training distribution')
    probabilities = probabilities / probabilities.sum()
    for position in positions:
        weights = probabilities.copy()
        if position > 0 and result[position-1] in (5, 6):
            weights[result[position-1]-1] = 0
        if not weights.sum():
            raise ValueError('Training types cannot represent adjacent random starts without merging held notes')
        result[position] = rng.choice(np.arange(1, 7), p=weights/weights.sum())
    assert np.count_nonzero(note_starts(result)) == count
    return result


def evaluate_random_baseline(context, output_dir, chart_ids=None):
    """Average metrics/flags per chart across seeds, then summarize across charts."""
    ids = sorted(context.test_humans if chart_ids is None else chart_ids)
    records, seed_records = [], []
    for chart_id in ids:
        human = context.test_humans[chart_id]
        difficulty = context.metadata[chart_id]['difficulty']
        samples = []
        for seed in context.config['RANDOM_BASELINE_SEEDS']:
            generated = random_chart(human, context.type_probabilities[difficulty], seed, chart_id)
            record = dict(chart_id=chart_id, song_id=chart_id, difficulty=difficulty, seed=seed,
                          raw=score_chart(human, generated, chart_id, context),
                          clean=score_chart(human, remove_adjacent_notes(generated), chart_id, context))
            samples.append(record); seed_records.append(record)
        record = dict(chart_id=chart_id, song_id=chart_id, difficulty=difficulty)
        for version in VERSIONS:
            record[version] = {}
            for metric in samples[0][version]:
                values = [s[version][metric] for s in samples if s[version][metric] is not None]
                record[version][metric] = float(np.mean(values)) if values else None
        records.append(record)
    output_dir = Path(output_dir)
    _write_json(output_dir/'results.json', records)
    _write_json(output_dir/'seed_results.json', seed_records)
    _write_json(output_dir/'config.json', dict(schema=ANALYSIS_SCHEMA, config=context.config, provenance=context.provenance,
                                             bands=context.bands, training_type_probabilities={d:p.tolist() for d,p in context.type_probabilities.items()}))
    _write_summary(output_dir/'summary.csv', records, DENSITY_METRICS, context.config)
    _write_json(output_dir/'summary_notes.json', dict(excluded_metrics=sorted(DENSITY_METRICS), explanation=RANDOM_DENSITY_NOTE))
    save_group_report(output_dir, context.grouping)
    return records


def write_micro_summary(path, records):
    """Count-aggregate F1 in addition to the chart-mean summaries (all/difficulty)."""
    from donderformer.evaluation import score_from_counts
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = [(None, records)] + [(d, [r for r in records if r['difficulty'] == d])
                                 for d in sorted({r['difficulty'] for r in records})]
    rows = []
    for difficulty, selected in groups:
        for version in VERSIONS:
            actual = sum(r[version]['human_onsets'] for r in selected)
            predicted = sum(r[version]['note_starts'] for r in selected)
            for metric, key in [('onset', 'onset_matches'), ('joint', 'joint_matches')]:
                matches = sum(r[version][key] for r in selected)
                precision, recall, f1 = score_from_counts(matches, predicted, actual)
                rows.append(dict(scope='overall' if difficulty is None else 'difficulty',
                                 difficulty=difficulty or '', version=version, metric=metric,
                                 precision=precision, recall=recall, f1=f1,
                                 true_starts=actual, predicted_starts=predicted, matches=matches))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scope', 'difficulty', 'version', 'metric',
                                              'precision', 'recall', 'f1', 'true_starts', 'predicted_starts', 'matches'])
        writer.writeheader(); writer.writerows(rows)
    return rows
