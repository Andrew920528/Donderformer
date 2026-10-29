"""Oni star-level metadata and deterministic training-count-based group merging."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path

from donderformer.evaluation_config import evaluation_config


def merge_star_groups(metadata, train_chart_ids, test_chart_ids, minimum):
    """Merge sparse groups toward the nearest adjacent star-level boundary.

    Process the lowest-star sparse group first. Distance ties select the lower
    neighbor. Only training counts determine merges; test counts are reporting
    data. If even all training charts cannot reach minimum, keep a single group
    and explicitly mark meets_minimum=False rather than discarding the data.
    """
    if not isinstance(minimum, int) or minimum <= 0:
        raise ValueError('Group minimum must be a positive integer')
    train_ids, test_ids = set(train_chart_ids), set(test_chart_ids)
    if train_ids & test_ids:
        raise ValueError('Training and test chart IDs overlap')
    rows = deepcopy(metadata)
    for chart_id in train_ids | test_ids:
        row = rows[chart_id]
        if row.get('course', 'oni') != 'oni':
            raise ValueError(f'{chart_id}: this evaluation expects Oni charts only')
        try:
            level = float(row['level'])
            if not level.is_integer() or level <= 0:
                raise ValueError()
            row['level'] = int(level)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise ValueError(f'{chart_id}: a positive integer star level is required') from None
    train = Counter(rows[c]['level'] for c in train_ids)
    test = Counter(rows[c]['level'] for c in test_ids)
    levels = sorted(set(train) | set(test))
    groups = [[level] for level in levels]
    history = []
    while len(groups) > 1:
        sparse = next((i for i, g in enumerate(groups) if sum(train[l] for l in g) < minimum), None)
        if sparse is None:
            break
        group = groups[sparse]
        neighbors = [i for i in (sparse-1, sparse+1) if 0 <= i < len(groups)]
        def distance(i):
            neighbor = groups[i]
            gap = group[0]-neighbor[-1] if i < sparse else neighbor[0]-group[-1]
            return gap, neighbor[0]
        adjacent = min(neighbors, key=distance)
        merged = sorted(group + groups[adjacent])
        history.append(dict(source=group.copy(), neighbor=groups[adjacent].copy(), merged=merged.copy(),
                            train_charts=sum(train[l] for l in merged)))
        low, high = sorted((sparse, adjacent))
        groups[low:high+1] = [merged]
    mapping, report_groups = {}, []
    for group in groups:
        label = f'stars={group[0]}' if len(group) == 1 else f'stars={group[0]}-{group[-1]}'
        for level in group:
            mapping[str(level)] = label
        count = sum(train[l] for l in group)
        report_groups.append(dict(difficulty=label, levels=group, train_charts=count,
                                  test_charts=sum(test[l] for l in group), meets_minimum=count >= minimum))
    for chart_id in train_ids | test_ids:
        row = rows[chart_id]
        row['original_difficulty'] = row.get('original_difficulty', f"stars={row['level']}")
        row['difficulty'] = mapping[str(row['level'])]
    report = dict(dataset='Oni only; difficulty means star level within Oni',
                  minimum_training_charts=minimum, mapping=mapping, groups=report_groups,
                  original_groups=[dict(level=l, train_charts=train[l], test_charts=test[l]) for l in levels],
                  merge_history=history, policy='Lowest sparse group first; nearest boundary; ties toward lower stars')
    return rows, report


def load_group_metadata(paths, split, config=None, failed_chart_ids=()):
    """Resolve original star levels without reading/extracting any feature arrays."""
    from donderformer.data.data_splitter import read_manifest, manifest_hash
    from donderformer.data.tja import course_sections

    config = evaluation_config(config)
    failed = set(failed_chart_ids)
    metadata, parts = {}, {'train': [], 'test': []}
    for source in read_manifest(paths, split):
        if source['split'] not in parts or source['chart_id'] in failed:
            continue
        row = dict(source)
        if row.get('course') != 'oni':
            raise ValueError(f"{row['chart_id']}: this evaluation expects Oni charts only")
        if not row.get('level'):
            tja = paths.raw / row['tja']
            if not tja.is_file():
                raise FileNotFoundError(f'LEVEL missing in manifest and TJA unavailable: {tja}. Restore raw chart metadata.')
            sections = [s for s in course_sections(tja.read_text(encoding='utf-8-sig')) if s.course == row['course']]
            if len(sections) != 1 or sections[0].level is None:
                raise ValueError(f"Cannot resolve a unique LEVEL for {row['chart_id']}")
            row['level'] = sections[0].level
        metadata[row['chart_id']] = row
        parts[row['split']].append(row['chart_id'])
    metadata, report = merge_star_groups(metadata, parts['train'], parts['test'], config['MIN_CHARTS_PER_BAND_GROUP'])
    report.update(split=split, manifest_sha256=manifest_hash(paths, split), skipped_chart_ids=sorted(failed))
    return metadata, parts, report


def print_group_report(report):
    print('Oni star groups: training / test chart counts')
    for group in report['groups']:
        warning = '' if group['meets_minimum'] else ' (total training data below configured minimum)'
        print(f"  {group['difficulty']}: {group['train_charts']} / {group['test_charts']}{warning}")


def save_group_report(directory, report):
    """Save both final mapping and original/merged counts under analysis/."""
    import csv
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/'difficulty_groups.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (directory/'difficulty_groups.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['difficulty', 'levels', 'train_charts', 'test_charts', 'meets_minimum'])
        writer.writeheader()
        writer.writerows({**g, 'levels': ','.join(map(str, g['levels']))} for g in report['groups'])
