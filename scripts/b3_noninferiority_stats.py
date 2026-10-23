"""task42 statistics: per-family and pooled B3-vs-A paired non-inferiority.

The whole game is the cluster. The primary interval is a t interval over the 64 game
means; the bootstrap resamples whole games 10000 times (ratio of summed deltas to summed
points). Both must have a 95% lower bound above -3pp for a family to pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

import numpy as np  # noqa: E402
from b3_noninferiority import FAMILIES, MARGIN, correctness_difference, key  # noqa: E402
from teacher_audit_stats import t_critical  # noqa: E402


def counts(row):
    arms = row['arms']
    same = key(arms['A']['chosen']) == key(arms['B3']['chosen'])
    assert row['judge_required'] == (not same)
    assert (row['heavy'] is None) == same
    label = row['heavy']['label'] if row['heavy'] else None
    delta = correctness_difference(arms['A']['chosen'], arms['B3']['chosen'], label)
    assert row['delta'] == delta
    out = {'points': 1, 'delta': delta, 'identical': int(same), 'judged': int(not same),
           'only_A_correct': int(delta == -1), 'only_B3_correct': int(delta == 1),
           'disagreed_both_wrong': int(not same and delta == 0),
           'truth_continuations': row['heavy']['truth_continuations'] if row['heavy'] else 0,
           'heavy_verification_continuations':
               row['heavy']['verification_continuations'] if row['heavy'] else 0}
    for name in ('A', 'B3'):
        a = arms[name]
        assert a['continuations'] == a['halving_continuations'] + a['verification_continuations']
        assert abs(a['cpu']['total_s'] - a['cpu']['worker']['total_s']
                   - a['cpu']['simulator']['total_s']) < 1e-9
        out[name + '_cost'] = a['continuations']
        out[name + '_halving'] = a['halving_continuations']
        out[name + '_verify'] = a['verification_continuations']
        out[name + '_cpu'] = a['cpu']['total_s']
        out[name + '_worker_cpu'] = a['cpu']['worker']['total_s']
        out[name + '_sim_cpu'] = a['cpu']['simulator']['total_s']
        out[name + '_candidates'] = len(a['candidates'])
    return out


def summarize(games, formal=True):
    if len(games) < 2 or any(g['points'] <= 0 for g in games):
        raise ValueError('need at least two non-empty games')
    names = sorted(set().union(*(set(g) for g in games)))
    arrays = {n: np.asarray([g.get(n, 0) for g in games], dtype=float) for n in names}
    means = arrays['delta'] / arrays['points']
    estimate = float(means.mean())
    se = float(means.std(ddof=1) / math.sqrt(len(means)))
    critical = t_critical(len(means) - 1)
    ci = [estimate - critical * se, estimate + critical * se]
    picks = np.random.default_rng(20261023).integers(0, len(games), size=(10000, len(games)))

    def ratio(num, den):
        assert den.sum() > 0
        boot = num[picks].sum(axis=1) / den[picks].sum(axis=1)
        return {'estimate': float(num.sum() / den.sum()),
                'game_cluster_ci95': np.quantile(boot, [.025, .975]).tolist()}

    metrics = {n: ratio(v, arrays['points']) for n, v in arrays.items() if n != 'points'}
    for cost in ('cost', 'cpu', 'worker_cpu', 'sim_cpu', 'candidates'):
        metrics['B3_to_A_' + cost + '_ratio'] = ratio(arrays['B3_' + cost], arrays['A_' + cost])
        v = metrics['B3_to_A_' + cost + '_ratio']
        metrics['B3_' + cost + '_saving'] = {
            'estimate': 1 - v['estimate'],
            'game_cluster_ci95': [1 - v['game_cluster_ci95'][1], 1 - v['game_cluster_ci95'][0]]}
    passed = ci[0] > -MARGIN
    bootstrap_passed = metrics['delta']['game_cluster_ci95'][0] > -MARGIN
    return {'games': len(games), 'points': int(arrays['points'].sum()),
            'formal_final_analysis': formal,
            'primary': {'difference_B3_minus_A': estimate, 'se_of_game_means': se,
                        'df': len(games) - 1, 't_critical': critical, 'ci95': ci,
                        'NI_margin': MARGIN,
                        'noninferiority_passed': bool(passed) if formal else None,
                        'bootstrap_agrees_with_NI': bool(bootstrap_passed) if formal else None,
                        'robust_NI': bool(passed and bootstrap_passed) if formal else None},
            'metrics': metrics, 'totals': {n: float(v.sum()) for n, v in arrays.items()},
            'judged': int(arrays['judged'].sum()), 'identical': int(arrays['identical'].sum()),
            'only_A_correct': int(arrays['only_A_correct'].sum()),
            'only_B3_correct': int(arrays['only_B3_correct'].sum()),
            'disagreed_both_wrong': int(arrays['disagreed_both_wrong'].sum()),
            'bootstrap_resamples': 10000, 'bootstrap_seed': 20261023,
            'absolute_accuracy': ('Not estimated: identical-choice roots cancel exactly but are '
                                  'deliberately unlabelled. The paired difference is exact over all '
                                  'sampled roots.')}


def collect(out):
    metas = [json.loads(p.read_text()) for p in sorted((out / 'trajectories').glob('*/*/meta.json'))]
    expected = {(m['family'], m['seed'], p['decision']) for m in metas for p in m['points']}
    rows = [json.loads(p.read_text()) for p in sorted((out / 'points').glob('*.json'))]
    seen = {(r['family'], r['seed'], r['decision']) for r in rows}
    if len(rows) != len(expected) or seen != expected:
        missing = sorted(expected - seen)[:5]
        extra = sorted(seen - expected)[:5]
        raise SystemExit(f'points do not match trajectories: {len(rows)} rows, '
                         f'{len(expected)} expected, missing {missing}, extra {extra}')
    return metas, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    metas, rows = collect(args.data)
    by_family = defaultdict(list)
    for row in rows:
        by_family[row['family']].append(row)
    report = {'families': {}, 'formal_final_analysis': True,
              'expected': {'families': [f['name'] for f in FAMILIES], 'games_per_family': 64,
                           'points_per_family': 512, 'total_games': 256, 'total_points': 2048},
              'observed': {'games': len(metas), 'points': len(rows),
                           'games_per_family': {name: len({r['seed'] for r in rows
                                                          if r['family'] == name})
                                                for name in {r['family'] for r in rows}}}}
    all_games = []
    for family in FAMILIES:
        name = family['name']
        if name not in by_family:
            continue
        grouped = defaultdict(lambda: defaultdict(float))
        for row in by_family[name]:
            for k, v in counts(row).items():
                grouped[row['seed']][k] += v
        seeds = sorted(grouped)
        games = [grouped[s] for s in seeds]
        all_games.extend(games)
        part = summarize(games, formal=len(seeds) == 64)
        part['seeds'] = [seeds[0], seeds[-1]]
        report['families'][name] = part
    if all_games:
        report['pooled'] = summarize(all_games, formal=True)
    summary_path = args.data / 'compare_summary.json'
    compared = json.loads(summary_path.read_text()) if summary_path.exists() else []
    report['strict_replay_frames'] = int(sum(r['strict_replay_frames'] for r in compared))
    report['terminal_trajectories'] = int(sum(m['terminal'] for m in metas))
    report['total_actual_continuations'] = float(
        sum(report['pooled']['totals'][n] for n in
            ('A_cost', 'B3_cost', 'truth_continuations', 'heavy_verification_continuations')))
    families = [v for v in report['families'].values() if v['formal_final_analysis']]
    report['all_families_formal'] = len(families) == len(FAMILIES)
    report['all_families_robust_NI'] = (report['all_families_formal']
                                        and all(v['primary']['robust_NI'] for v in families))
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, v in report['families'].items():
        print(json.dumps({'family': name, 'games': v['games'], 'points': v['points'],
                          'primary': v['primary']}))
    print(json.dumps({'all_families_formal': report['all_families_formal'],
                      'all_families_robust_NI': report['all_families_robust_NI'],
                      'pooled': report.get('pooled', {}).get('primary')}))


if __name__ == '__main__':
    main()
