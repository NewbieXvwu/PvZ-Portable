"""task42 round-2 statistics: per-family, per-k B4/B5-vs-A paired non-inferiority.

The whole game is the cluster. For each family and each k in {4, 5} the primary interval
is a t interval over the 64 game means; the bootstrap resamples whole games 10000 times
(ratio of summed deltas to summed points). Both 95% lower bounds must be above -3pp for
that k to pass in that family; a k passes overall only if both families pass. Pooled
numbers are reported as reference only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

import numpy as np  # noqa: E402
from b3_noninferiority import MARGIN, key  # noqa: E402
from b3_round2_arms import FAMILIES, SEED_BASE, TOPKS  # noqa: E402
from teacher_audit_stats import t_critical  # noqa: E402

ARMS = ('A',) + tuple(TOPKS)


def counts(row):
    arms = row['arms']
    same = len({key(arms[a]['chosen']) for a in ARMS}) == 1
    assert row['judge_required'] == (not same)
    assert (row['heavy'] is None) == same
    for k in TOPKS:
        if same:
            assert row['delta_' + k] == 0
    label = row['heavy']['label'] if row['heavy'] else None
    deltas = {k: correctness_difference(arms['A']['chosen'], arms[k]['chosen'], label)
              for k in TOPKS}
    for k, d in deltas.items():
        assert row['delta_' + k] == d
    out = {'points': 1, 'identical': int(same), 'judged': int(not same)}
    for k in TOPKS:
        d = deltas[k]
        out['delta_' + k] = d
        out['only_A_correct_' + k] = int(d == -1)
        out['only_B' + k[1:] + '_correct'] = int(d == 1)
        out['disagreed_both_wrong_' + k] = int(not same and d == 0)
    out['truth_continuations'] = row['heavy']['truth_continuations'] if row['heavy'] else 0
    out['heavy_verification_continuations'] = (
        row['heavy']['verification_continuations'] if row['heavy'] else 0)
    for name in ARMS:
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


def summarize(games, k, formal=True):
    if len(games) < 2 or any(g['points'] <= 0 for g in games):
        raise ValueError('need at least two non-empty games')
    names = sorted(set().union(*(set(g) for g in games)))
    arrays = {n: np.asarray([g.get(n, 0) for g in games], dtype=float) for n in names}
    means = arrays['delta_' + k] / arrays['points']
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

    metrics = {n: ratio(v, arrays['points']) for n, v in arrays.items()
               if n != 'points' and not n.endswith('_cost')}
    for cost in ('cost', 'cpu', 'worker_cpu', 'sim_cpu', 'candidates'):
        metrics['B' + k[1:] + '_to_A_' + cost + '_ratio'] = ratio(arrays['B' + k[1:] + '_' + cost],
                                                                  arrays['A_' + cost])
        v = metrics['B' + k[1:] + '_to_A_' + cost + '_ratio']
        metrics['B' + k[1:] + '_' + cost + '_saving'] = {
            'estimate': 1 - v['estimate'],
            'game_cluster_ci95': [1 - v['game_cluster_ci95'][1], 1 - v['game_cluster_ci95'][0]]}
    passed = ci[0] > -MARGIN
    bootstrap_passed = metrics['delta_' + k]['game_cluster_ci95'][0] > -MARGIN
    return {'games': len(games), 'points': int(arrays['points'].sum()),
            'formal_final_analysis': formal,
            'primary': {'difference_B' + k[1:] + '_minus_A': estimate,
                        'se_of_game_means': se,
                        'df': len(games) - 1, 't_critical': critical, 'ci95': ci,
                        'NI_margin': MARGIN,
                        'noninferiority_passed': bool(passed) if formal else None,
                        'bootstrap_agrees_with_NI': bool(bootstrap_passed) if formal else None,
                        'robust_NI': bool(passed and bootstrap_passed) if formal else None},
            'metrics': metrics, 'totals': {n: float(v.sum()) for n, v in arrays.items()},
            'judged': int(arrays['judged'].sum()), 'identical': int(arrays['identical'].sum()),
            'only_A_correct': int(arrays['only_A_correct_' + k].sum()),
            'only_B' + k[1:] + '_correct': int(arrays['only_B' + k[1:] + '_correct'].sum()),
            'disagreed_both_wrong': int(arrays['disagreed_both_wrong_' + k].sum()),
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
              'expected': {'families': list(FAMILIES), 'games_per_family': 64,
                           'points_per_family': 512, 'total_games': 128, 'total_points': 1024,
                           'seed_range': [SEED_BASE, SEED_BASE + 63]},
              'observed': {'games': len(metas), 'points': len(rows)}}
    family_games = {}
    for name in FAMILIES:
        if name not in by_family:
            continue
        grouped = defaultdict(lambda: defaultdict(float))
        for row in by_family[name]:
            for k, v in counts(row).items():
                grouped[row['seed']][k] += v
        seeds = sorted(grouped)
        family_games[name] = [grouped[s] for s in seeds]
    pooled = [g for games in family_games.values() for g in games]
    report['decision'] = {}
    for k in TOPKS:
        per = {}
        for name, games in family_games.items():
            part = summarize(games, k, formal=len(games) == 64)
            per[name] = part
        for name, part in per.items():
            seeds = sorted({r['seed'] for r in by_family[name]})
            part['seeds'] = [seeds[0], seeds[-1]]
            part['games_seeded'] = len(seeds)
        report['families'][k] = per
        report['decision']['B' + k[1:]] = {
            'passed_both_families': all(per[n]['primary']['robust_NI'] for n in per)
            if all(len(family_games[n]) == 64 for n in per) else None}
    for k in TOPKS:
        report['pooled_B' + k[1:] + '_reference_only'] = summarize(pooled, k,
                                                                   formal=len(pooled) == 128)
    summary_path = args.data / 'compare_summary.json'
    compared = json.loads(summary_path.read_text()) if summary_path.exists() else []
    report['strict_replay_frames'] = int(sum(r['strict_replay_frames'] for r in compared))
    report['terminal_trajectories'] = int(sum(m['terminal'] for m in metas))
    passing = [k for k, d in report['decision'].items() if d['passed_both_families']]
    report['conclusion'] = ('B' + min(passing)[1:] + ' passes in both families'
                            if passing else 'no k passes in both families')
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for k, per in report['families'].items():
        for name, v in per.items():
            print(json.dumps({'family': name, 'arm': 'B' + k[1:], 'games': v['games'],
                              'points': v['points'], 'primary': v['primary']}))
    print(json.dumps({'decision': report['decision'], 'conclusion': report['conclusion']}))


if __name__ == '__main__':
    main()
