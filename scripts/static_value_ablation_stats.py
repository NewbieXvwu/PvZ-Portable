"""Paired whole-game uncertainty for preregistered candidate-generation arms."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from static_value_ablation import ARMS
from static_value_compare import key


def point_counts(row):
    out = {'points': 1, 'expanded_label_changes': int(key(row['heavy_label']) != key(row['expanded']['label'])),
           'new_actions': len(row['expanded']['new_actions']),
           'new_truth_continuations': row['expanded']['new_continuations'],
           'expanded_verification_continuations': row['expanded']['verification_continuations']}
    for name in ARMS:
        arm, noise = row['arms'][name], row['arms'][name + '_noise']
        assert arm['candidates'] == noise['candidates']
        out[name + '_noise'] = int(key(arm['chosen']) == key(noise['chosen']))
        for suffix, v in [('', arm), ('_reseed', noise)]:
            for ref, label in [('frozen', row['heavy_label']), ('expanded', row['expanded']['label'])]:
                out[f'{name}{suffix}_{ref}'] = int(key(v['chosen']) == key(label))
            out[name + suffix + '_cost'] = v['continuations']
            out[name + suffix + '_halving'] = v['halving_continuations']
            out[name + suffix + '_verify'] = v['verification_continuations']
        out[name + '_candidates'] = len(arm['candidates'])
        cards = {a['packet'] for a in row['actions'] if a['type'] == 'plant'}
        included = {a['packet'] for a in arm['candidates'] if a['type'] == 'plant'}
        out[name + '_missing_cards'] = len(cards - included)
        out[name + '_heavy_in_pool'] = int(key(row['heavy_label']) in {key(a) for a in arm['candidates']})
    out['student_frozen'] = int(key(row['policy']) == key(row['heavy_label']))
    out['student_expanded'] = int(key(row['policy']) == key(row['expanded']['label']))
    return out


def clustered(games):
    assert games
    names = sorted(set().union(*(set(g) for g in games)))
    values = {n: np.asarray([g.get(n, 0) for g in games], dtype=float) for n in names}
    picks = np.random.default_rng(20261020).integers(0, len(games), size=(10000, len(games)))
    def ratio(num, den):
        assert den.sum() > 0
        boot = num[picks].sum(axis=1) / den[picks].sum(axis=1)
        return {'estimate': float(num.sum() / den.sum()),
                'game_cluster_ci95': np.quantile(boot, [.025, .975]).tolist(),
                'game_cluster_ci98_75': np.quantile(boot, [.00625, .99375]).tolist()}
    metrics = {n: ratio(v, values['points']) for n, v in values.items() if n != 'points'}
    comparisons = {}
    for name in ARMS:
        if name == 'A':
            continue
        result = {}
        for suffix in ('', '_reseed'):
            for ref in ('frozen', 'expanded'):
                result[suffix + '_' + ref + '_agreement_minus_A'] = ratio(
                    values[name + suffix + '_' + ref] - values['A' + suffix + '_' + ref], values['points'])
            result[suffix + '_cost_ratio_to_A'] = ratio(values[name + suffix + '_cost'], values['A' + suffix + '_cost'])
            result[suffix + '_cost_minus_A'] = ratio(values[name + suffix + '_cost'] - values['A' + suffix + '_cost'], values['points'])
        result['noise_agreement_minus_A'] = ratio(values[name + '_noise'] - values['A_noise'], values['points'])
        # Decision uses the original D, with reseed arms reported as a stability check.
        cost_ci = result['_cost_ratio_to_A']['game_cluster_ci98_75']
        result['decision_gate_by_reference'] = {}
        for ref in ('frozen', 'expanded'):
            ci = result['_' + ref + '_agreement_minus_A']['game_cluster_ci98_75']
            result['decision_gate_by_reference'][ref] = {
                'noninferior_3pp': bool(ci[0] > -.03), 'superior': bool(ci[0] > 0),
                'saves_continuations': bool(cost_ci[1] < 1),
                'passes': bool(ci[0] > 0 or (ci[0] > -.03 and cost_ci[1] < 1))}
        result['recommendation_gate'] = all(g['passes'] for g in result['decision_gate_by_reference'].values())
        comparisons[name] = result
    return {'games': len(games), 'points': int(values['points'].sum()),
            'totals': {n: float(v.sum()) for n, v in values.items()}, 'metrics': metrics,
            'paired_vs_A': comparisons, 'bootstrap_resamples': 10000, 'bootstrap_seed': 20261020,
            'primary_family': 'four alternatives vs A; 98.75% two-sided percentile intervals (Bonferroni4); ordinary95% descriptive',
            'noninferiority_margin': .03}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads((args.source / 'generate_summary.json').read_text())
    expected = {(m['seed'], p['decision']) for m in meta for p in m['points']}
    rows = [json.loads(p.read_text()) for p in sorted((args.data / 'points').glob('*.json'))]
    assert len(rows) == len(expected) == 256 and {(r['seed'], r['decision']) for r in rows} == expected
    games = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for k, v in point_counts(row).items():
            games[row['seed']][k] += v
    assert len(games) == 32
    result = clustered([games[s] for s in sorted(games)])
    result['per_game'] = {str(s): dict(games[s]) for s in sorted(games)}
    result['actual_light_continuations'] = sum(a['continuations'] for r in rows for a in r['arms'].values())
    result['actual_total_continuations'] = (result['actual_light_continuations']
        + result['totals']['new_truth_continuations'] + result['totals']['expanded_verification_continuations'])
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'points': result['points'], 'games': result['games'],
                      'actual_total_continuations': result['actual_total_continuations']}))


if __name__ == '__main__':
    main()
