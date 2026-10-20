"""Precision supplement for the unchanged, preregistered task15 Student/Holm tests."""
import argparse
import hashlib
import json
from pathlib import Path

import mpmath as mp


def student_two_sided(t, df):
    with mp.workdps(100):
        t = mp.mpf(t)
        df = mp.mpf(df)
        return +mp.betainc(df / 2, mp.mpf('.5'), 0,
                          df / (df + t * t), regularized=True)


def holm(pvalues):
    result = {}
    with mp.workdps(100):
        maximum = mp.mpf(0)
        for rank, name in enumerate(sorted(pvalues, key=pvalues.get)):
            maximum = max(maximum, min(mp.mpf(1), (len(pvalues) - rank) * pvalues[name]))
            result[name] = +maximum
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(plan_path, stats_path, execution_path):
    plan_path = plan_path.resolve()
    plan = json.loads(plan_path.read_text())
    stats = json.loads(stats_path.read_text())
    execution = json.loads(execution_path.read_text())
    assert execution['complete'] and execution['evaluation']['results_sha256'] == sha(stats_path)
    assert execution['plan_sha256'] == sha(plan_path)
    assert stats['models'] == 14 and stats['total_games'] == 86016
    family = ['B-F', 'M-F', 'T-F', 'I-R', 'I-F', 'R-F', 'S-F']
    assert stats['primary_family'] == family
    wins = {}
    for job in plan['jobs']:
        for tier, task in enumerate(plan['eval_contract']['task_ids']):
            path = plan_path.parent / 'eval' / job['name'] / task / 'games.jsonl'
            assert sha(path) == stats['game_files_sha256'][str(path)]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 1024
            mapping = {r['seed']: int(r['won']) for r in rows}
            assert set(mapping) == set(range(30000, 31024))
            wins[job['arm'], job['seed'], tier] = mapping
    pvalues = {}
    for name in family:
        a, b = name.split('-')
        with mp.workdps(100):
            clusters = [mp.mpf(sum(wins[a, seed, tier][env] - wins[b, seed, tier][env]
                                  for seed in (1, 2) for tier in range(6))) / 12
                        for env in range(30000, 31024)]
            mean = mp.fsum(clusters) / 1024
            variance = mp.fsum((v - mean)**2 for v in clusters) / 1023
            assert abs(float(mean) - stats['contrasts'][name]['difference']) < 1e-15
            pvalues[name] = (student_two_sided(abs(mean) / mp.sqrt(variance / 1024), 1023)
                             if variance else mp.mpf(0 if mean else 1))
    adjusted = holm(pvalues)
    comparisons = {}
    for name in family:
        original = stats['contrasts'][name]
        lo, hi = original['env_seed_cluster_ci95']
        positive = original['both_training_seeds_positive'] and lo > 0 and adjusted[name] < mp.mpf('.05')
        negative = original['both_training_seeds_negative'] and hi < 0 and adjusted[name] < mp.mpf('.05')
        assert positive == original['supports_positive_effect']
        assert negative == original['supports_negative_effect']
        comparisons[name] = dict(t_p=mp.nstr(pvalues[name], 20), holm_p=mp.nstr(adjusted[name], 20),
                                 frozen_t_p=original['t_p'], frozen_holm_p=original['holm_p'],
                                 supports_positive_effect=bool(positive), supports_negative_effect=bool(negative))
    return dict(method='same paired Student t, df1023, same Holm7; 100-decimal arithmetic from exact integer cluster counts',
                purpose='numerical precision supplement only; frozen primary outputs and all seven decisions unchanged',
                statistics_sha256=sha(stats_path), execution_sha256=sha(execution_path),
                plan_sha256=sha(plan_path), source_sha256=sha(Path(__file__)),
                games_recounted=86016, all_decisions_unchanged=True, contrasts=comparisons)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--execution', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(check(args.plan, args.stats, args.execution), indent=2) + '\n')
