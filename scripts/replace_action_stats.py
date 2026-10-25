"""Preregistered task69 paired win tests and game-clustered diagnostic intervals."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BOOTSTRAPS = 20000


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.quantile(values, [.025, .975]).tolist() if len(values) else None


def mean_with_interval(values, rng):
    values = np.asarray(values, dtype=float)
    samples = rng.integers(len(values), size=(BOOTSTRAPS, len(values)))
    return dict(mean=float(values.mean()), ci95=interval(values[samples].mean(axis=1)))


def exact_mcnemar(before_only, after_only):
    discordant = before_only + after_only
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(before_only, after_only) + 1))
    return min(1.0, 2.0 * tail / 2 ** discordant)


def holm(pvalues):
    ordered = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [None] * len(pvalues)
    running = 0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def ratio_samples(numerator, denominator, samples):
    top = np.asarray(numerator)[samples].sum(axis=1)
    bottom = np.asarray(denominator)[samples].sum(axis=1)
    return np.divide(top, bottom, out=np.full(len(top), np.nan), where=bottom > 0)


def verification_counts(games, kind):
    counts = [g['study']['verification_by_type'].get(kind, {}) for g in games]
    return ([c.get('revoked', 0) for c in counts], [c.get('attempts', 0) for c in counts])


def verification_summary(games, kind, rng):
    revoked, attempts = verification_counts(games, kind)
    samples = rng.integers(len(games), size=(BOOTSTRAPS, len(games)))
    return dict(revoked=sum(revoked), attempts=sum(attempts),
                ratio=sum(revoked) / sum(attempts) if sum(attempts) else None,
                ci95=interval(ratio_samples(revoked, attempts, samples)))


def ratio_difference(first, first_kind, second, second_kind, rng):
    if len(first) != len(second):
        raise ValueError('ratio differences require paired games/clusters')
    samples = rng.integers(len(first), size=(BOOTSTRAPS, len(first)))
    a, b = verification_counts(first, first_kind), verification_counts(second, second_kind)
    return dict(difference=(sum(a[0]) / sum(a[1]) - sum(b[0]) / sum(b[1]))
                if sum(a[1]) and sum(b[1]) else None,
                ci95=interval(ratio_samples(*a, samples) - ratio_samples(*b, samples)))


def arm_summary(games, rng):
    metrics = [g['study'] for g in games]
    replacements = [s.get('replace_attempts', 0) - s.get('replace_rejected', 0) for s in metrics]
    transitions, removed, added = Counter(), Counter(), Counter()
    for stats in metrics:
        for event in stats['replace_events']:
            removed.update(map(str, event['removed']))
            added.update(map(str, event['added']))
            transitions['+'.join(map(str, sorted(event['removed']))) + ' -> ' +
                        '+'.join(map(str, sorted(event['added'])))] += 1
    result = dict(wins=sum(g['won'] for g in games), n=len(games),
                nonterminal=sum(not g['terminal'] for g in games),
                replace_executed=sum(replacements), replace_rejected=sum(s['replace_rejected'] for s in metrics),
                replacements_per_game=mean_with_interval(replacements, rng),
                replacement_transitions=dict(transitions.most_common()),
                removed_types=dict(removed.most_common()), added_types=dict(added.most_common()),
                sun={key: mean_with_interval([s.get(key, 0) for s in metrics], rng)
                     for key in ('sun_produced', 'sun_spent', 'sun_remaining')},
                verification={kind: verification_summary(games, kind, rng)
                              for kind in ('plant', 'replace', 'wait')})
    # Descriptive resource cost, outside the preregistered win-test family. Use a separate
    # stream so adding diagnostics cannot change any primary/verification bootstrap draws.
    performance_rng = np.random.default_rng((sum(g['seed'] for g in games) + 69010000) % 2 ** 32)
    result['observed_performance'] = {key: mean_with_interval([g.get(key, 0) for g in games], performance_rng)
        for key in ('seconds', 'decisions', 'searched', 'rollout_lanes', 'halving_candidates', 'mowers_used', 'final_tick')}
    result['observed_performance_note'] = ('seconds is each worker game duration, not manager batch wall time; '
        'scheduler idle time is excluded; timing is descriptive, not a controlled throughput benchmark')
    return result


def paired_panel(before, after, seed):
    before = sorted(before, key=lambda g: g['seed'])
    after = sorted(after, key=lambda g: g['seed'])
    if len(before) != len(after) or [g['seed'] for g in before] != [g['seed'] for g in after]:
        raise ValueError('paired seed sets differ')
    if len({g['seed'] for g in before}) != len(before):
        raise ValueError('duplicate seed in paired panel')
    rng = np.random.default_rng(seed)
    both_win = sum(a['won'] and b['won'] for a, b in zip(before, after))
    both_lose = sum(not a['won'] and not b['won'] for a, b in zip(before, after))
    before_only = sum(a['won'] and not b['won'] for a, b in zip(before, after))
    after_only = sum(not a['won'] and b['won'] for a, b in zip(before, after))
    difference = mean_with_interval([int(b['won']) - int(a['won']) for a, b in zip(before, after)], rng)
    return dict(n=len(before), both_win=both_win, both_lose=both_lose,
                before_only_win=before_only, after_only_win=after_only,
                win_rate_difference=difference, mcnemar_two_sided_p=exact_mcnemar(before_only, after_only),
                before=arm_summary(before, rng), after=arm_summary(after, rng),
                sun_paired_differences={key: mean_with_interval(
                    [b['study'].get(key, 0) - a['study'].get(key, 0) for a, b in zip(before, after)], rng)
                    for key in ('sun_produced', 'sun_spent', 'sun_remaining')},
                verification_ratio_differences=dict(
                    after_replace_minus_after_plant=ratio_difference(after, 'replace', after, 'plant', rng),
                    after_plant_minus_before_plant=ratio_difference(after, 'plant', before, 'plant', rng)))


def load_panel(raw, task, arm):
    path = raw / task['task_id'] / arm / 'games.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if sorted(g['seed'] for g in rows) != sorted(task['seeds']):
        raise ValueError(f'incomplete or extra seeds: {path}')
    if any(g['arm'] != 'fast_search' or 'study' not in g for g in rows):
        raise ValueError(f'wrong arm or missing measurements: {path}')
    return rows, dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def analyze(manifest, raw):
    study = json.loads(manifest.read_text())
    results, provenance, fog_before, fog_after = {}, [], [], []
    for index, task in enumerate(study['tasks']):
        before, before_source = load_panel(raw, task, 'before')
        after, after_source = load_panel(raw, task, 'after')
        provenance.extend([before_source, after_source])
        results[task['task_id']] = paired_panel(before, after, 69000000 + index)
        if task['task_id'].startswith('fog_'):
            fog_before.extend(before)
            fog_after.extend(after)
    for panel, adjusted in zip(results.values(), holm([p['mcnemar_two_sided_p'] for p in results.values()])):
        panel['holm_adjusted_p'] = adjusted
    fog_names = [name for name in results if name.startswith('fog_')]
    positive_fog = any(results[name]['win_rate_difference']['mean'] > 0 and
                       results[name]['holm_adjusted_p'] < .05 for name in fog_names)
    nonnegative = all(panel['win_rate_difference']['mean'] >= 0 for panel in results.values())
    clean = all(panel[arm]['replace_rejected'] == 0 and panel[arm]['nonterminal'] == 0
                for panel in results.values() for arm in ('before', 'after'))
    return dict(preregistration=str(manifest), manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                before=study['before'], after=study['after'], models=study['models'],
                bootstraps=BOOTSTRAPS, cluster='paired seed/game; continuations are not independent observations',
                primary=results, supplementary_fog_pooled=paired_panel(fog_before, fog_after, 69000003),
                gate=dict(corrected_positive_fog=positive_fog, nonnegative_all_tasks=nonnegative,
                          no_rejected_replace_or_truncation=clean,
                          statistical_screen_passed=positive_fog and nonnegative and clean,
                          note='Deployment suggestion additionally requires legality/noise review; day N=32 does not prove noninferiority'),
                raw_sources=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(analyze(args.manifest, args.raw), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
