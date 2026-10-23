"""task49A preregistered paired promotion policy, with no scipy dependency.

Newcombe method 10 (paired Wilson intervals, corrected phi), not a zero-width
Wald interval on unanimous outcomes. Formula/reference:
https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.confint_proportions_paired.html
"""
from __future__ import annotations

import copy
import math
from statistics import NormalDist

GATE_VERSION = 1
ALPHA = .05
MARGIN = .03
SEEDS_PER_TASK = 8
PANEL_TASKS = (
    ('day', 'day_l7_eval_d7'), ('day', 'day_l10_orig_eval'),
    ('heldout', 'day_l7_eval_d7p_holdout1'), ('heldout', 'day_l9_eval_d7p_holdout1'),
    ('night', 'night_l12_eval_n10'), ('night', 'night_l18_eval_n10'),
    ('pool', 'pool_l21_eval_p9k'), ('pool', 'pool_l26_eval_p9k'),
)


def policy():
    return dict(version=GATE_VERSION, alpha=ALPHA, noninferiority_margin=MARGIN,
                overall_interval='Newcombe paired method 10, two-sided 95%',
                zone_veto='one-sided exact McNemar p < 0.05 in any panel group',
                pair_count=64, episode_count=128, seeds_per_task=SEEDS_PER_TASK,
                tasks=[dict(zone=zone, task_id=name) for zone, name in PANEL_TASKS])


def freeze_panel(document):
    """Subset registered v2 tasks, retaining decks/terrain and disjoint evaluation seeds."""
    if document.get('panel_version') != 'v2':
        raise ValueError('promotion requires the registered unified v2 panel')
    table = {task['task_id']: task for task in document['tasks']}
    tasks, groups, seen = [], {}, set()
    for index, (zone, name) in enumerate(PANEL_TASKS):
        if document['groups'].get(name) != zone or name not in table:
            raise ValueError(f'missing/wrong v2 group: {name}')
        task = copy.deepcopy(table[name])
        start = index * SEEDS_PER_TASK
        seeds = task['seeds'][start:start + SEEDS_PER_TASK]
        if len(seeds) != SEEDS_PER_TASK or any(s < 30000 or s >= 30128 or s in seen for s in seeds):
            raise ValueError(f'insufficient/disallowed evaluation seeds: {name}')
        seen.update(seeds)
        if task.get('wave_cap') is not None or task['zombie_count_multiplier'] != 1:
            raise ValueError(f'promotion must use complete, original-difficulty evaluation games: {name}')
        task['seeds'] = seeds
        tasks.append(task)
        groups[name] = zone
    return dict(schema_version=1, panel_version='v2-task49A', groups=groups, tasks=tasks,
                purpose='task49A fixed 64-pair student promotion subset')


def _wilson(wins, total):
    z = NormalDist().inv_cdf(1 - ALPHA / 2)
    p, z2 = wins / total, z * z
    center = (p + z2 / (2 * total)) / (1 + z2 / total)
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / (1 + z2 / total)
    return max(0., center - radius), min(1., center + radius)


def paired_interval(both_win, gain, harm, both_lose):
    """95% CI for candidate minus incumbent, from a matched binary contingency table."""
    counts = (both_win, gain, harm, both_lose)
    if any(type(x) is not int or x < 0 for x in counts) or not sum(counts):
        raise ValueError('paired table requires nonnegative integer counts and at least one pair')
    n = sum(counts)
    candidate, incumbent = (both_win + gain) / n, (both_win + harm) / n
    lc, uc = _wilson(both_win + gain, n)
    li, ui = _wilson(both_win + harm, n)
    denominator = math.sqrt(candidate * (1 - candidate) * incumbent * (1 - incumbent))
    covariance = (both_win * both_lose - gain * harm) / (n * n)
    if covariance > 0:
        covariance = max(0., covariance - 1 / (2 * n))
    correlation = covariance / denominator if denominator else 0.
    a, b = candidate - lc, ui - incumbent
    lower = candidate - incumbent - math.sqrt(max(0., a*a + b*b - 2*correlation*a*b))
    a, b = uc - candidate, incumbent - li
    upper = candidate - incumbent + math.sqrt(max(0., a*a + b*b - 2*correlation*a*b))
    return max(-1., lower), min(1., upper)


def summarize(pairs):
    if not pairs or any(type(r[side]) is not bool for r in pairs for side in ('candidate', 'incumbent')):
        raise ValueError('complete boolean paired outcomes required')
    both_win = sum(r['candidate'] and r['incumbent'] for r in pairs)
    gain = sum(r['candidate'] and not r['incumbent'] for r in pairs)
    harm = sum(not r['candidate'] and r['incumbent'] for r in pairs)
    both_lose = len(pairs) - both_win - gain - harm
    low, high = paired_interval(both_win, gain, harm, both_lose)
    discordant = gain + harm
    p_worse = sum(math.comb(discordant, k) for k in range(harm, discordant + 1)) / 2**discordant
    return dict(pairs=len(pairs), candidate_wins=both_win + gain, incumbent_wins=both_win + harm,
                gain=gain, harm=harm, both_win=both_win, both_lose=both_lose,
                delta_rate=(gain - harm) / len(pairs), ci95=[low, high],
                mcnemar_worse_p=p_worse)


def decide(pairs, manifest):
    expected = {(task['task_id'], seed) for task in manifest['tasks'] for seed in task['seeds']}
    actual = {(row['task_id'], row['seed']) for row in pairs}
    if len(pairs) != 64 or len(actual) != 64 or actual != expected:
        raise ValueError('promotion requires all 64 registered pairs exactly once')
    for row in pairs:
        if row['zone'] != manifest['groups'][row['task_id']]:
            raise ValueError('paired outcome has the wrong zone')
    overall = summarize(pairs)
    zones = {zone: summarize([row for row in pairs if row['zone'] == zone])
             for zone in sorted(set(manifest['groups'].values()))}
    reasons = []
    if overall['ci95'][0] <= -MARGIN:
        reasons.append('overall_ci_lower_not_above_minus_3pp')
    for zone, result in zones.items():
        if result['mcnemar_worse_p'] < ALPHA:
            reasons.append('significant_zone_regression:' + zone)
    return dict(version=GATE_VERSION, approved=not reasons, reasons=reasons,
                policy=policy(), overall=overall, zones=zones)
