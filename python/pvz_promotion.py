"""task49A/task73b/task73c paired student promotion policy, no scipy dependency.

Newcombe method 10 (paired Wilson intervals, corrected phi), not a zero-width
Wald interval on unanimous outcomes. Formula/reference:
https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.confint_proportions_paired.html

Gate versions:
- 1 (task49A): fixed 64 pairs on the registered seed window [30000, 30128),
  same seeds every round. Kept byte-compatible so existing certificates verify.
- 2 (task73b): fresh seeds every round from the gate-dedicated pool
  [31024, 65536) via an out-of-git ledger, 1536 pairs, incumbent re-evaluated
  on the same fresh seeds (no cached incumbent outcomes).
- 3 (task73c): panel extended with the zones the curriculum actually practices -
  fog (fog_user L31, human observation) and roof (L41 r7) - and the floor
  day_l9 holdout replaced by a measured in-band holdout. 1920 pairs. Decision
  rules unchanged; v1/v2 certificates stay verifiable.
"""
from __future__ import annotations

import copy
import math
from statistics import NormalDist

GATE_VERSION = 3
ALPHA = .05
MARGIN = .03
SEEDS_PER_TASK = 8
PANEL_TASKS = (
    ('day', 'day_l7_eval_d7'), ('day', 'day_l10_orig_eval'),
    ('heldout', 'day_l7_eval_d7p_holdout1'), ('heldout', 'day_l9_eval_d7p_holdout1'),
    ('night', 'night_l12_eval_n10'), ('night', 'night_l18_eval_n10'),
    ('pool', 'pool_l21_eval_p9k'), ('pool', 'pool_l26_eval_p9k'),
)
# task73c: the zones the gate must measure are the zones the curriculum practices.
# fog_user_l31_snow / roof_l41_train_r7 are the practiced cells (multiplier 1.0,
# read from curriculum_state on 2026-10-25 and frozen here); the fog task keeps its
# registered human fog observation. The second heldout task replaces the floor
# day_l9 holdout (0-2/192 on both sides in the round-13 gate). Every registered
# day-family holdout was measured on the incumbent student first
# (scripts/task73c_holdout_probe.py, 64 fresh ledger seeds each): the group is
# strongly polarized - floors at 0-2% and ceilings at 87.5-100%, none inside
# 15-85%. day_l9_eval_d7p_holdout2 is the closest to band (56/64 = 87.5%, 95% CI
# [77%, 94%] straddling the upper edge) and keeps full regression veto power.
PANEL_TASKS_V3 = (
    ('day', 'day_l7_eval_d7'), ('day', 'day_l10_orig_eval'),
    ('heldout', 'day_l7_eval_d7p_holdout1'), ('heldout', 'day_l9_eval_d7p_holdout2'),
    ('night', 'night_l12_eval_n10'), ('night', 'night_l18_eval_n10'),
    ('pool', 'pool_l21_eval_p9k'), ('pool', 'pool_l26_eval_p9k'),
    ('fog', 'fog_user_l31_snow'), ('roof', 'roof_l41_train_r7'),
)
V1_PANEL = 'v2-task49A'
V2_PANEL = 'v2-task73b'
V3_PANEL = 'v3-task73c'
VERSION_BY_PANEL = {V1_PANEL: 1, V2_PANEL: 2, V3_PANEL: 3}
V1_POOL = (30000, 30128)
SEED_POOL_V2 = (31024, 65536)
V2_SEEDS_PER_TASK = 192
V3_SEEDS_PER_TASK = 192
SEED_LEDGER_DEFAULT = '/home/newbiexvwu/pvz-logs/gate_seed_ledger.json'


def policy(version=GATE_VERSION):
    if version == 1:
        return dict(version=version, alpha=ALPHA, noninferiority_margin=MARGIN,
                    overall_interval='Newcombe paired method 10, two-sided 95%',
                    zone_veto='one-sided exact McNemar p < 0.05 in any panel group',
                    pair_count=64, episode_count=128, seeds_per_task=SEEDS_PER_TASK,
                    tasks=[dict(zone=zone, task_id=name) for zone, name in PANEL_TASKS])
    if version == 2:
        return dict(version=version, alpha=ALPHA, noninferiority_margin=MARGIN,
                    overall_interval='Newcombe paired method 10, two-sided 95%',
                    zone_veto='one-sided exact McNemar p < 0.05 in any panel group',
                    pair_count=V2_SEEDS_PER_TASK * len(PANEL_TASKS),
                    episode_count=2 * V2_SEEDS_PER_TASK * len(PANEL_TASKS),
                    seeds_per_task=V2_SEEDS_PER_TASK,
                    seed_pool=list(SEED_POOL_V2),
                    seed_ledger=SEED_LEDGER_DEFAULT,
                    seed_reuse='never; ledger-allocated contiguous block per round, '
                               'exhaustion aborts the gate instead of wrapping',
                    incumbent_rerun='both sides are evaluated on the round seeds; '
                                    'cached incumbent outcomes are never used',
                    tasks=[dict(zone=zone, task_id=name) for zone, name in PANEL_TASKS])
    if version == 3:
        return dict(version=version, alpha=ALPHA, noninferiority_margin=MARGIN,
                    overall_interval='Newcombe paired method 10, two-sided 95%',
                    zone_veto='one-sided exact McNemar p < 0.05 in any panel group',
                    pair_count=V3_SEEDS_PER_TASK * len(PANEL_TASKS_V3),
                    episode_count=2 * V3_SEEDS_PER_TASK * len(PANEL_TASKS_V3),
                    seeds_per_task=V3_SEEDS_PER_TASK,
                    seed_pool=list(SEED_POOL_V2),
                    seed_ledger=SEED_LEDGER_DEFAULT,
                    seed_reuse='never; ledger-allocated contiguous block per round, '
                               'exhaustion aborts the gate instead of wrapping',
                    incumbent_rerun='both sides are evaluated on the round seeds; '
                                    'cached incumbent outcomes are never used',
                    tasks=[dict(zone=zone, task_id=name) for zone, name in PANEL_TASKS_V3])
    raise ValueError(f'unknown gate version: {version}')


def _freeze_ledger_panel(document, spec, seed_block, per_task, panel_name, version, purpose):
    """Shared v2/v3 fresh-seed panel: ledger block partitioned across the spec tasks."""
    if seed_block is None or [int(s) for s in seed_block] != list(seed_block) or len(seed_block) != 2:
        raise ValueError(f'gate v{version} requires the ledger-allocated seed block')
    start, end = int(seed_block[0]), int(seed_block[1])
    if start < SEED_POOL_V2[0] or end > SEED_POOL_V2[1] or end - start != per_task * len(spec):
        raise ValueError(f'seed block [{start}, {end}) is outside the gate pool '
                         f'{list(SEED_POOL_V2)} or has the wrong size')
    table = {task['task_id']: task for task in document['tasks']}
    tasks, groups = [], {}
    for index, (zone, name) in enumerate(spec):
        if document['groups'].get(name) != zone or name not in table:
            raise ValueError(f'missing/wrong group for gate v{version}: {name}')
        task = copy.deepcopy(table[name])
        task['seeds'] = list(range(start + index * per_task, start + (index + 1) * per_task))
        if task.get('wave_cap') is not None or task['zombie_count_multiplier'] != 1:
            raise ValueError(f'promotion must use complete, original-difficulty evaluation games: {name}')
        tasks.append(task)
        groups[name] = zone
    seeds = [seed for task in tasks for seed in task['seeds']]
    if len(set(seeds)) != len(seeds) or any(s < start or s >= end for s in seeds):
        raise ValueError(f'gate v{version} seed assignment is not a disjoint partition of the block')
    return dict(schema_version=1, panel_version=panel_name, groups=groups, tasks=tasks,
                seed_block=[start, end], purpose=purpose)


def freeze_panel(document, version=GATE_VERSION, seed_block=None):
    """Subset registered v2 tasks, retaining decks/terrain and disjoint evaluation seeds."""
    if document.get('panel_version') != 'v2':
        raise ValueError('promotion requires the registered unified v2 panel')
    if version == 1 and seed_block is not None:
        raise ValueError('gate v1 uses fixed panel slices; seed_block is a v2 argument')
    table = {task['task_id']: task for task in document['tasks']}
    if version == 1:
        tasks, groups, seen = [], {}, set()
        for index, (zone, name) in enumerate(PANEL_TASKS):
            if document['groups'].get(name) != zone or name not in table:
                raise ValueError(f'missing/wrong v2 group: {name}')
            task = copy.deepcopy(table[name])
            start = index * SEEDS_PER_TASK
            seeds = task['seeds'][start:start + SEEDS_PER_TASK]
            if len(seeds) != SEEDS_PER_TASK or any(s < V1_POOL[0] or s >= V1_POOL[1] or s in seen for s in seeds):
                raise ValueError(f'insufficient/disallowed evaluation seeds: {name}')
            seen.update(seeds)
            task['seeds'] = seeds
            tasks.append(task)
            groups[name] = zone
        return dict(schema_version=1, panel_version=V1_PANEL, groups=groups, tasks=tasks,
                    purpose='task49A fixed 64-pair student promotion subset')
    if version == 2:
        return _freeze_ledger_panel(
            document, PANEL_TASKS, seed_block, V2_SEEDS_PER_TASK, V2_PANEL, 2,
            'task73b fresh-seed 1536-pair student promotion subset; '
            'block allocated from the out-of-git gate seed ledger')
    if version == 3:
        return _freeze_ledger_panel(
            document, PANEL_TASKS_V3, seed_block, V3_SEEDS_PER_TASK, V3_PANEL, 3,
            'task73c fresh-seed 1920-pair student promotion subset covering the '
            'practiced fog and roof zones; block allocated from the out-of-git gate seed ledger')
    raise ValueError(f'unknown gate version: {version}')


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
    try:
        version = VERSION_BY_PANEL[manifest['panel_version']]
    except (KeyError, TypeError):
        raise ValueError(f'unknown promotion manifest: {manifest.get("panel_version")!r}') from None
    expected = {(task['task_id'], seed) for task in manifest['tasks'] for seed in task['seeds']}
    actual = {(row['task_id'], row['seed']) for row in pairs}
    if len(pairs) != len(expected) or len(actual) != len(expected) or actual != expected:
        raise ValueError('promotion requires every registered pair exactly once')
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
    return dict(version=version, approved=not reasons, reasons=reasons,
                policy=policy(version), overall=overall, zones=zones)
