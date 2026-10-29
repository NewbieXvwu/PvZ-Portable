"""Paired student promotion policy, no scipy dependency.

Gate version 6 (task168): the panel is the adventure set the game actually plays
(original 1.0x, scoreboard decks, 38 levels x 45-46 pairs) instead of synthetic
curriculum cells; the two registered heldout cells keep running at 96 pairs each
for monitoring only. The overall -3pp rule uses the adventure pairs; four non-roof
zones use binary Holm vetoes and 90 pooled roof pairs use a one-sided exact
Wilcoxon final_wave veto at .01. Heldout never vetoes or enters the overall rate.

Newcombe method 10 (paired Wilson intervals, corrected phi), not a zero-width
Wald interval on unanimous outcomes. Formula/reference:
https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.confint_proportions_paired.html
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from itertools import groupby
from statistics import NormalDist, median

GATE_VERSION = 6
ROOF_WAVE_ALPHA = .01
ALPHA = .05
MARGIN = .03
# task128 (gate v5): the panel is the adventure set, not synthetic curriculum cells.
# Levels come from configs/adventure_scoreboard.json (task103: 45 levels, 1.0x,
# scoreboard decks) and enter only when the r11 student or the teacher has a win
# there (scripts/make_gate_v5_panel.py). The original scoreboard qualifies roof
# L41 (teacher 3/16) and L46 (2/16), despite the later unattributed task115 summary.
# The two registered heldout cells run for monitoring only.
PANEL_TASKS_V5 = (
    ('day', 'task101_l1_day_d7', 46),
    ('day', 'task101_l2_day_d7', 46),
    ('day', 'task101_l3_day_d7', 46),
    ('day', 'task101_l4_day_d7', 46),
    ('day', 'task101_l6_day_d7', 46),
    ('day', 'task101_l7_day_d7', 46),
    ('day', 'task101_l8_day_d7', 46),
    ('day', 'task101_l9_day_d7', 46),
    ('day', 'task101_l10_day_d7', 46),
    ('night', 'task101_l11_night_n10', 46),
    ('night', 'task101_l12_night_n10', 46),
    ('night', 'task101_l13_night_n10', 46),
    ('night', 'task101_l14_night_n10', 46),
    ('night', 'task101_l16_night_n10', 46),
    ('night', 'task101_l17_night_n10', 46),
    ('night', 'task101_l18_night_n10', 46),
    ('night', 'task101_l19_night_n10', 46),
    ('night', 'task101_l20_night_n10', 46),
    ('pool', 'task101_l21_pool_p9k', 45),
    ('pool', 'task101_l22_pool_p9k', 45),
    ('pool', 'task101_l23_pool_p9k', 45),
    ('pool', 'task101_l24_pool_p9k', 45),
    ('pool', 'task101_l26_pool_p9k', 45),
    ('pool', 'task101_l27_pool_p9k', 45),
    ('pool', 'task101_l28_pool_p9k', 45),
    ('pool', 'task101_l29_pool_p9k', 45),
    ('pool', 'task101_l30_native', 45),
    ('fog', 'task101_l31_fog_selected', 45),
    ('fog', 'task101_l32_fog_selected', 45),
    ('fog', 'task101_l33_fog_selected', 45),
    ('fog', 'task101_l34_fog_selected', 45),
    ('fog', 'task101_l36_fog_selected', 45),
    ('fog', 'task101_l37_fog_selected', 45),
    ('fog', 'task101_l38_fog_selected', 45),
    ('fog', 'task101_l39_fog_selected', 45),
    ('fog', 'task101_l40_native', 45),
    ('roof', 'task101_l41_roof_demo', 45),
    ('roof', 'task101_l46_roof_demo', 45),
    ('heldout', 'day_l7_eval_d7p_holdout1', 96),
    ('heldout', 'day_l9_eval_d7p_holdout2', 96),
)
V5_PANEL = 'v5-task128'
V6_PANEL = 'v6-task168'
V5_MONITOR_TASKS = ('day_l7_eval_d7p_holdout1', 'day_l9_eval_d7p_holdout2')
V5_SEEDS_PER_TASK = 45
V5_MONITOR_SEEDS_PER_TASK = 96
# task88: ordered pool list. [31024, 65536) was registered for task73b and is
# nearly spent (ledger allocated up to 36272 on 2026-10-26); the second window is
# 1,000,000 seeds, i.e. ~520 rounds, and sits above every other registered
# segment (highest is 83,000,307, llmplay robustness). Allocation walks the list
# in order, so extending it never moves a block that was already issued.
SEED_POOLS_V2 = ((31024, 65536), (100_000_000, 101_000_000))
SEED_LEDGER_DEFAULT = str(Path.home() / 'pvz-logs/gate_seed_ledger.json')
# task99: future seed blocks of the demo start cells (configs/human_demo_tasks.json
# demo_start.future_seed_start). The block written in the manifest stays the first
# segment forever, so old rows stay comparable; when a cell's offset reaches the end
# of its table, qc renews it from this dedicated pool through its own out-of-git
# ledger. Checked with scripts/gate_seed_inventory.py (full scan, not --quick: only
# the full scan parses the JSON manifests). Above every declared segment (highest is
# [130,000,000, 140,000,000)); 9 digits, like the gate's second window.
# The first candidate [90,000,000, 91,000,000) was rejected: task89's llm-run-import
# e2e test had already declared and played future seeds from 90,000,000 upward
# (worktree llmimport, artifacts/e2e/scan_jobs.json + scan_out/*/games.jsonl).
START_SEED_POOL = (200_000_000, 201_000_000)
START_SEED_LEDGER_DEFAULT = str(Path.home() / 'pvz-logs/start_cell_seed_ledger.json')
START_SEED_BLOCK = 256


def policy(version=GATE_VERSION):
    if version != GATE_VERSION:
        raise ValueError(f'unknown gate version: {version}')
    return dict(version=version, alpha=ALPHA, noninferiority_margin=MARGIN,
                overall_interval='Newcombe paired method 10, two-sided 95%',
                zone_veto='one-sided exact McNemar p across the four non-roof adventure zones; '
                          'Holm–Bonferroni step-down correction, family-wise alpha = 0.05; '
                          'veto while ordered p < 0.05 / (group_count - rank + 1)',
                roof_veto=dict(endpoint='candidate_wave - incumbent_wave (final_wave)',
                               test='one-sided exact Wilcoxon signed-rank, alternative less',
                               alpha=ROOF_WAVE_ALPHA, unit='roof zone, pooling both tasks (90 pairs)',
                               zeros='discard', ties='average ranks',
                               distribution='exact conditional sign-flip, inclusive lower tail',
                               tasks=[name for zone, name, _ in PANEL_TASKS_V5 if zone == 'roof']),
                pair_count=sum(count for _, _, count in PANEL_TASKS_V5),
                episode_count=2 * sum(count for _, _, count in PANEL_TASKS_V5),
                seeds_per_task=V5_SEEDS_PER_TASK,
                seed_assignment="1728 adventure pairs evenly divided: first 18 tasks take 46, remaining 20 take 45",
                monitor_seeds_per_task=V5_MONITOR_SEEDS_PER_TASK,
                seed_pool=list(SEED_POOLS_V2[0]),
                seed_ledger=SEED_LEDGER_DEFAULT,
                seed_reuse='never; ledger-allocated contiguous block per round, '
                           'exhaustion aborts the gate instead of wrapping',
                incumbent_rerun='both sides are evaluated on the round seeds; '
                                'cached incumbent outcomes are never used',
                panel='adventure levels (original 1.0x, scoreboard decks) where the r11 student or '
                      'the teacher has a recorded win; all-loss cells carry no regression signal',
                heldout='monitor only: 96 pairs each, written into the round report, '
                        'never vetoes and never enters the overall rate',
                tasks=[dict(zone=zone, task_id=name, seeds=count, monitor=name in V5_MONITOR_TASKS)
                       for zone, name, count in PANEL_TASKS_V5])


def freeze_panel(document, version=GATE_VERSION, seed_block=None):
    """Partition one ledger block across the registered adventure panel.

    Adventure tasks take 45 or 46 pairs each and the two monitoring
    cells V5_MONITOR_SEEDS_PER_TASK; the block must be exactly the registered
    size, so a stale source panel cannot silently shrink or grow the gate.
    """
    if version != GATE_VERSION:
        raise ValueError(f'unknown gate version: {version}')
    if document.get('panel_version') != 'v5':
        raise ValueError('promotion requires the registered v5 panel')
    if seed_block is None or [int(s) for s in seed_block] != list(seed_block) or len(seed_block) != 2:
        raise ValueError('gate v5 requires the ledger-allocated seed block')
    start, end = int(seed_block[0]), int(seed_block[1])
    if not any(low <= start and end <= high for low, high in SEED_POOLS_V2):
        raise ValueError(f'seed block [{start}, {end}) is outside the gate pools '
                         f'{[list(pool) for pool in SEED_POOLS_V2]}')
    total = sum(count for _, _, count in PANEL_TASKS_V5)
    if end - start != total:
        raise ValueError(f'seed block [{start}, {end}) has the wrong size for gate v5 '
                         f'(needs {total} seeds)')
    table = {task['task_id']: task for task in document['tasks']}
    tasks, groups, cursor = [], {}, start
    for zone, name, count in PANEL_TASKS_V5:
        if document['groups'].get(name) != zone or name not in table:
            raise ValueError(f'missing/wrong group for gate v5: {name}')
        task = copy.deepcopy(table[name])
        task['seeds'] = list(range(cursor, cursor + count))
        cursor += count
        if task.get('wave_cap') is not None or task['zombie_count_multiplier'] != 1:
            raise ValueError(f'promotion must use complete, original-difficulty evaluation games: {name}')
        tasks.append(task)
        groups[name] = zone
    if cursor != end:
        raise ValueError('gate v5 seed assignment is not a disjoint partition of the block')
    return dict(schema_version=1, panel_version=V6_PANEL, groups=groups, tasks=tasks,
                seed_block=[start, end], monitor=list(V5_MONITOR_TASKS),
                purpose='task168 v5 panel and seeds; roof final_wave Wilcoxon veto at 0.01 '
                        'across 90 roof pairs, four non-roof binary zones Holm at 0.05')


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


def wilcoxon_worse(differences):
    """Exact lower tail of W+, dropping zeros and averaging tied absolute ranks."""
    ranked, position = [], 0
    for _, group in groupby(sorted((d for d in differences if d), key=abs), key=abs):
        values = list(group)
        rank = 2 * position + len(values) + 1
        ranked.extend((value, rank) for value in values)
        position += len(values)
    observed = sum(rank for value, rank in ranked if value > 0)
    counts = [1] + [0] * sum(rank for _, rank in ranked)
    reachable = 0
    for _, rank in ranked:
        for total in range(reachable, -1, -1):
            counts[total + rank] += counts[total]
        reachable += rank
    return dict(positive_rank_sum=observed / 2, nonzero_pairs=len(ranked),
                wilcoxon_worse_p=sum(counts[:observed + 1]) / 2**len(ranked))


def summarize_waves(pairs):
    differences = [row['candidate_wave'] - row['incumbent_wave'] for row in pairs]
    result = wilcoxon_worse(differences)
    result.update(pairs=len(pairs), candidate_mean_wave=sum(row['candidate_wave'] for row in pairs) / len(pairs),
                  incumbent_mean_wave=sum(row['incumbent_wave'] for row in pairs) / len(pairs),
                  mean_delta_wave=sum(differences) / len(pairs), median_delta_wave=median(differences))
    return result


def decide_scored(scored):
    """Holm-Bonferroni zone veto on scored pairs, independent of panel size."""
    overall = summarize(scored)
    zones = {zone: summarize([row for row in scored if row['zone'] == zone])
             for zone in sorted({row['zone'] for row in scored})}
    reasons = []
    if overall['ci95'][0] <= -MARGIN:
        reasons.append('overall_ci_lower_not_above_minus_3pp')
    # Include non-roof zones with no discordant pairs; roof has a numeric endpoint.
    # Cumulative maximum implements Holm's stop-at-first-nonrejection rule.
    adjusted = 0.
    ordered = sorted((zone for zone in zones if zone != 'roof'),
                     key=lambda zone: (zones[zone]['mcnemar_worse_p'], zone))
    for rank, zone in enumerate(ordered):
        adjusted = max(adjusted, min(1., (len(ordered) - rank) * zones[zone]['mcnemar_worse_p']))
        zones[zone]['holm_adjusted_p'] = adjusted
        if adjusted < ALPHA:
            reasons.append('significant_zone_regression:' + zone)
    if 'roof' in zones:
        roof = [row for row in scored if row['zone'] == 'roof']
        result = summarize_waves(roof)
        result.update(alpha=ROOF_WAVE_ALPHA, veto=result['wilcoxon_worse_p'] < ROOF_WAVE_ALPHA)
        zones['roof']['waves'] = result
        if result['veto']:
            reasons.append('significant_roof_wave_regression:roof')
        zones['roof']['wave_tasks'] = {
            name: summarize_waves([row for row in roof if row['task_id'] == name])
            for name in sorted({row['task_id'] for row in roof})}
    return dict(version=GATE_VERSION, approved=not reasons, reasons=reasons,
                policy=policy(), overall=overall, zones=zones)


def decide(pairs, manifest):
    if manifest.get('panel_version') != V6_PANEL:
        raise ValueError(f'unknown promotion manifest: {manifest.get("panel_version")!r}')
    expected = {(task['task_id'], seed) for task in manifest['tasks'] for seed in task['seeds']}
    actual = {(row['task_id'], row['seed']) for row in pairs}
    if len(pairs) != len(expected) or len(actual) != len(expected) or actual != expected:
        raise ValueError('promotion requires every registered pair exactly once')
    for row in pairs:
        if row['zone'] != manifest['groups'][row['task_id']]:
            raise ValueError('paired outcome has the wrong zone')
    monitor = tuple(manifest.get('monitor') or ())
    if not monitor:
        raise ValueError('gate v5 manifest has no monitoring cells')
    result = decide_scored([row for row in pairs if row['task_id'] not in set(monitor)])
    result['monitor'] = {name: summarize([row for row in pairs if row['task_id'] == name])
                         for name in monitor}
    result['monitor_note'] = ('heldout cells are reported, never vetoed and never counted '
                              'in the overall rate')
    return result
