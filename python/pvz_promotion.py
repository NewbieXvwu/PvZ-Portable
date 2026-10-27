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
- 4 (task113): same panel, seeds and overall noninferiority as v3; zone vetoes
  use Holm–Bonferroni step-down correction at family-wise alpha 0.05.
- 5 (task128): the panel is the adventure set the game actually plays (original
  1.0x, scoreboard decks, 38 levels x 45-46 pairs) instead of synthetic curriculum
  cells; the two registered heldout cells keep running at 96 pairs each for
  monitoring only. The overall -3pp rule and the Holm-Bonferroni zone veto are
  computed on the adventure pairs alone; heldout never vetoes and never enters
  the overall rate.
- 3 (task73c): panel extended with the zones the curriculum actually practices -
  fog (fog_user L31, human observation) and roof (L41 r7) - and the floor
  day_l9 holdout replaced by a measured in-band holdout. 1920 pairs. Decision
  rules unchanged; v1/v2 certificates stay verifiable.

task88 extends the v2/v3 seed pool from one window to an ordered list
(SEED_POOLS_V2). Blocks are still handed out in order and never reused; the list
only says what to do when the first window runs out. The panel, the decision
rules, the policy document and GATE_VERSION are untouched, so v2/v3 certificates
issued before task88 stay byte-identical and verifiable.
"""
from __future__ import annotations

import copy
import math
from statistics import NormalDist

GATE_VERSION = 5
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
V5_MONITOR_TASKS = ('day_l7_eval_d7p_holdout1', 'day_l9_eval_d7p_holdout2')
V5_SEEDS_PER_TASK = 45
V5_MONITOR_SEEDS_PER_TASK = 96
V1_PANEL = 'v2-task49A'
V2_PANEL = 'v2-task73b'
V3_PANEL = 'v3-task73c'
V4_PANEL = 'v4-task113'
VERSION_BY_PANEL = {V1_PANEL: 1, V2_PANEL: 2, V3_PANEL: 3, V4_PANEL: 4, V5_PANEL: 5}
V1_POOL = (30000, 30128)
# task88: ordered pool list. [31024, 65536) was registered for task73b and is
# nearly spent (ledger allocated up to 36272 on 2026-10-26); the second window is
# 1,000,000 seeds, i.e. ~520 v3 rounds, and sits above every other registered
# segment (highest is 83,000,307, llmplay robustness). Allocation walks the list
# in order, so extending it never moves a block that was already issued.
SEED_POOLS_V2 = ((31024, 65536), (100_000_000, 101_000_000))
SEED_POOL_V2 = SEED_POOLS_V2[0]  # legacy name: the window v2/v3 were registered against
V2_SEEDS_PER_TASK = 192
V3_SEEDS_PER_TASK = 192
SEED_LEDGER_DEFAULT = '/home/newbiexvwu/pvz-logs/gate_seed_ledger.json'
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
START_SEED_LEDGER_DEFAULT = '/home/newbiexvwu/pvz-logs/start_cell_seed_ledger.json'
START_SEED_BLOCK = 256


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
                    # Registered window of this gate version, kept verbatim after
                    # task88 extended the pool list: the policy document is hashed
                    # into every contract, so editing it would invalidate in-flight
                    # certificates. The ledger records which window each block came
                    # from; see SEED_POOLS_V2.
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
                    # Registered window of this gate version, kept verbatim after
                    # task88 extended the pool list: the policy document is hashed
                    # into every contract, so editing it would invalidate in-flight
                    # certificates. The ledger records which window each block came
                    # from; see SEED_POOLS_V2.
                    seed_pool=list(SEED_POOL_V2),
                    seed_ledger=SEED_LEDGER_DEFAULT,
                    seed_reuse='never; ledger-allocated contiguous block per round, '
                               'exhaustion aborts the gate instead of wrapping',
                    incumbent_rerun='both sides are evaluated on the round seeds; '
                                    'cached incumbent outcomes are never used',
                    tasks=[dict(zone=zone, task_id=name) for zone, name in PANEL_TASKS_V3])
    if version == 4:
        result = policy(3)
        result.update(version=4, zone_veto='one-sided exact McNemar p across all panel groups; '
                      'Holm–Bonferroni step-down correction, family-wise alpha = 0.05; '
                      'veto while ordered p < 0.05 / (group_count - rank + 1)')
        return result
    if version == 5:
        return dict(version=version, alpha=ALPHA, noninferiority_margin=MARGIN,
                    overall_interval='Newcombe paired method 10, two-sided 95%',
                    zone_veto='one-sided exact McNemar p across the adventure zones; '
                              'Holm–Bonferroni step-down correction, family-wise alpha = 0.05; '
                              'veto while ordered p < 0.05 / (group_count - rank + 1)',
                    pair_count=sum(count for _, _, count in PANEL_TASKS_V5),
                    episode_count=2 * sum(count for _, _, count in PANEL_TASKS_V5),
                    seeds_per_task=V5_SEEDS_PER_TASK,
                    seed_assignment="1728 adventure pairs evenly divided: first 18 tasks take 46, remaining 20 take 45",
                    monitor_seeds_per_task=V5_MONITOR_SEEDS_PER_TASK,
                    seed_pool=list(SEED_POOL_V2),
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
    raise ValueError(f'unknown gate version: {version}')


def _freeze_ledger_panel(document, spec, seed_block, per_task, panel_name, version, purpose):
    """Shared v2/v3 fresh-seed panel: ledger block partitioned across the spec tasks."""
    if seed_block is None or [int(s) for s in seed_block] != list(seed_block) or len(seed_block) != 2:
        raise ValueError(f'gate v{version} requires the ledger-allocated seed block')
    start, end = int(seed_block[0]), int(seed_block[1])
    if not any(low <= start and end <= high for low, high in SEED_POOLS_V2):
        raise ValueError(f'seed block [{start}, {end}) is outside the gate pools '
                         f'{[list(pool) for pool in SEED_POOLS_V2]} or has the wrong size')
    if end - start != per_task * len(spec):
        raise ValueError(f'seed block [{start}, {end}) has the wrong size for gate v{version}')
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


def _freeze_adventure_panel(document, seed_block):
    """task128: partition one ledger block across the registered adventure panel.

    Adventure tasks take 45 or 46 pairs each and the two monitoring
    cells V5_MONITOR_SEEDS_PER_TASK; the block must be exactly the registered
    size, so a stale source panel cannot silently shrink or grow the gate.
    """
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
    return dict(schema_version=1, panel_version=V5_PANEL, groups=groups, tasks=tasks,
                seed_block=[start, end], monitor=list(V5_MONITOR_TASKS),
                purpose='task128 fresh-seed 1920-pair adventure promotion subset; adventure '
                        'zones veto at Holm-Bonferroni family-wise 0.05, heldout cells monitored only')


def freeze_panel(document, version=GATE_VERSION, seed_block=None):
    """Subset registered tasks, retaining decks/terrain and disjoint evaluation seeds."""
    if document.get('panel_version') != ('v5' if version == 5 else 'v2'):
        raise ValueError(f'promotion requires the registered {"v5" if version == 5 else "unified v2"} panel')
    if version == 5:
        return _freeze_adventure_panel(document, seed_block)
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
    if version == 4:
        return _freeze_ledger_panel(
            document, PANEL_TASKS_V3, seed_block, V3_SEEDS_PER_TASK, V4_PANEL, 4,
            'task113 fresh-seed 1920-pair student promotion subset; v3 panel and seed pools, '
            'Holm–Bonferroni zone veto at family-wise alpha 0.05')
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
    # task128: the heldout cells are monitoring only. They are still evaluated and
    # must be complete, but they neither veto nor move the overall rate.
    monitor = tuple(manifest.get('monitor') or ()) if version >= 5 else ()
    if version >= 5 and not monitor:
        raise ValueError('gate v5 manifest has no monitoring cells')
    scored = [row for row in pairs if row['task_id'] not in set(monitor)]
    overall = summarize(scored)
    zones = {zone: summarize([row for row in scored if row['zone'] == zone])
             for zone in sorted({row['zone'] for row in scored})}
    reasons = []
    if overall['ci95'][0] <= -MARGIN:
        reasons.append('overall_ci_lower_not_above_minus_3pp')
    if version >= 4:
        # The family includes every zone, including those with no discordant pairs.
        # Cumulative maximum implements Holm's stop-at-first-nonrejection rule.
        adjusted = 0.
        ordered = sorted(zones, key=lambda zone: (zones[zone]['mcnemar_worse_p'], zone))
        for rank, zone in enumerate(ordered):
            adjusted = max(adjusted, min(1., (len(zones) - rank) * zones[zone]['mcnemar_worse_p']))
            zones[zone]['holm_adjusted_p'] = adjusted
            if adjusted < ALPHA:
                reasons.append('significant_zone_regression:' + zone)
    else:
        for zone, result in zones.items():
            if result['mcnemar_worse_p'] < ALPHA:
                reasons.append('significant_zone_regression:' + zone)
    result = dict(version=version, approved=not reasons, reasons=reasons,
                  policy=policy(version), overall=overall, zones=zones)
    if version >= 5:
        result['monitor'] = {name: summarize([row for row in pairs if row['task_id'] == name])
                             for name in monitor}
        result['monitor_note'] = ('heldout cells are reported, never vetoed and never counted '
                                  'in the overall rate')
    return result
