"""Production v7 shared-seed sequential gate, staircase and paired statistics."""
from __future__ import annotations
import copy
import math
from pathlib import Path
from itertools import groupby
from statistics import NormalDist, median

from pvz_env import deck_card

GATE_VERSION = 8
ROOF_WAVE_ALPHA = .01
ALPHA = .05
MARGIN = .03
# task223: a wave endpoint must not hide a significant win loss.  With at least
# this many discordant win pairs the exact McNemar test is meaningful; below it,
# the zone's wins are genuinely too rare to compare and waves may stand in.
ZONE_WIN_GUARD_ALPHA = .05
ZONE_WIN_GUARD_DISCORDANT = 6
# task128 (gate v5): the panel is the adventure set, not synthetic curriculum cells.
# Levels come from configs/adventure_scoreboard.json (task103: 45 levels, 1.0x,
# scoreboard decks) and enter only when the r11 student or the teacher has a win
# there (scripts/make_gate_v5_panel.py). The original scoreboard qualifies roof
# L41 (teacher 3/16) and L46 (2/16), despite the later unattributed task115 summary.
# The two registered heldout cells run for monitoring only.
PANEL_TASKS = (
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
MONITOR_TASKS = ('day_l7_eval_d7p_holdout1', 'day_l9_eval_d7p_holdout2')
SEEDS_PER_TASK = 45
MONITOR_SEEDS_PER_TASK = 96
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

VERSION = GATE_VERSION
V8_PANEL_VERSION = 'v8'
LOOKS = (0.25, 0.5, 0.75, 1.0)
MULTIPLIERS = (.5, .75, 1., 1.25, 1.5, 1.75, 2., 2.5, 3.)
# task180 evidence; retain the registered decks. Nonmatching fog demos are not substituted.
INITIAL_MULTIPLIERS = {1: 2., 3: 2., 4: 2., 6: 2., 8: 2., 10: 2.5,
                       11: 2., 13: 1.5, 16: 1.5, 18: 1.5, 20: 2.5,
                       21: 1.25, 31: 1.5, 36: 1.25, 38: 1.25,
                       33: .75, 34: .75, 39: .75, 41: .75}
LOW_LEVELS = {33, 34, 39, 40, 41, 46}
DEMO_TASKS = {40: ('demo_storm_40_orig_m100', 0),
              46: ('demo_roof_46_doom_m200', 3)}


def _panel_tasks(document=None):
    """Registered task order and pair counts, v5 legacy or v8 panel document."""
    if document is None or document.get('panel_version') == 'v5':
        return [dict(zone=zone, task_id=name, seeds=count, monitor=name in MONITOR_TASKS)
                for zone, name, count in PANEL_TASKS]
    if document.get('panel_version') not in (V8_PANEL_VERSION, 'v8-task222'):
        raise ValueError(f"unknown panel_version: {document.get('panel_version')!r}")
    monitor = list(document.get('monitor') or ())
    groups = document.get('groups') or {}
    rows = []
    for task in document.get('tasks') or []:
        name = task.get('task_id')
        if not name or name not in groups:
            raise ValueError(f'v8 panel task has no group: {name!r}')
        is_monitor = name in monitor
        count = (document.get('monitor_seeds_per_task', MONITOR_SEEDS_PER_TASK) if is_monitor
                 else task.get('pairs', len(task.get('seeds') or ()) or document.get('seeds_per_task', SEEDS_PER_TASK)))
        if type(count) is not int or count < 1:
            raise ValueError(f'v8 panel task {name} has invalid pair count {count!r}')
        rows.append(dict(zone=groups[name], task_id=name, seeds=count, monitor=is_monitor))
    missing = set(monitor) - {row['task_id'] for row in rows}
    if missing:
        raise ValueError(f'v8 panel monitor cells are missing: {sorted(missing)}')
    return rows


def policy(document=None):
    rows = _panel_tasks(document)
    v8 = document is not None and document.get('panel_version') in (V8_PANEL_VERSION, 'v8-task222')
    roof = [row['task_id'] for row in rows if row['zone'] == 'roof']
    result = dict(version=GATE_VERSION, alpha=ALPHA, noninferiority_margin=MARGIN,
                overall_interval='Newcombe paired method 10, two-sided 95%',
                zone_veto='one-sided exact McNemar p across the four non-roof adventure zones; '
                          'Holm–Bonferroni step-down correction, family-wise alpha = 0.05; '
                          'veto while ordered p < 0.05 / (group_count - rank + 1); '
                          'a significant win loss (p < .05, >= 6 discordant pairs) also vetoes '
                          'when the zone uses wave endpoints',
                zone_win_guard=dict(alpha=ZONE_WIN_GUARD_ALPHA,
                                    min_discordant=ZONE_WIN_GUARD_DISCORDANT,
                                    rule='waves substitute for wins only below the discordant '
                                         'threshold; otherwise the win McNemar p decides'),
                roof_veto=dict(endpoint='candidate_wave - incumbent_wave (final_wave)',
                               test='one-sided exact Wilcoxon signed-rank, alternative less',
                               alpha=ROOF_WAVE_ALPHA,
                               unit=f"roof zone, pooling {sum(row['seeds'] for row in rows if row['zone'] == 'roof')} pairs",
                               zeros='discard', ties='average ranks',
                               distribution='exact conditional sign-flip, inclusive lower tail',
                               tasks=roof,
                               win_first='wins first when win discordant >= 6; wave test is '
                                         'the fallback only below that'),
                pair_count=sum(row['seeds'] for row in rows),
                episode_count=2 * sum(row['seeds'] for row in rows),
                seeds_per_task=SEEDS_PER_TASK if not v8 else 'per task (v8 panel pairs)',
                seed_assignment=('1728 adventure pairs evenly divided: first 18 tasks take 46, '
                                 'remaining 20 take 45' if not v8 else
                                 f"{sum(row['seeds'] for row in rows if not row['monitor'])} adventure pairs: "
                                 'first 18 tasks take 46, the rest 45'),
                monitor_seeds_per_task=MONITOR_SEEDS_PER_TASK,
                seed_pool=list(SEED_POOLS_V2[0]),
                seed_ledger=SEED_LEDGER_DEFAULT,
                seed_reuse='never; ledger-allocated contiguous block per round, '
                           'exhaustion aborts the gate instead of wrapping',
                incumbent_rerun='both sides are evaluated on the round seeds; '
                                'cached incumbent outcomes are never used',
                panel=("task222 gate v8: the user's in-game deck picks, task208 first good decks "
                       'elsewhere; every terrain zone represented' if v8 else
                       'adventure levels (original 1.0x, scoreboard decks) where the r11 student or '
                       'the teacher has a recorded win; all-loss cells carry no regression signal'),
                heldout='monitor only: 96 pairs each, written into the round report, '
                        'never vetoes and never enters the overall rate',
                tasks=rows)

    result.update(version=VERSION, episode_count=3 * result['pair_count'],
                  overall_interval='paired score normal interval with variance floor 1/n; '
                                   'Bonferroni alpha=.05/(4 looks * 2 candidates)',
                  overall_endpoint='win, or final_wave/wave_count for frozen low-win tasks; won=1',
                  drift_guard='overall delta >= 0',
                  sequential=dict(looks=list(LOOKS), arms=2, spending='equal Bonferroni across looks and arms',
                                  stop='both candidates pass or are clearly rejected at the same look; '
                                       'at cap reject any unresolved candidate'),
                  panel=('task180 seeded staircase, persisted in curriculum_state.gate_panel' if not v8
                         else 'task222 v8 deck panel with the task180 staircase, persisted in '
                              'curriculum_state.gate_panel'),
                  zone_veto='same four-zone Holm family at .05 and roof Wilcoxon at .01, '
                            'each spent over four looks and two candidates; low-win endpoints use Wilcoxon')
    return result


def _registered_panel_v8(document, seed_block):
    if seed_block is None or [int(s) for s in seed_block] != list(seed_block) or len(seed_block) != 2:
        raise ValueError('gate v8 requires the ledger-allocated seed block')
    start, end = int(seed_block[0]), int(seed_block[1])
    if not any(low <= start and end <= high for low, high in SEED_POOLS_V2):
        raise ValueError(f'seed block [{start}, {end}) is outside the gate pools '
                         f'{[list(pool) for pool in SEED_POOLS_V2]}')
    rows = _panel_tasks(document)
    total = sum(row['seeds'] for row in rows)
    if end - start != total:
        raise ValueError(f'seed block [{start}, {end}) has the wrong size for gate v8 '
                         f'(needs {total} seeds)')
    table = {task['task_id']: task for task in document['tasks']}
    tasks, groups, cursor = [], {}, start
    for row in rows:
        name = row['task_id']
        task = copy.deepcopy(table[name])
        if task.get('wave_cap') is not None:
            raise ValueError(f'promotion must use complete evaluation games: {name}')
        if float(task.get('zombie_count_multiplier', 1.)) != 1:
            raise ValueError(f'promotion must register original difficulty; calibration sets it: {name}')
        deck = task.get('deck') or ()
        # Raw card identities: (48, 1) is a different slot from the ordinary 1.
        cards = [deck_card(card) for card in deck]
        if deck and (not 6 <= len(deck) <= 10 or len(set(cards)) != len(cards)):
            raise ValueError(f'v8 panel deck must be 6..10 distinct cards: {name}')
        task['seeds'] = list(range(cursor, cursor + row['seeds']))
        cursor += row['seeds']
        tasks.append(task)
        groups[name] = row['zone']
    if cursor != end:
        raise ValueError('gate v8 seed assignment is not a disjoint partition of the block')
    return dict(schema_version=1, panel_version='v8-task222', groups=groups, tasks=tasks,
                seed_block=[start, end], monitor=list(document.get('monitor') or ()),
                purpose='task222 v8 user/task208 deck panel; roof wave Wilcoxon at 0.01, '
                        'four non-roof binary zones Holm at 0.05')


def _registered_panel(document, seed_block):
    if document.get('panel_version') == V8_PANEL_VERSION:
        return _registered_panel_v8(document, seed_block)
    if document.get('panel_version') != 'v5':
        raise ValueError('promotion requires the registered v5 or v8 panel')
    if seed_block is None or [int(s) for s in seed_block] != list(seed_block) or len(seed_block) != 2:
        raise ValueError('gate v5 requires the ledger-allocated seed block')
    start, end = int(seed_block[0]), int(seed_block[1])
    if not any(low <= start and end <= high for low, high in SEED_POOLS_V2):
        raise ValueError(f'seed block [{start}, {end}) is outside the gate pools '
                         f'{[list(pool) for pool in SEED_POOLS_V2]}')
    total = sum(count for _, _, count in PANEL_TASKS)
    if end - start != total:
        raise ValueError(f'seed block [{start}, {end}) has the wrong size for gate v5 '
                         f'(needs {total} seeds)')
    table = {task['task_id']: task for task in document['tasks']}
    tasks, groups, cursor = [], {}, start
    for zone, name, count in PANEL_TASKS:
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
    return dict(schema_version=1, panel_version='v7-task179', groups=groups, tasks=tasks,
                seed_block=[start, end], monitor=list(MONITOR_TASKS),
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


def initial_panel(source):
    if source.get('panel_version') == V8_PANEL_VERSION:
        initial = {str(level): value for level, value in (source.get('initial_multipliers') or {}).items()}
        low = set(source.get('initial_low_win') or ())
        result = {}
        for task in source['tasks']:
            name = task['task_id']
            if name in (source.get('monitor') or ()):
                continue
            level = int(task['level'])
            result[name] = dict(multiplier=float(initial.get(str(level), INITIAL_MULTIPLIERS.get(level, 1.))),
                                low_win=level in low)
        return result
    return {t['task_id']: dict(multiplier=2. if t['level'] == 46 else INITIAL_MULTIPLIERS.get(t['level'], 1.),
                               low_win=t['level'] in LOW_LEVELS,
                               **(dict(demo_task=DEMO_TASKS[t['level']][0],
                                       demo_index=DEMO_TASKS[t['level']][1])
                                  if t['level'] in DEMO_TASKS else {}))
            for t in source['tasks'] if t['task_id'] not in MONITOR_TASKS}


def freeze_panel(source, seed_block, calibration, demos):
    is_v8 = source.get('panel_version') == V8_PANEL_VERSION
    panel = _registered_panel(source, seed_block)
    panel['panel_version'] = 'v8-task222' if is_v8 else 'v7-task179'
    demo_table = {t['task_id']: t for t in (demos or {}).get('tasks', [])}
    for task in panel['tasks']:
        key = task['task_id']
        setting = calibration.get(key, {})
        task['gate_key'] = key
        task['endpoint'] = 'waves' if setting.get('low_win') else 'win'
        if setting.get('demo_task'):
            definition = copy.deepcopy(demo_table[setting['demo_task']])
            seeds = task['seeds']
            task.clear()
            task.update(definition, gate_seeds=seeds, gate_key=key,
                        endpoint='waves' if setting.get('low_win') else 'win')
            task['task_id'] += '_gate7'
            task['demo_start']['index'] = setting['demo_index']
        else:
            task['zombie_count_multiplier'] = setting.get('multiplier', 1.)
    panel['groups'] = {t['task_id']: source['groups'][t['gate_key']] for t in panel['tasks']}
    panel['purpose'] = policy(source)['panel']
    return panel


def _advance_setting(setting, rate, roof, base_multiplier=None):
    """One staircase step for the side that will be incumbent after this gate."""
    setting['low_win'] = rate <= .2 or roof
    index = MULTIPLIERS.index(base_multiplier if base_multiplier is not None
                              else setting.get('multiplier', 1.))
    direction = 1 if rate >= .9 else -1 if rate <= .1 else 0
    setting['multiplier'] = MULTIPLIERS[max(0, min(len(MULTIPLIERS) - 1, index + direction))]
    return setting


def next_panel(calibration, manifest, outcomes, incumbent_side='incumbent'):
    """Advance the staircase for the CURRENT incumbent after this gate.

    ``incumbent_side`` is 'A'/'B' when that candidate was promoted, otherwise
    'incumbent'.  The old code always read outcomes['incumbent'], which after a
    promotion described the previous student; that left stale low_win flags and
    multipliers until another gate.  Reusing the promoted candidate's same-gate
    rows costs no extra games.
    """
    result = copy.deepcopy(calibration)
    for task in manifest['tasks']:
        key = task['gate_key']
        if key not in result:
            continue
        setting = result[key]
        rates = {side: sum(row['won'] for row in rows[task['task_id']].values()) /
                       len(rows[task['task_id']]) for side, rows in outcomes.items()}
        setting['rates'] = rates
        current = rates.get(incumbent_side, rates['incumbent'])
        roof = manifest['groups'][task['task_id']] == 'roof'
        if setting.get('demo_task'):
            setting['low_win'] = current <= .2 or roof
            direction = 1 if current >= .9 else -1 if current <= .1 else 0
            setting['demo_index'] = max(0, min(len(task['demo_start']['decisions']) - 1,
                                              setting['demo_index'] + direction))
        else:
            _advance_setting(setting, current, roof)
    return result


def recalibrate_panel(pre_calibration, post_calibration, incumbent_side):
    """Redo the staircase for a promotion using already-stored gate rates.

    Startup migration for states written by the old next_panel: the post-gate
    rates contain the promoted side's rows, and pre_calibration is the frozen
    pre-gate setting, so no games are replayed.
    """
    result = copy.deepcopy(post_calibration)
    for key, setting in result.items():
        rate = (setting.get('rates') or {}).get(incumbent_side)
        if rate is None:
            continue
        pre = pre_calibration.get(key) or {}
        if pre.get('demo_task'):
            continue
        _advance_setting(setting, rate, '_roof_' in key,
                         base_multiplier=pre.get('multiplier', setting.get('multiplier', 1.)))
    return result


def score(row, side):
    if row.get('endpoint') != 'waves':
        return float(row[side])
    return 1. if row[side] else min(1., max(0., row[side + '_wave'] / row['wave_count']))


def summary(pairs, alpha):
    binary = summarize(pairs)
    differences = [score(row, 'candidate') - score(row, 'incumbent') for row in pairs]
    n = len(differences)
    delta = sum(differences) / n
    variance = sum((d - delta)**2 for d in differences) / max(1, n - 1)
    # Avoid zero-width evidence when every observed pair agrees.
    se = math.sqrt(max(variance, 1 / n) / n)
    radius = NormalDist().inv_cdf(1 - alpha / 2) * se
    binary.update(win_delta_rate=binary['delta_rate'], delta_rate=delta,
                  score_ci=[max(-1., delta - radius), min(1., delta + radius)],
                  wave_pairs=sum(row.get('endpoint') == 'waves' for row in pairs))
    return binary


def decide(pairs, manifest, look):
    alpha = ALPHA / (len(LOOKS) * 2)
    scored = [r for r in pairs if r['task_id'] not in manifest['monitor']]
    overall = summary(scored, alpha)
    zones = {zone: summary([r for r in scored if r['zone'] == zone], alpha)
             for zone in sorted({r['zone'] for r in scored})}
    reasons = []
    if overall['score_ci'][0] <= -MARGIN:
        reasons.append('overall_ci_lower_not_above_minus_3pp')
    if overall['delta_rate'] < 0:
        reasons.append('overall_point_estimate_negative')
    for zone, result in zones.items():
        rows = [r for r in scored if r['zone'] == zone]
        win_p = result['mcnemar_worse_p']
        discordant = result['gain'] + result['harm']
        result['win_discordant'] = discordant
        if zone == 'roof':
            waves = summarize_waves(rows)
            waves.update(alpha=ROOF_WAVE_ALPHA / (len(LOOKS) * 2))
            result['waves'] = waves
            result['wave_test_p'] = waves['wilcoxon_worse_p']
            # task223b: roof follows the same win-first rule.  Its wave test
            # only stands in while the win comparison is underpowered.
            if discordant >= ZONE_WIN_GUARD_DISCORDANT:
                result['win_guard_veto'] = win_p < ZONE_WIN_GUARD_ALPHA
                result['endpoint_used'] = 'win'
                result['endpoint_worse_p'] = win_p
                waves['veto'] = False
                if result['win_guard_veto']:
                    reasons.append('significant_win_regression:roof')
            else:
                result['win_guard_veto'] = False
                result['endpoint_used'] = 'waves'
                result['endpoint_worse_p'] = waves['wilcoxon_worse_p']
                waves['veto'] = waves['wilcoxon_worse_p'] < waves['alpha']
                if waves['veto']:
                    reasons.append('significant_roof_wave_regression:roof')
            continue
        wave_p = None
        if result['wave_pairs']:
            wave_p = wilcoxon_worse(
                [score(r, 'candidate') - score(r, 'incumbent') for r in rows])['wilcoxon_worse_p']
            result['wave_test_p'] = wave_p
        # task223: waves may substitute for wins only when the win comparison is
        # underpowered.  With enough discordant win pairs, a significant win loss
        # vetoes the zone even when wave progress is flat.
        result['win_guard_veto'] = bool(
            wave_p is not None and discordant >= ZONE_WIN_GUARD_DISCORDANT
            and win_p < ZONE_WIN_GUARD_ALPHA)
        if result['win_guard_veto']:
            reasons.append('significant_win_regression:' + zone)
        if wave_p is not None and discordant < ZONE_WIN_GUARD_DISCORDANT:
            result['endpoint_used'] = 'waves'
            result['endpoint_worse_p'] = wave_p
        else:
            result['endpoint_used'] = 'win'
            result['endpoint_worse_p'] = win_p
    adjusted = 0.
    ordered = sorted((z for z in zones if z != 'roof'), key=lambda z: (zones[z]['endpoint_worse_p'], z))
    for rank, zone in enumerate(ordered):
        adjusted = max(adjusted, min(1., (len(ordered) - rank) * zones[zone]['endpoint_worse_p']))
        zones[zone]['holm_adjusted_p'] = adjusted
        if adjusted < alpha:
            reasons.append('significant_zone_regression:' + zone)
    approved = not reasons
    veto = any(reason.startswith('significant_') for reason in reasons)
    clear_reject = veto or overall['score_ci'][1] < 0
    status = 'pass' if approved else 'reject' if clear_reject or look == len(LOOKS) else 'continue'
    panel_policy = policy(manifest) if manifest.get('panel_version') in (V8_PANEL_VERSION, 'v8-task222') else policy()
    return dict(version=VERSION, policy=panel_policy, approved=approved, reasons=reasons,
                overall=overall, zones=zones, status=status,
                monitor={name: summarize([r for r in pairs if r['task_id'] == name])
                         for name in manifest['monitor']},
                monitor_note='heldout reported only; never vetoes or enters overall score')


def ab_summary(pairs):
    scored = [r for r in pairs if r['zone'] != 'heldout']
    alpha = ALPHA
    return dict(direction='A-B', overall=summary(scored, alpha),
                zones={zone: summary([r for r in scored if r['zone'] == zone], alpha)
                       for zone in sorted({r['zone'] for r in scored})})
