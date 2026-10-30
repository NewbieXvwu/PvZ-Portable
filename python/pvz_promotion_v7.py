"""task179: shared-seed two-arm group sequential gate and persisted staircase.

Four balanced looks, Bonferroni spending over looks and candidate arms. The
non-roof Holm family and separate roof .01 family are retained; low-win tasks
use final_wave (normalized by the episode's wave_count in the overall score).
Endpoint choice is frozen from the preceding gate, never selected on this data.
"""
from __future__ import annotations

import copy
import math
from statistics import NormalDist

import pvz_promotion as legacy

VERSION = 7
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


def policy():
    result = legacy.policy()
    result.update(version=VERSION, episode_count=3 * result['pair_count'],
                  overall_interval='paired score normal interval with variance floor 1/n; '
                                   'Bonferroni alpha=.05/(4 looks * 2 candidates)',
                  overall_endpoint='win, or final_wave/wave_count for frozen low-win tasks; won=1',
                  drift_guard='overall delta >= 0',
                  sequential=dict(looks=list(LOOKS), arms=2, spending='equal Bonferroni across looks and arms',
                                  stop='both candidates pass or are clearly rejected at the same look; '
                                       'at cap reject any unresolved candidate'),
                  panel='task180 seeded staircase, persisted in curriculum_state.gate_panel',
                  zone_veto='same four-zone Holm family at .05 and roof Wilcoxon at .01, '
                            'each spent over four looks and two candidates; low-win endpoints use Wilcoxon')
    return result


def initial_panel(source):
    return {t['task_id']: dict(multiplier=2. if t['level'] == 46 else INITIAL_MULTIPLIERS.get(t['level'], 1.),
                               low_win=t['level'] in LOW_LEVELS,
                               **(dict(demo_task=DEMO_TASKS[t['level']][0],
                                       demo_index=DEMO_TASKS[t['level']][1])
                                  if t['level'] in DEMO_TASKS else {}))
            for t in source['tasks'] if t['task_id'] not in legacy.V5_MONITOR_TASKS}


def freeze_panel(source, seed_block, calibration, demos):
    panel = legacy.freeze_panel(source, seed_block=seed_block)
    panel['panel_version'] = 'v7-task179'
    demo_table = {t['task_id']: t for t in demos.get('tasks', [])}
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
    panel['purpose'] = policy()['panel']
    return panel


def next_panel(calibration, manifest, outcomes):
    result = copy.deepcopy(calibration)
    for task in manifest['tasks']:
        key = task['gate_key']
        if key not in result:
            continue
        setting = result[key]
        rates = {side: sum(row['won'] for row in rows[task['task_id']].values()) /
                       len(rows[task['task_id']]) for side, rows in outcomes.items()}
        setting['rates'] = rates
        setting['low_win'] = (rates['incumbent'] <= .2 or manifest['groups'][task['task_id']] == 'roof')
        direction = 1 if min(rates.values()) >= .9 else -1 if max(rates.values()) <= .1 else 0
        if setting.get('demo_task'):
            setting['demo_index'] = max(0, min(len(task['demo_start']['decisions']) - 1,
                                              setting['demo_index'] + direction))
        else:
            index = MULTIPLIERS.index(setting['multiplier'])
            setting['multiplier'] = MULTIPLIERS[max(0, min(len(MULTIPLIERS) - 1, index + direction))]
    return result


def score(row, side):
    if row.get('endpoint') != 'waves':
        return float(row[side])
    return 1. if row[side] else min(1., max(0., row[side + '_wave'] / row['wave_count']))


def summary(pairs, alpha):
    binary = legacy.summarize(pairs)
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
    alpha = legacy.ALPHA / (len(LOOKS) * 2)
    scored = [r for r in pairs if r['task_id'] not in manifest['monitor']]
    overall = summary(scored, alpha)
    zones = {zone: summary([r for r in scored if r['zone'] == zone], alpha)
             for zone in sorted({r['zone'] for r in scored})}
    reasons = []
    if overall['score_ci'][0] <= -legacy.MARGIN:
        reasons.append('overall_ci_lower_not_above_minus_3pp')
    if overall['delta_rate'] < 0:
        reasons.append('overall_point_estimate_negative')
    for zone, result in zones.items():
        rows = [r for r in scored if r['zone'] == zone]
        if zone == 'roof':
            waves = legacy.summarize_waves(rows)
            waves.update(alpha=legacy.ROOF_WAVE_ALPHA / (len(LOOKS) * 2))
            waves['veto'] = waves['wilcoxon_worse_p'] < waves['alpha']
            result['waves'] = waves
            if waves['veto']:
                reasons.append('significant_roof_wave_regression:roof')
        elif result['wave_pairs']:
            result['endpoint_worse_p'] = legacy.wilcoxon_worse(
                [score(r, 'candidate') - score(r, 'incumbent') for r in rows])['wilcoxon_worse_p']
        else:
            result['endpoint_worse_p'] = result['mcnemar_worse_p']
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
    return dict(version=VERSION, policy=policy(), approved=approved, reasons=reasons,
                overall=overall, zones=zones, status=status,
                monitor={name: legacy.summarize([r for r in pairs if r['task_id'] == name])
                         for name in manifest['monitor']},
                monitor_note='heldout reported only; never vetoes or enters overall score')


def ab_summary(pairs):
    scored = [r for r in pairs if r['zone'] != 'heldout']
    alpha = legacy.ALPHA
    return dict(direction='A-B', overall=summary(scored, alpha),
                zones={zone: summary([r for r in scored if r['zone'] == zone], alpha)
                       for zone in sorted({r['zone'] for r in scored})})
