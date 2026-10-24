"""task42 round 2 step 2: where does A's winning action rank in the student's order?

Every round-1 point whose recomputed delta is -1 (A matched the heavy teacher, B3 did not)
is strict-replayed from its archived trajectory to the search root, and ``ranked_plants`` is
recomputed with exactly the student frame / history the round-1 compare loop supplied. The
A-chosen plant's 1-based rank is recorded; a wait choice is bucketed separately.

No search, no rollouts and no heavy teacher: this only reads the round-1 archives, so it
cannot change a round-1 result. The pre-registered rule for round 2: >50% of the 71 points
ranked >10 -> B4/B5 cannot cover them and the run is skipped.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]

from b3_noninferiority import FAMILY_BY_NAME, correctness_difference, digest  # noqa: E402

MARGIN = .03
BUCKETS = ('1', '2-3', '4', '5', '6-10', '>10', 'wait', 'not_found')


def bucket(rank, status):
    if status == 'wait':
        return 'wait'
    if rank is None:
        return 'not_found'
    if rank == 1:
        return '1'
    if rank <= 3:
        return '2-3'
    if rank <= 10:
        return str(rank) if rank in (4, 5) else '6-10'
    return '>10'


def init(args):
    import search_play as t
    keep, paths = t._export_values(args.value, args.static_value)
    t._W["value_keep"] = keep
    t._init_worker(paths[0], args.student, 61, paths[1])


def ranks_for_seed(family_name, seed, points, data, args):
    import search_play as t
    from pvz_frame_store import read_frames
    from pvz_rollout_policy import ranked_plants
    from pvz_student_observation import StudentHistory
    from pvz_value_net import observation_context
    from render_episode import _frame
    from teacher_counterfactual_audit import physical
    from b3_noninferiority import open_game
    family = FAMILY_BY_NAME[family_name]
    source = Path(data) / 'trajectories' / family_name / f'seed_{seed}'
    meta = json.loads((source / 'meta.json').read_text())
    path = Path(meta['archive'])
    assert digest(path) == meta['archive_sha256']
    frames = read_frames(path)
    selected = {p['decision']: p for p in meta['points']}
    chosen = {p['decision']: p['chosen'] for p in points}
    assert set(chosen) <= set(selected), (family_name, seed, sorted(set(chosen) - set(selected)))
    last = max(chosen)
    env, obs = open_game(seed, family, args)
    history = StudentHistory(t._W['fast'].obs)
    out = []
    try:
        for i in range(last):
            assert physical(_frame(obs)) == physical(frames[i]), (family_name, seed, i)
            if i + 1 in chosen:
                t._W['context'] = observation_context(obs)
                state = {'history': history, 'action': frames[i].get('action')}
                if i:
                    state['frame'] = frames[i - 1]
                policy = t._fast_action(obs, state)
                t._W['student_history'] = history
                assert policy == frames[i + 1]['action'], ('policy replay', family_name, seed, i)
                legal = [{'type': a.get('type', 'plant'), 'packet': a['packet'], 'col': a['col'],
                          'row': a['row']}
                         for a in (obs.get('legal_actions') or {}).get('plants') or []]
                ranked = ranked_plants(t._W['fast'], t._W['student_frame'],
                                       t._W.get('student_previous_frame'), legal,
                                       context=t._W.get('context'), history=history)
                action = chosen[i + 1]
                if action['type'] != 'plant':
                    target, status = None, 'wait'
                else:
                    target = (action['packet'], action['row'], action['col'])
                    status = 'not_found'
                rank = None
                if target is not None:
                    for position, (_score, entry) in enumerate(ranked, 1):
                        if (entry['packet'], entry['row'], entry['col']) == target:
                            rank, status = position, 'ranked'
                            break
                out.append({'family': family_name, 'seed': seed, 'decision': i + 1,
                            'rank': rank, 'status': status, 'legal_plants': len(legal),
                            'student_plant': policy['type'] == 'plant',
                            'bucket': bucket(rank, status)})
            history.extra(frames[i])
            if i + 1 == last:
                break
            obs, _r, _d, _t, info = env.step(frames[i + 1]['action'])
            assert (not info.get('ok')) == bool(frames[i + 1].get('action_rejected'))
            if not info.get('ok'):
                obs, _r, _d, _t, info = env.step({'type': 'wait', 'ticks': 60})
    finally:
        env.close()
    return out


def summarize(points):
    counts = Counter(p['bucket'] for p in points)
    ranks = [p['rank'] for p in points if p['rank'] is not None]
    total = len(points)
    over = sum(1 for r in ranks if r > 10)
    return {'points': total, 'buckets': {b: counts.get(b, 0) for b in BUCKETS},
            'rank_le_4': sum(1 for r in ranks if r <= 4),
            'rank_le_5': sum(1 for r in ranks if r <= 5),
            'rank_le_10': sum(1 for r in ranks if r <= 10),
            'rank_gt_10': over,
            'rank_gt_10_fraction_of_points': over / total if total else None,
            'rank_median': sorted(ranks)[len(ranks) // 2] if ranks else None,
            'rank_max': max(ranks) if ranks else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'artifacts/task42/full')
    p.add_argument('--out', type=Path, default=ROOT / 'artifacts/task42/round2/round1_ranks.json')
    p.add_argument('--families', default='night_l14,pool_l24')
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        p.error('workers must be 1..3')
    family_names = [n for n in args.families.split(',') if n]
    for name in family_names:
        if name not in FAMILY_BY_NAME:
            p.error(f'unknown family: {name}')
    plan = json.loads((args.data / 'plan.json').read_text())
    models = plan['models']
    for role in ('student', 'value', 'static_value', 'executable'):
        assert digest(models[role]) == models['sha256'][role], f'{role} changed since round 1'
    rows = [json.loads(path.read_text()) for path in sorted((args.data / 'points').glob('*.json'))]
    target = []
    for row in rows:
        if row['family'] not in family_names:
            continue
        arms = row['arms']
        label = row['heavy']['label'] if row['heavy'] else None
        delta = correctness_difference(arms['A']['chosen'], arms['B3']['chosen'], label)
        assert delta == row['delta']
        if delta == -1:
            target.append(row)
        row['_delta'] = delta
    jobs = {}
    for row in target:
        jobs.setdefault((row['family'], row['seed']), []).append(
            {'decision': row['decision'], 'chosen': row['arms']['A']['chosen']})
    frozen = SimpleNamespace(student=models['student'], value=models['value'],
                             static_value=models['static_value'], executable=models['executable'],
                             resource_dir=str(Path.home() / '.cache/pvz-research-resources'))
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init,
                             initargs=(frozen,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(ranks_for_seed, family, seed, points, str(args.data), frozen)
                   for (family, seed), points in sorted(jobs.items())]
        for future in futures:
            part = future.result()
            results.extend(part)
            print(json.dumps({'seed_done': part[0]['seed'], 'family': part[0]['family'],
                              'points': len(part)}), flush=True)
    results.sort(key=lambda p: (p['family'], p['seed'], p['decision']))
    assert len(results) == len(target), (len(results), len(target))
    by_family = {name: [p for p in results if p['family'] == name] for name in family_names}
    round1 = {}
    for name in family_names:
        fam = [r for r in rows if r['family'] == name]
        round1[name] = {'points': len(fam), 'delta_sum': sum(r['_delta'] for r in fam)}
    per_family = {}
    for name, part in by_family.items():
        summary = summarize(part)
        coverage = {'need_delta_sum_above': -MARGIN * round1[name]['points'],
                    'round1_delta_sum': round1[name]['delta_sum']}
        for k in (4, 5):
            recovered = summary[f'rank_le_{k}']
            optimistic = round1[name]['delta_sum'] + recovered
            coverage[f'k{k}'] = {
                'recoverable_A_only_points': recovered,
                'optimistic_delta_sum': optimistic,
                'can_reach_point_estimate_above_margin': optimistic > coverage['need_delta_sum_above']}
        summary['futility_check'] = coverage
        per_family[name] = summary
    overall = summarize(results)
    overall['futility_check'] = {
        f'k{k}': {name: per_family[name]['futility_check'][f'k{k}']
                  ['can_reach_point_estimate_above_margin'] for name in family_names}
        for k in (4, 5)}
    decision = {'rule': '>50% of target points ranked >10 -> skip B4/B5',
                'rank_gt_10_fraction': overall['rank_gt_10_fraction_of_points'],
                'skip_B4_B5_by_rank': (overall['rank_gt_10_fraction_of_points'] or 0) > .5,
                'all_k_infeasible_by_optimistic_bound': all(
                    not all(overall['futility_check'][f'k{k}'][name] for name in family_names)
                    for k in (4, 5))}
    decision['skip_B4_B5'] = (decision['skip_B4_B5_by_rank']
                              or decision['all_k_infeasible_by_optimistic_bound'])
    report = {'source': str(args.data.resolve()), 'families': family_names,
              'round1': round1, 'per_family': per_family, 'overall': overall,
              'decision': decision, 'points': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=1) + '\n')
    tmp.replace(args.out)
    print(json.dumps({'per_family': per_family, 'overall': overall,
                      'decision': decision}, indent=1))


if __name__ == '__main__':
    main()
