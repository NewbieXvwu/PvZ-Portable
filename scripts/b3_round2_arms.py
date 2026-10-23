"""task42 round 2: does a larger k (student global top-4 / top-5 candidates) recover the
night / pool families that B3 failed in round 1?

Three arms on the same roots and futures: A (the unchanged formal default, static value
per-card top-1 + wait + student action + 2 extra student actions), B4 (student top-4) and
B5 (student top-5). New seeds 30064..30127 (outside round 1), night_l14 and pool_l24 only.
The measurement order rotates with `(seed + decision) % 3` (0->A,B4,B5; 1->B4,B5,A;
2->B5,A,B4). The heavy reference runs once per disagreement point when the three arms do
not all choose the same action; `delta_k = 1[Bk correct] - 1[A correct]` and equal
choices cancel exactly.

Comparison arm only: the formal collection path is untouched, `candidate_mode` stays
private to this harness.

Usage:
  python scripts/b3_round2_arms.py --out artifacts/task42/round2/smoke --games 2 --workers 3
  python scripts/b3_round2_arms.py --out artifacts/task42/round2/full --games 64 --workers 3
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')

from b3_noninferiority import (  # noqa: E402
    FAMILY_BY_NAME, HEAVY_OFFSET, HEAVY_FUTURES, HEAVY_VERIFY, LIGHT_SAMPLES,
    LIGHT_VERIFY, HORIZON, PER_CARD, STUDENT_CANDIDATES_A, MARGIN, SAMPLE_SEED,
    digest, correctness_difference, key, open_game, heavy_reference)

def init(args):
    import search_play as t
    keep, paths = t._export_values(args.value, args.static_value)
    t._W["value_keep"] = keep
    t._init_worker(paths[0], args.student, 61, paths[1])


SEED_BASE = 30064
FAMILIES = ('night_l14', 'pool_l24')
TOPKS = {'B4': 4, 'B5': 5}
ROTATIONS = (('A', 'B4', 'B5'), ('B4', 'B5', 'A'), ('B5', 'A', 'B4'))


def normal(env, obs, policy, args, decision_seed, name):
    """One light teacher arm; only candidate generation differs between arms."""
    import search_play as t
    old_mode, old_k = t._W.get('candidate_mode'), t._W.get('student_topk')
    if name == 'A':
        t._W['candidate_mode'] = 'static'
    else:
        t._W['candidate_mode'] = 'student-topk'
        t._W['student_topk'] = TOPKS[name]
    arm_args = SimpleNamespace(**vars(args))
    arm_args.student_candidates = STUDENT_CANDIDATES_A
    arm_args.per_card = PER_CARD
    original = t._halving_choice
    captured = []

    def record(*pos):
        captured.extend(deepcopy(pos[3]))
        return original(*pos)

    stats = {'searched': 0, 'overrides': 0, 'override_kinds': Counter()}
    try:
        t._halving_choice = record
        chosen = t._search(env, obs, policy, arm_args, decision_seed, stats)
    finally:
        t._halving_choice = original
        if old_mode is None:
            t._W.pop('candidate_mode', None)
        else:
            t._W['candidate_mode'] = old_mode
        if old_k is None:
            t._W.pop('student_topk', None)
        else:
            t._W['student_topk'] = old_k
    assert captured and stats['searched'] == 1
    return {'chosen': chosen, 'candidates': captured,
            'continuations': stats['halving_rollouts'] + 2 * LIGHT_VERIFY * bool(stats.get('last_verification')),
            'halving_continuations': stats['halving_rollouts'],
            'verification_continuations': 2 * LIGHT_VERIFY * bool(stats.get('last_verification'))}


def measured_arm(env, obs, policy, args, decision_seed, name):
    from pvz_cpu_usage import usage, process_usage, difference
    before_worker = usage()
    before_sim = process_usage(env._process.pid)
    result = normal(env, obs, policy, args, decision_seed, name)
    after_sim = process_usage(env._process.pid)
    after_worker = usage()
    worker, sim = difference(after_worker, before_worker), difference(after_sim, before_sim)
    assert min(worker['total_s'], sim['total_s']) >= 0
    result['cpu'] = {'worker': worker, 'simulator': sim,
                     'total_s': worker['total_s'] + sim['total_s']}
    return result


def evaluate(env, obs, policy, args, seed, point, family):
    decision_seed = seed * 10000 + point['decision'] - 1
    order = ROTATIONS[(seed + point['decision']) % 3]
    arms = {}
    for name in order:
        arms[name] = measured_arm(env, obs, policy, args, decision_seed, name)
    same = len({key(a['chosen']) for a in arms.values()}) == 1
    heavy = None if same else heavy_reference(env, obs, policy, args, decision_seed, arms)
    label = heavy['label'] if heavy else None
    deltas = {k: correctness_difference(arms['A']['chosen'], arms[k]['chosen'], label)
              for k in TOPKS}
    return {'family': family['name'], 'seed': seed, **point, 'policy': policy,
            'arms': arms, 'measurement_order': list(order),
            'actions': heavy['actions'] if heavy else [],
            'heavy': heavy, 'judge_required': not same,
            'delta_B4': deltas['B4'], 'delta_B5': deltas['B5']}


def compare(family, meta, args):
    import search_play as t
    from pvz_frame_store import read_frames
    from pvz_student_observation import StudentHistory
    from pvz_value_net import observation_context
    from teacher_counterfactual_audit import physical
    from render_episode import _frame
    path = Path(meta['archive'])
    assert digest(path) == meta['archive_sha256']
    frames = read_frames(path)
    selected = {p['decision']: p for p in meta['points']}
    if not selected:
        return {'family': family['name'], 'seed': meta['seed'], 'completed': 0,
                'strict_replay_frames': 0}
    env, obs = open_game(meta['seed'], family, args)
    history = StudentHistory(t._W['fast'].obs)
    completed = 0
    try:
        for i in range(max(selected)):
            assert physical(_frame(obs)) == physical(frames[i]), (family['name'], meta['seed'], i)
            if i + 1 in selected:
                dest = args.out / 'points' / f"{family['name']}_{meta['seed']}_{i + 1}.json"
                if not dest.exists():
                    t._W['context'] = observation_context(obs)
                    state = {'history': history, 'action': frames[i].get('action')}
                    if i:
                        state['frame'] = frames[i - 1]
                    policy = t._fast_action(obs, state)
                    t._W['student_history'] = history
                    assert policy == frames[i + 1]['action'], ('policy replay', family['name'],
                                                              meta['seed'], i)
                    row = evaluate(env, obs, policy, args, meta['seed'], selected[i + 1], family)
                    tmp = dest.with_suffix('.tmp')
                    tmp.write_text(json.dumps(row) + '\n')
                    tmp.replace(dest)
                    completed += 1
                    print(json.dumps({'point': dest.stem, 'done': completed,
                                      'pool': len(row['actions'])}), flush=True)
            history.extra(frames[i])
            if i + 1 == max(selected):
                break
            obs, _r, _d, _t, info = env.step(frames[i + 1]['action'])
            assert (not info.get('ok')) == bool(frames[i + 1].get('action_rejected'))
            if not info.get('ok'):
                obs, _r, _d, _t, info = env.step({'type': 'wait', 'ticks': 60})
    finally:
        env.close()
    return {'family': family['name'], 'seed': meta['seed'], 'completed': completed,
            'strict_replay_frames': max(selected)}


def run_game(family, seed, args):
    from b3_noninferiority import generate
    meta = generate(family, seed, args)
    return compare(family, meta, args)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--families', default=','.join(FAMILIES))
    p.add_argument('--games', type=int, default=64)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--student', default=str(MAIN / 'artifacts/student/dw4i_s1/model.pt'))
    p.add_argument('--value', default=str(MAIN / 'artifacts/value_net/v3/model.pt'))
    p.add_argument('--static-value', default=str(MAIN / 'artifacts/hard/value_branch1/model.pt'))
    p.add_argument('--executable', default=str(ROOT / 'artifacts/task42/pvz-portable'))
    p.add_argument('--resource-dir', default=str(Path.home() / '.cache/pvz-research-resources'))
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        p.error('workers must be 1..3')
    if not 1 <= args.games <= 64:
        p.error('games must be 1..64 (new seeds 30064..30127 are outside round 1)')
    names = [n for n in args.families.split(',') if n]
    unknown = [n for n in names if n not in FAMILIES]
    if unknown:
        p.error(f'round 2 covers only night_l14 and pool_l24, unknown: {unknown}')
    families = [FAMILY_BY_NAME[n] for n in names]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'points').mkdir(exist_ok=True)
    args.samples, args.horizon, args.verify_samples = LIGHT_SAMPLES, HORIZON, LIGHT_VERIFY
    args.per_card, args.student_candidates, args.min_gap = PER_CARD, STUDENT_CANDIDATES_A, 300
    args.halving = True
    jobs = [(family, SEED_BASE + i) for family in families for i in range(args.games)]
    plan = {'families': names, 'games': args.games,
            'seeds': [SEED_BASE, SEED_BASE + args.games - 1],
            'sample_seed': SAMPLE_SEED, 'heavy_offset': HEAVY_OFFSET,
            'heavy_futures': HEAVY_FUTURES, 'heavy_verify': HEAVY_VERIFY,
            'light': {'samples': LIGHT_SAMPLES, 'verify_samples': LIGHT_VERIFY,
                      'horizon': HORIZON, 'per_card': PER_CARD, 'min_gap': 300,
                      'halving': True, 'student_candidates_A': STUDENT_CANDIDATES_A,
                      'student_topk_B4': TOPKS['B4'], 'student_topk_B5': TOPKS['B5'],
                      'margin': MARGIN, 'rollout_sim_skip': 61,
                      'measurement_rotation': [list(r) for r in ROTATIONS]},
            'models': {'student': args.student, 'value': args.value,
                       'static_value': args.static_value, 'executable': args.executable,
                       'sha256': {name: digest(path) for name, path in
                                  [('student', args.student), ('value', args.value),
                                   ('static_value', args.static_value), ('executable', args.executable)]}}}
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=1) + '\n')
    print(f'{len(jobs)} games on {args.workers} workers', flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(args,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(run_game, family, seed, args) for family, seed in jobs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps({'game_complete': row}), flush=True)
    (args.out / 'compare_summary.json').write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
