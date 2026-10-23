"""Independent new-root A/B3 comparison, with exact cancellation and CPU accounting."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import static_value_compare as common
from static_value_ablation import digest, normal


def correctness_difference(a, b, heavy=None):
    """Equal choices cancel for every common reference, without knowing its label."""
    if common.key(a) == common.key(b):
        return 0
    if heavy is None:
        raise ValueError('a disagreement requires the common heavy teacher')
    return int(common.key(b) == common.key(heavy)) - int(common.key(a) == common.key(heavy))


def measured_arm(env, obs, policy, args, seed):
    from pvz_cpu_usage import usage, process_usage, difference
    before_worker = usage()
    before_sim = process_usage(env._process.pid)
    result = normal(env, obs, policy, args, seed)
    after_sim = process_usage(env._process.pid)
    after_worker = usage()
    worker, sim = difference(after_worker, before_worker), difference(after_sim, before_sim)
    assert min(worker['total_s'], sim['total_s']) >= 0
    result['cpu'] = {'worker': worker, 'simulator': sim, 'total_s': worker['total_s'] + sim['total_s']}
    return result


def mac_candidates(env, obs, policy, args, seed):
    """Preserve the original reference pool without running a third light arm."""
    import search_play as t
    old_halve = t._halving_choice
    candidates = []
    def capture(*pos):
        candidates.extend(deepcopy(pos[3]))
        return policy
    stats = {'searched': 0, 'overrides': 0, 'override_kinds': Counter()}
    try:
        t._halving_choice = capture
        env.load_value_net(t._W['static_models']['mac'], static=True)
        t._search(env, obs, policy, args, seed, stats)
    finally:
        t._halving_choice = old_halve
        env.load_value_net(t._W['static_models']['desktop'], static=True)
    assert candidates and stats['searched'] == 1 and not stats.get('halving_rollouts')
    return candidates


def heavy_reference(env, obs, policy, args, D, arms):
    import math
    import search_play as t
    root = env.snapshot()
    try:
        reference_arms = {'desktop': arms['A'], 'mac': {
            'candidates': mac_candidates(env, obs, policy, args, D)}}
        plants = [{'type': 'plant', **{k: a[k] for k in ('packet', 'row', 'col')}}
                  for a in obs['legal_actions']['plants']]
        scores = {}
        for name in ('mac', 'desktop'):
            env.load_value_net(t._W['static_models'][name], static=True)
            scores[name] = t._static_scores(env, root, plants, obs.get('sun_income_rate'))
            assert len(scores[name]) == len(plants)
        pool = common.pool_actions(scores, reference_arms, plants, 20261020 + D)
        actions = {common.key(a): a for a in pool}
        for arm in arms.values():
            for action in arm['candidates']:
                actions.setdefault(common.key(action), action)
        actions = [actions[k] for k in sorted(actions)]
        truthD = D + 33333333
        values, totals = [], []
        for action in actions:
            leaves = t._continuations(env, obs, root,
                [(action, (truthD * 1000 + k) % 2**32) for k in range(64)], args, {})
            v = t._leaf_values(leaves)
            assert len(v) == 64 and all(x is not None and math.isfinite(x) and 0 <= x <= 1 for x in v)
            values.append(v)
            terminal = [x for leaf, x in zip(leaves, v) if t._score_terminal_or_value(leaf) is not None]
            other = [x for leaf, x in zip(leaves, v) if t._score_terminal_or_value(leaf) is None]
            totals.append(sum(terminal + other) / 64)
        anchor = common.key(policy if policy['type'] == 'plant' else t.NOOP)
        anchor_index = next(i for i, a in enumerate(actions) if common.key(a) == anchor)
        best = max(range(len(actions)), key=lambda i: totals[i])
        heavy_args = SimpleNamespace(**vars(args))
        heavy_args.samples, heavy_args.verify_samples = 64, 18
        stats = {'overrides': 0, 'override_kinds': Counter()}
        label = t._verified_choice(env, obs, root, actions, totals, [64] * len(actions), best,
                                   anchor_index, policy, heavy_args, truthD, stats)
        return {'label': label, 'actions': actions, 'truth_scores': values,
                'truth_continuations': 64 * len(actions),
                'verification_continuations': 36 * bool(stats.get('last_verification')),
                'verification': stats.get('last_verification')}
    finally:
        env.load_value_net(t._W['static_models']['desktop'], static=True)
        env.restore(root)
        env.release_snapshot(root)


def evaluate(env, obs, policy, args, seed, point):
    import search_play as t
    D = seed * 10000 + point['decision'] - 1
    env.load_value_net(t._W['static_models']['desktop'], static=True)
    arms = {}
    order = ('A', 'B3') if (seed + point['decision']) % 2 == 0 else ('B3', 'A')
    for name in order:
        arm_args = SimpleNamespace(**vars(args))
        arm_args.candidate_generator = 'static' if name == 'A' else 'student-topk'
        arm_args.student_candidates = 2 if name == 'A' else 3
        arms[name] = measured_arm(env, obs, policy, arm_args, D)
    a, b = arms['A']['chosen'], arms['B3']['chosen']
    disagree = common.key(a) != common.key(b)
    heavy = heavy_reference(env, obs, policy, args, D, arms) if disagree else None
    label = heavy['label'] if heavy else None
    return {'seed': seed, **point, 'policy': policy, 'arms': arms, 'measurement_order': order,
            'actions': heavy['actions'] if heavy else [], 'heavy': heavy,
            'judge_required': disagree, 'delta': correctness_difference(a, b, label)}


def run_game(seed, args):
    meta = common.generate(seed, args)
    assert len(meta['points']) == 8, ('fewer than eight eligible roots', seed)
    result = common.compare(meta, args, evaluator=evaluate)
    return {'trajectory': meta, 'comparison': result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', choices=['first', 'second'], required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        p.error('workers must be 1..4')
    plan = json.loads((ROOT / 'MEMORY/docs/static-value-ablation-ni-plan.json').read_text())
    for name, record in plan['models'].items():
        assert digest(record['path']) == record['sha256'], name
    for attr, name in [('value', 'leaf'), ('student', 'student'), ('desktop', 'desktop'),
                       ('mac', 'mac'), ('manifest', 'manifest'), ('executable', 'simulator')]:
        setattr(args, attr, plan['models'][name]['path'])
    args.resource_dir = str(Path.home() / '.cache/pvz-research-resources')
    args.samples, args.horizon, args.verify_samples = 3, 2500, 9
    args.per_card, args.student_candidates, args.shovel_candidates = 1, 2, 0
    args.min_gap, args.halving, args.margin = 300, True, .02
    args.candidate_generator = 'static'
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'points').mkdir(exist_ok=True)
    seeds = plan['sampling']['seeds'][:128] if args.batch == 'first' else plan['sampling']['seeds'][128:]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=common.init, initargs=(args,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_game, s, args) for s in seeds]):
            row = future.result()
            results.append(row)
            print(json.dumps({'game_complete': row['comparison']}), flush=True)
    (args.out / (args.batch + '_summary.json')).write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
