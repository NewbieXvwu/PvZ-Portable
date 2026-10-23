"""Frozen-root candidate ablation; never execute a search teacher's labels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
import static_value_compare as common

ARMS = {'A': ('static', 2), 'B2': ('student-topk', 2),
        'B3': ('student-topk', 3), 'B4': ('student-topk', 4),
        'C': ('student-per-card', 2)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normal(env, obs, policy, args, decision_seed):
    """Instrument actual simulator calls, including verification, not a cost formula."""
    import search_play as t
    original_halving, original_rollout = t._halving_choice, t._rollout_policy
    candidates, jobs = [], []
    def capture(*pos):
        candidates.extend(deepcopy(pos[3]))
        return original_halving(*pos)
    def rollout(env, obs, first, until_tick, root=None, seed=None):
        jobs.append([common.key(first), seed])
        return original_rollout(env, obs, first, until_tick, root, seed)
    stats = {'searched': 0, 'overrides': 0, 'override_kinds': Counter()}
    start = time.perf_counter()
    try:
        t._halving_choice, t._rollout_policy = capture, rollout
        chosen = t._search(env, obs, policy, args, decision_seed, stats)
    finally:
        t._halving_choice, t._rollout_policy = original_halving, original_rollout
    assert stats['searched'] == 1 and candidates
    assert len(jobs) == stats['halving_rollouts'] + 18 * bool(stats.get('last_verification'))
    return {'chosen': chosen, 'candidates': candidates, 'continuations': len(jobs),
            'halving_continuations': stats['halving_rollouts'],
            'verification_continuations': 18 * bool(stats.get('last_verification')),
            'jobs_sha256': hashlib.sha256(json.dumps(jobs).encode()).hexdigest(),
            'seconds': time.perf_counter() - start}


def expanded_reference(env, obs, policy, args, decision_seed, frozen, arms):
    """Reuse frozen 64-future values; evaluate only new actions at the same seeds."""
    import search_play as t
    lookup = {common.key(a): (a, scores) for a, scores in zip(frozen['actions'], frozen['truth_scores'])}
    missing = {common.key(a): a for arm in arms.values() for a in arm['candidates']
               if common.key(a) not in lookup}
    root = env.snapshot()
    truth_seed = decision_seed + 33333333
    try:
        for action_key, action in sorted(missing.items()):
            leaves = t._continuations(env, obs, root,
                [(action, (truth_seed * 1000 + k) % 2**32) for k in range(64)], args, {})
            scores = t._leaf_values(leaves)
            assert len(scores) == 64 and all(v is not None and math.isfinite(v) and 0 <= v <= 1 for v in scores)
            lookup[action_key] = action, scores
        actions, scores = zip(*(lookup[k] for k in sorted(lookup)))
        # A deterministic correctly-rounded sum avoids order effects when combining
        # cached future-order arrays and newly scored arrays. Primary labels stay frozen.
        means = [math.fsum(v) / 64 for v in scores]
        anchor = common.key(policy if policy['type'] == 'plant' else t.NOOP)
        anchor_index = next(i for i, a in enumerate(actions) if common.key(a) == anchor)
        best = max(range(len(actions)), key=lambda i: means[i])
        heavy_args = SimpleNamespace(**vars(args))
        heavy_args.samples, heavy_args.verify_samples = 64, 18
        stats = {'overrides': 0, 'override_kinds': Counter()}
        heavy = t._verified_choice(env, obs, root, list(actions), means, [64] * len(actions),
                                  best, anchor_index, policy, heavy_args, truth_seed, stats)
        return {'label': heavy, 'actions': list(actions), 'truth_scores': list(scores),
                'new_actions': [missing[k] for k in sorted(missing)],
                'new_continuations': 64 * len(missing),
                'verification_continuations': 36 * bool(stats.get('last_verification')),
                'verification': stats.get('last_verification')}
    finally:
        env.restore(root)
        env.release_snapshot(root)


def evaluate(env, obs, policy, args, seed, point):
    import search_play as t
    source = args.source / 'points' / f"{seed}_{point['decision']}.json"
    assert digest(source) == args.source_hashes[source.name]
    frozen = json.loads(source.read_text())
    assert common.key(policy) == common.key(frozen['policy'])
    D = seed * 10000 + point['decision'] - 1
    env.load_value_net(t._W['static_models']['desktop'], static=True)
    arms = {}
    for name, (generator, count) in ARMS.items():
        arm_args = SimpleNamespace(**vars(args))
        arm_args.candidate_generator, arm_args.student_candidates = generator, count
        for suffix, offset in [('', 0), ('_noise', 7777777)]:
            arm = normal(env, obs, policy, arm_args, D + offset)
            if name == 'A':
                old = frozen['normal_arms']['desktop' + suffix]
                assert arm['candidates'] == old['candidates'], (seed, point, suffix, 'candidates')
                assert arm['chosen'] == old['chosen'], (seed, point, suffix, 'chosen')
                assert arm['continuations'] == old['continuations'], (seed, point, suffix, 'cost')
            arms[name + suffix] = arm
        assert arms[name]['candidates'] == arms[name + '_noise']['candidates']
    expanded = expanded_reference(env, obs, policy, args, D, frozen, arms)
    return {'seed': seed, **point, 'policy': policy, 'actions': expanded['actions'],
            'arms': arms, 'heavy_label': frozen['heavy_label'],
            'source_sha256': digest(source), 'expanded': expanded}


def run_game(meta, args):
    return common.compare(meta, args, evaluator=evaluate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'artifacts/cuda/task12/comparison')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        p.error('workers must be 1..4')
    plan = json.loads((ROOT / 'MEMORY/docs/static-value-ablation-plan.json').read_text())
    for name, record in plan['models'].items():
        assert digest(record['path']) == record['sha256'], name
    args.source_hashes = plan['source_points_sha256']
    assert digest(args.source / 'generate_summary.json') == plan['selection_sha256']
    for attr, name in [('value', 'leaf'), ('student', 'student'), ('desktop', 'desktop'),
                       ('mac', 'mac'), ('manifest', 'manifest'), ('executable', 'simulator')]:
        setattr(args, attr, plan['models'][name]['path'])
    args.resource_dir = str(Path.home() / '.cache/pvz-research-resources')
    args.samples, args.horizon, args.verify_samples = 3, 2500, 9
    args.per_card, args.student_candidates, args.shovel_candidates = 1, 2, 0
    args.min_gap, args.halving, args.margin = 300, True, .02
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'points').mkdir(exist_ok=True)
    jobs = json.loads((args.source / 'generate_summary.json').read_text())
    assert len(jobs) == 32 and sum(len(m['points']) for m in jobs) == 256
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=common.init, initargs=(args,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_game, m, args) for m in jobs]):
            row = future.result()
            results.append(row)
            print(json.dumps({'game_complete': row}), flush=True)
    (args.out / 'compare_summary.json').write_text(json.dumps(results, indent=1) + '\n')


if __name__ == '__main__':
    main()
