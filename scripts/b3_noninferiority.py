"""task42: B3 (student global top-3 candidates) non-inferiority on the current setup.

Comparison arm only. The formal collection path (curriculum_queue.py -> search_play.py) is
unchanged; this script drives ``search_play._search`` directly and sets the private
``_W["candidate_mode"] = "student-topk"`` only for the B3 arm.

Per family (day L7, night L14, pool L24, original conveyor L10), 64 evaluation games
(seeds 30000..30063), 8 eligible search roots per game sampled with
``Random(20261023 + seed)``. A and B3 share the decision seed ``D = seed*10000 +
decision - 1``; the heavy teacher sees 64 common futures with seed ``D + 33333333`` and
uses 18 verification futures per override pair (task14 definition). When A and B3 choose
the same action the label cancels exactly (delta 0) and no heavy reference is run.

Usage:
  python scripts/b3_noninferiority.py --out artifacts/task42/smoke --games 4 --workers 3
  python scripts/b3_noninferiority.py --out artifacts/task42/full --games 64 --workers 3
"""
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
import random
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path('/home/newbiexvwu/PvZ-search')
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from day_orig_tasks import DAY_ORIG_MANIFEST

SEED_BASE = 30000
SAMPLE_SEED = 20261023
HEAVY_OFFSET = 33333333
HEAVY_FUTURES = 64
HEAVY_VERIFY = 18
LIGHT_SAMPLES = 3
LIGHT_VERIFY = 9
MIN_GAP = 300
HORIZON = 2500
PER_CARD = 1
STUDENT_CANDIDATES_A = 2
STUDENT_TOPK_B3 = 3
MARGIN = .03

FAMILIES = (
    {'name': 'day_l7', 'manifest': MAIN / 'artifacts/stage4/day_panel.json',
     'task': 'day_l7_eval_d7'},
    {'name': 'night_l14', 'manifest': MAIN / 'artifacts/stage4/night_panel.json',
     'task': 'night_l14_eval_n10'},
    {'name': 'pool_l24', 'manifest': MAIN / 'artifacts/stage4/pool_panel.json',
     'task': 'pool_l24_eval_p9k'},
    {'name': 'day_l10_orig', 'manifest': DAY_ORIG_MANIFEST,
     'task': 'day_l10_orig_eval'},
)
FAMILY_BY_NAME = {f['name']: f for f in FAMILIES}


def key(a):
    return ('plant', a['packet'], a['row'], a['col']) if a['type'] == 'plant' else (a['type'],)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correctness_difference(a, b, heavy=None):
    """Equal choices cancel for every common reference, without knowing its label."""
    if key(a) == key(b):
        return 0
    if heavy is None:
        raise ValueError('a disagreement requires the common heavy teacher')
    return int(key(b) == key(heavy)) - int(key(a) == key(heavy))


def init(args):
    import search_play as t
    keep, paths = t._export_values(args.value, args.static_value)
    t._W["value_keep"] = keep
    t._init_worker(paths[0], args.student, 61, paths[1])


def open_game(seed, family, args):
    from pvz_tasks import profile_for_deck
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from render_episode import load_training_task, training_task_kwargs
    import search_play as t
    kw = training_task_kwargs(load_training_task(family['manifest'], family['task']), seed)
    task = TaskSpec(level=kw['level'], seed=kw['seed'], playthrough=2,
                    profile=profile_for_deck(tuple(kw['deck']), kw.get('first_aid_owned', False)),
                    **{k: v for k, v in kw['task_extra'].items() if v is not None})
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    try:
        obs, _ = env.reset(deck=tuple(kw['deck']), task=task)
        env.load_rollout_net(t._W['rollout_file'])
        env.load_value_net(t._W['value_files'][0])
        env.load_value_net(t._W['value_files'][1], static=True)
    except BaseException:
        env.close()
        raise
    return env, obs


def generate(family, seed, args):
    import search_play as t
    from pvz_value_net import observation_context
    from pvz_frame_store import write_frames
    from render_episode import _frame
    dest = args.out / 'trajectories' / family['name'] / f'seed_{seed}'
    meta_path = dest / 'meta.json'
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    env, obs = open_game(seed, family, args)
    try:
        t._W['context'] = observation_context(obs)
        frames = [_frame(obs)]
        state = {}
        eligible = []
        last = -10 ** 9
        for i in range(3000):
            action = t._fast_action(obs, state)
            legal = (obs.get('legal_actions') or {}).get('plants') or []
            if legal and (action['type'] == 'plant' or obs['tick'] - last >= MIN_GAP):
                eligible.append({'decision': i + 1, 'tick': obs['tick'], 'wave': obs['wave']})
                last = obs['tick']
            obs, _r, done, _t, info = env.step(action)
            rejected = not info.get('ok')
            if rejected:
                obs, _r, done, _t, info = env.step({'type': 'wait', 'ticks': 60})
            frame = _frame(obs)
            frame['action'] = action
            frame['action_rejected'] = rejected
            frames.append(frame)
            if obs.get('terminal') or done:
                break
        count = min(8, len(eligible))
        chosen = sorted(random.Random(SAMPLE_SEED + seed).sample(eligible, count),
                        key=lambda p: p['decision'])
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / 'frames.jsonl'
        write_frames(path, frames)
        meta = {'family': family['name'], 'task': family['task'], 'seed': seed,
                'archive': str(path.resolve()), 'frames': len(frames),
                'eligible_points': len(eligible), 'points': chosen,
                'terminal': bool(obs.get('terminal')),
                'archive_sha256': digest(path)}
        meta_path.write_text(json.dumps(meta, indent=1) + '\n')
        return meta
    finally:
        env.close()


def normal(env, obs, policy, args, decision_seed, name):
    """One light teacher arm; the only difference is how its candidates are generated."""
    import search_play as t
    old_mode, old_k = t._W.get('candidate_mode'), t._W.get('student_topk')
    t._W['candidate_mode'] = 'static' if name == 'A' else 'student-topk'
    if name == 'B3':
        t._W['student_topk'] = STUDENT_TOPK_B3
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


def pool_actions(scored, arms, legal, seed):
    """Symmetric audit pool: both choices, top2/card, and one random remaining cell/card."""
    actions = []
    seen = set()

    def add(a):
        k = key(a)
        if k not in seen:
            actions.append(a)
            seen.add(k)

    for arm in arms.values():
        for action in arm['candidates']:
            add(action)
    counts = Counter()
    for _score, action in sorted(scored, key=lambda item: -item[0]):
        if counts[action['packet']] < 2:
            add(action)
        counts[action['packet']] += 1
    rng = random.Random(seed)
    for packet in sorted({a['packet'] for a in legal}):
        rest = [a for a in legal if a['packet'] == packet and key(a) not in seen]
        if rest:
            add(rng.choice(rest))
    return sorted(actions, key=key)


def heavy_reference(env, obs, policy, args, decision_seed, arms):
    import search_play as t
    root = env.snapshot()
    try:
        plants = [{'type': 'plant', **{k: a[k] for k in ('packet', 'row', 'col')}}
                  for a in obs['legal_actions']['plants']]
        scores = t._static_scores(env, root, plants, obs.get('sun_income_rate'))
        assert len(scores) == len(plants)
        actions = pool_actions(scores, arms, plants, SAMPLE_SEED + decision_seed)
        merged = {key(a): a for a in actions}
        for arm in arms.values():
            for action in arm['candidates']:
                merged.setdefault(key(action), action)
        actions = [merged[k] for k in sorted(merged)]
        truth_seed = decision_seed + HEAVY_OFFSET
        values, totals = [], []
        for action in actions:
            leaves = t._continuations(env, obs, root,
                                      [(action, (truth_seed * 1000 + k) % 2 ** 32)
                                       for k in range(HEAVY_FUTURES)], args, {})
            v = t._leaf_values(leaves)
            assert len(v) == HEAVY_FUTURES and all(
                x is not None and math.isfinite(x) and 0 <= x <= 1 for x in v)
            values.append(v)
            terminal = [x for leaf, x in zip(leaves, v) if t._score_terminal_or_value(leaf) is not None]
            other = [x for leaf, x in zip(leaves, v) if t._score_terminal_or_value(leaf) is None]
            totals.append(sum(terminal + other) / HEAVY_FUTURES)
        anchor = key(policy if policy['type'] == 'plant' else t.NOOP)
        anchor_index = next(i for i, a in enumerate(actions) if key(a) == anchor)
        best = max(range(len(actions)), key=lambda i: totals[i])
        heavy_args = SimpleNamespace(**vars(args))
        heavy_args.samples, heavy_args.verify_samples = HEAVY_FUTURES, HEAVY_VERIFY
        stats = {'overrides': 0, 'override_kinds': Counter()}
        label = t._verified_choice(env, obs, root, actions, totals, [HEAVY_FUTURES] * len(actions),
                                   best, anchor_index, policy, heavy_args, truth_seed, stats)
        return {'label': label, 'actions': actions, 'truth_scores': values,
                'truth_continuations': HEAVY_FUTURES * len(actions),
                'verification_continuations': 2 * HEAVY_VERIFY * bool(stats.get('last_verification')),
                'verification': stats.get('last_verification')}
    finally:
        env.restore(root)
        env.release_snapshot(root)


def evaluate(env, obs, policy, args, seed, point, family):
    decision_seed = seed * 10000 + point['decision'] - 1
    arms = {}
    order = ('A', 'B3') if (seed + point['decision']) % 2 == 0 else ('B3', 'A')
    for name in order:
        arms[name] = measured_arm(env, obs, policy, args, decision_seed, name)
    a, b = arms['A']['chosen'], arms['B3']['chosen']
    same = key(a) == key(b)
    heavy = None if same else heavy_reference(env, obs, policy, args, decision_seed, arms)
    delta = correctness_difference(a, b, heavy['label'] if heavy else None)
    return {'family': family['name'], 'seed': seed, **point, 'policy': policy,
            'arms': arms, 'measurement_order': order,
            'actions': heavy['actions'] if heavy else [],
            'heavy': heavy, 'judge_required': not same, 'delta': delta}


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
    meta = generate(family, seed, args)
    return compare(family, meta, args)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--families', default=','.join(f['name'] for f in FAMILIES))
    p.add_argument('--games', type=int, default=64)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--student', default=str(MAIN / 'artifacts/student/dw4i_s1/model.pt'))
    p.add_argument('--value', default=str(MAIN / 'artifacts/value_net/v3/model.pt'))
    p.add_argument('--static-value', default=str(MAIN / 'artifacts/hard/value_branch1/model.pt'))
    # Frozen copy of the in-service qc build (the Oct 18 main build lacks rollout_version >= 3,
    # which the current v2 student needs). Recorded in the task42 pre-registration.
    p.add_argument('--executable', default=str(ROOT / 'artifacts/task42/pvz-portable'))
    p.add_argument('--resource-dir', default=str(Path.home() / '.cache/pvz-research-resources'))
    args = p.parse_args()
    if not 1 <= args.workers <= 3:
        p.error('workers must be 1..3')
    if not 1 <= args.games <= 64:
        p.error('games must be 1..64 (the evaluation segment holds 1024 seeds from 30000)')
    names = [n for n in args.families.split(',') if n]
    unknown = [n for n in names if n not in FAMILY_BY_NAME]
    if unknown:
        p.error(f'unknown families: {unknown}')
    families = [FAMILY_BY_NAME[n] for n in names]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'points').mkdir(exist_ok=True)
    args.samples, args.horizon, args.verify_samples = LIGHT_SAMPLES, HORIZON, LIGHT_VERIFY
    args.per_card, args.student_candidates, args.min_gap = PER_CARD, STUDENT_CANDIDATES_A, MIN_GAP
    args.halving = True
    jobs = [(family, SEED_BASE + i) for family in families for i in range(args.games)]
    plan = {'families': names, 'games': args.games, 'seeds': [SEED_BASE, SEED_BASE + args.games - 1],
            'sample_seed': SAMPLE_SEED, 'heavy_offset': HEAVY_OFFSET,
            'heavy_futures': HEAVY_FUTURES, 'heavy_verify': HEAVY_VERIFY,
            'light': {'samples': LIGHT_SAMPLES, 'verify_samples': LIGHT_VERIFY,
                      'horizon': HORIZON, 'per_card': PER_CARD, 'min_gap': MIN_GAP,
                      'halving': True, 'student_candidates_A': STUDENT_CANDIDATES_A,
                      'student_topk_B3': STUDENT_TOPK_B3},
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
