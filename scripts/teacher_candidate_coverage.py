"""Reproducible dense training collection and preregistered candidate-coverage audit.

Archives are replayed strictly, in groups of games; no evaluation state is used to fit.
Run under a host-memory guard (the experiment driver records pauses and owned RSS).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]


def init(args):
    import search_play as teacher
    teacher._init_worker(args.value, args.student, 61, True, args.static_value,
                         coverage_path=args.model if args.mode == 'audit' else None)


def game(points, args):
    import numpy as np
    import search_play as teacher
    from pvz_candidate_coverage import action_key, features
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    from pvz_frame_store import read_frames
    from pvz_rollout_policy import ranked_plants
    from pvz_student_observation import StudentHistory
    from pvz_tasks import profile_for_deck
    from pvz_value_net import observation_context
    from render_episode import _frame, load_training_task, training_task_kwargs
    from teacher_counterfactual_audit import physical
    p = points[0]
    frames = read_frames(Path(p['archive'])/'frames.npz')
    kw = training_task_kwargs(load_training_task(args.manifest, 'day_l7_z17_train'), p['seed'])
    assert [v[1] for v in frames[0]['packets']] == kw['deck']
    task = TaskSpec(level=kw['level'], seed=p['seed'], playthrough=2,
                   profile=profile_for_deck(tuple(kw['deck']), kw['first_aid_owned']),
                   **{k:v for k,v in kw['task_extra'].items() if v is not None})
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    coverage = teacher._W.pop('coverage', None)
    selected = defaultdict(list)
    for point in points:
        selected[point['decision']].append(point)
    history = StudentHistory(teacher._W['fast'].obs)
    count = 0
    try:
        obs, _ = env.reset(deck=tuple(kw['deck']), task=task)
        env.load_rollout_net(teacher._W['rollout_file'])
        for i in range(max(selected)):
            assert physical(_frame(obs)) == physical(frames[i]), (p['seed'], i)
            if i+1 in selected:
                teacher._W['context'] = observation_context(obs)
                state = {'history':history, 'action':frames[i].get('action')}
                if i:
                    state['frame'] = frames[i-1]
                policy = teacher._fast_action(obs, state)
                teacher._W['student_history'] = history
                for point in selected[i+1]:
                    tag = f"{point['category']}_{point['seed']}_{point['decision']}"
                    if (args.out/(tag+'.json')).exists():
                        continue
                    seed = p['seed']*10000+i
                    if args.mode == 'collect':
                        row = collect(env, obs, policy, args, seed, point)
                        np.savez_compressed(args.out/(tag+'.npz'), **row.pop('arrays'))
                    else:
                        row = audit(env, obs, policy, args, seed, point, frames[i+1], coverage)
                    dest = args.out/(tag+'.json')
                    tmp = dest.with_suffix('.tmp')
                    tmp.write_text(json.dumps(row)+'\n')
                    tmp.replace(dest)
                    count += 1
                    print(json.dumps({'case':tag,'mode':args.mode,'done':count}), flush=True)
            history.extra(frames[i])
            if i+1 == max(selected):
                break
            obs, *_, info = env.step(frames[i+1]['action'])
            assert (not info['ok']) == bool(frames[i+1].get('action_rejected'))
            if not info['ok']:
                obs, *_ = env.step({'type':'wait','ticks':60})
    finally:
        if coverage is not None:
            teacher._W['coverage'] = coverage
        env.close()
    return {'seed':p['seed'], 'points':count, 'checked_frames':max(selected)}


def collect(env, obs, policy, args, seed, point):
    import numpy as np
    import search_play as teacher
    from pvz_candidate_coverage import features, action_key
    from pvz_rollout_policy import ranked_plants
    from render_episode import _frame
    original = teacher._halving_choice
    captured = []
    def capture(*pos):
        captured.extend(deepcopy(pos[3]))
        return policy
    teacher._halving_choice = capture
    stats = {'searched':0,'overrides':0,'override_kinds':Counter()}
    try:
        teacher._search(env, obs, policy, args, seed, stats)
    finally:
        teacher._halving_choice = original
    assert captured
    legal = obs['legal_actions']['plants']
    actions = [{'type':'plant', **{k:a[k] for k in ('packet','row','col')}} for a in legal] + [teacher.NOOP]
    root = env.snapshot()
    try:
        static = teacher._static_frames(env, root, actions[:-1], obs.get('sun_income_rate'))
        scored = list(zip(teacher._values([f for _, f in static], static=True), [a for a, _ in static]))
        scored.append((teacher._values([_frame(obs)], static=True)[0], teacher.NOOP))
        ranked = ranked_plants(teacher._W['fast'], teacher._W['student_frame'],
                               teacher._W.get('student_previous_frame'), legal,
                               context=teacher._W['context'], history=teacher._W['student_history'])
        s, x = features(_frame(obs), teacher._W['context'], actions, scored, ranked)
        scores = []
        for a in actions:
            handles = teacher._continuations(env, obs, root,
                        [(a, (seed*1000+k)%2**32) for k in range(3)], args, stats)
            values = teacher._leaf_values(handles)
            assert all(v is not None for v in values)
            scores.append(values)
    finally:
        env.restore(root)
        env.release_snapshot(root)
    index = {action_key(a):i for i,a in enumerate(actions)}
    base = [index[action_key(a)] for a in captured]
    return {**point,'actions':actions,'base_candidates':captured,
            'continuations':len(actions)*3,
            'arrays':{'state':s,'features':x,'scores':np.asarray(scores,dtype='float32'),
                      'base':np.asarray(base,dtype='int64')}}


def audit(env, obs, policy, args, seed, point, archived, coverage):
    import search_play as teacher
    from pvz_candidate_coverage import action_key
    original_halving, original_rollout = teacher._halving_choice, teacher._rollout_policy
    rows = {}
    try:
        for name, budget, offset in [('old',0,0),('old_noise',0,7777777),
                                     ('new2',2,0),('new2_noise',2,7777777),
                                     ('new4',4,0),('new4_noise',4,7777777)]:
            captured, actual = [], [0]
            def capture(*pos):
                captured.extend(deepcopy(pos[3]))
                return original_halving(*pos)
            def rollout(*pos, **kw):
                actual[0] += 1
                return original_rollout(*pos, **kw)
            teacher._halving_choice, teacher._rollout_policy = capture, rollout
            teacher._W.pop('coverage', None)
            if budget:
                assert coverage is not None
                teacher._W['coverage'], teacher._W['coverage_budget'] = coverage, budget
            stats = {'searched':0,'overrides':0,'override_kinds':Counter()}
            chosen = teacher._search(env, obs, policy, args, seed+offset, stats)
            rounded = [[a.get('packet',-1),a.get('row',-1),a.get('col',-1),round(v,4)]
                       for a,v in stats['last_targets'] if abs(v)!=float('inf') and a['type']!='shovel']
            assert actual[0] == stats['halving_rollouts']+18*bool(stats.get('last_verification'))
            keys = [action_key(a) for a in captured]
            assert len(keys) == len(set(keys))
            if name == 'old':
                assert chosen == point['label'], ('old label',point)
                assert rounded == archived['targets'], ('old targets',point)
            if budget:
                old = rows['old']['candidates']
                assert captured[:len(old)] == old
                assert len(captured)-len(old) <= budget
                legal = {action_key(a) for a in obs['legal_actions']['plants']}
                assert all(action_key(a) in legal for a in captured[len(old):])
            rows[name] = {'chosen':chosen,'candidates':captured,'actual_continuations':actual[0],
                          'halving':stats['halving_rollouts'], 'verification':bool(stats.get('last_verification')),
                          'added':stats.get('last_coverage_actions',[])}
    finally:
        teacher._halving_choice, teacher._rollout_policy = original_halving, original_rollout
        teacher._W.pop('coverage', None)
    return {**point,'arms':rows,'old_label_and_targets_exact':True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['collect','audit'])
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model')
    p.add_argument('--workers',type=int,default=4)
    base = Path.home()/'PvZ-search/artifacts'
    p.add_argument('--student',default=str(base/'student/v29/model.pt'))
    p.add_argument('--value',default=str(base/'value_net/v3/model.pt'))
    p.add_argument('--static-value',default=str(base/'hard/value_branch1/model.pt'))
    p.add_argument('--manifest',default=str(base/'stage4/day_l7_hard.json'))
    p.add_argument('--resource-dir',default=str(Path.home()/'.cache/pvz-research-resources'))
    p.add_argument('--executable',default=str(ROOT/'build/pvz-portable'))
    args = p.parse_args()
    if not 1 <= args.workers <= 4 or (args.mode == 'audit' and not args.model):
        p.error('workers must be1..4; audit requires --model')
    args.samples=3;args.horizon=2500;args.verify_samples=9;args.per_card=1
    args.student_candidates=2;args.shovel_candidates=0;args.min_gap=300;args.halving=True;args.margin=.02
    args.out.mkdir(parents=True,exist_ok=True)
    groups = defaultdict(list)
    for point in json.loads(args.selection.read_text())['points']:
        if not (args.out/f"{point['category']}_{point['seed']}_{point['decision']}.json").exists():
            groups[point['seed']].append(point)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(args,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(game,points,args) for points in groups.values()]
        for job in as_completed(jobs):
            print(json.dumps({'game_complete':job.result()}),flush=True)


if __name__ == '__main__':
    main()
