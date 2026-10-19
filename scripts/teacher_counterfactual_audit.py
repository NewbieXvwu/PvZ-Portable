"""Replay archived teacher labels and score legal alternatives on paired futures.

This is an audit only: it never changes a teacher or the archived executed actions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]


def key(action):
    return json.dumps(action, sort_keys=True)


def physical(frame):
    return json.loads(json.dumps({k: v for k, v in frame.items()
                                 if k not in ('action', 'action_rejected', 'label', 'targets', 'verification')}))


def category(state, label):
    from render_episode import _col_of
    if label.get('type') != 'plant':
        return None
    kind = state['packets'][label['packet']][1]
    if kind == 1 and state['wave'] <= 3 and label['col'] >= 4:
        return 'B', None
    if kind == 4:
        xs = [z[2] for z in state['zombies'] if z[0] == label['row'] and z[5]]
        gap = _col_of(min(xs)) - label['col'] if xs else None
        if gap is not None and gap <= 0:
            return 'A', gap
    return None


def select(args):
    from pvz_frame_store import read_frames
    paths = sorted(Path(args.archives).glob(f'*/{args.label}/frames.npz'))
    pools = defaultdict(list)
    rng = random.Random(args.seed)
    totals = Counter()
    for number, path in enumerate(paths):
        frames = read_frames(path)
        per_game = defaultdict(list)
        for i in range(1, len(frames)):
            label = frames[i].get('label') or {}
            found = category(frames[i-1], label)
            if found:
                c, gap = found
                per_game[c].append(dict(category=c, archive=str(path.parent), decision=i,
                                        seed=int(path.parents[1].name.removeprefix('seed_')),
                                        gap=gap, wave=frames[i-1]['wave'], tick=frames[i-1]['tick'],
                                        label=label))
                totals[c] += 1
        for c, points in per_game.items():
            pools[c].append(rng.choice(points))
        if number % 50 == 0:
            print(json.dumps({'scanned_games': number+1, 'eligible': dict(totals)}), flush=True)
    picked = []
    for c in ('A', 'B'):
        rng.shuffle(pools[c])
        if len(pools[c]) < args.points:
            raise RuntimeError(f'{c}: only {len(pools[c])} eligible games; need {args.points}')
        picked.extend(pools[c][:args.points])
    doc = {'sampling': 'one uniformly chosen eligible point per eligible game; 300 games/class without replacement',
           'selection_seed': args.seed, 'source_label': args.label,
           'source_games': len(paths), 'eligible_points': dict(totals),
           'eligible_games': {c:len(pools[c]) for c in pools}, 'points': picked}
    args.out.write_text(json.dumps(doc, indent=2)+'\n')
    print(json.dumps({k:v for k,v in doc.items() if k!='points'}), flush=True)


def alternatives(obs, point, archived):
    label = point['label']
    choices = [label]
    for a in obs['legal_actions']['plants']:
        if a['packet'] != label['packet']:
            continue
        if point['category'] == 'A' and a['row'] != label['row']:
            continue
        if point['category'] == 'B' and a['col'] > 3:
            continue
        choices.append({'type':'plant', 'packet':a['packet'], 'row':a['row'], 'col':a['col']})
    choices.append({'type':'wait','ticks':60})
    ranked = sorted(archived.get('targets') or [], key=lambda t:-t[3])
    runner_up = None
    if point['category'] == 'A' and len(ranked)>1:
        p,r,c,_ = ranked[1]
        runner_up = ({'type':'plant','packet':p,'row':r,'col':c} if p>=0
                     else {'type':'wait','ticks':60})
        choices.append(runner_up)
    unique = {key(a):a for a in choices}
    legal = {(a['packet'],a['row'],a['col']) for a in obs['legal_actions']['plants']}
    for a in unique.values():
        if a['type']=='plant' and (a['packet'],a['row'],a['col']) not in legal:
            raise RuntimeError(f'archived label or runner-up is illegal: {point}, {a}')
    return list(unique.values()), runner_up


def run_point(point, args, replay_only=False):
    from pvz_frame_store import read_frames
    from render_episode import _frame, load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_student_observation import StudentHistory
    import search_play as teacher
    from pvz_value_net import observation_context

    archive = Path(point['archive'])
    frames = read_frames(archive/'frames.npz')
    kw = training_task_kwargs(load_training_task(args.manifest,'day_l7_z17_train'), point['seed'])
    # The archive's packet table must agree before using the pinned manifest for missing reset metadata.
    assert [p[1] for p in frames[0]['packets']] == kw['deck']
    task = TaskSpec(level=kw['level'],seed=kw['seed'],playthrough=2,
                    profile=profile_for_deck(tuple(kw['deck']),kw['first_aid_owned']),
                    **{k:v for k,v in kw['task_extra'].items() if v is not None})
    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    try:
        obs,_ = env.reset(deck=tuple(kw['deck']),task=task)
        checked = 0
        for i in range(point['decision']):
            actual = physical(_frame(obs)); expected = physical(frames[i])
            if actual != expected:
                fields = [k for k in actual.keys()|expected.keys() if actual.get(k)!=expected.get(k)]
                args.out.mkdir(parents=True,exist_ok=True)
                (args.out/f"mismatch_{point['category']}_{point['seed']}.failure.json").write_text(
                    json.dumps({'point':point,'first_mismatch_frame':i,
                                'actual':actual,'expected':expected},indent=2)+'\n')
                raise RuntimeError(f'replay mismatch {archive} frame {i}: {fields}')
            checked += 1
            if i == point['decision']-1:
                break
            if getattr(args,'replay_search_restores',False) and frames[i+1].get('targets'):
                saved = env.snapshot()
                obs = env.restore(saved)
                env.release_snapshot(saved)
            obs,*_,info = env.step(frames[i+1]['action'])
            assert (not info.get('ok')) == bool(frames[i+1].get('action_rejected'))
            if not info.get('ok'):
                obs,*_ = env.step({'type':'wait','ticks':60})
        result = {**point,'replayed_frames':checked,'replay_exact':True}
        if replay_only:
            return result
        env.load_rollout_net(teacher._W['rollout_file'])
        teacher._W['context'] = observation_context(obs)
        history = StudentHistory(teacher._W['fast'].obs)
        for frame in frames[:point['decision']]:
            history.extra(frame)
        teacher._W['student_history'] = history
        actions,runner_up = alternatives(obs,point,frames[point['decision']])
        if 'control_actions' in point:
            actions=point['control_actions'];runner_up=None
            assert len(actions)==2 and actions[0]==point['label']
            legal={(a['packet'],a['row'],a['col']) for a in obs['legal_actions']['plants']}
            assert all(a['type']=='wait' or (a['packet'],a['row'],a['col']) in legal for a in actions)
        root = env.snapshot()
        # Separate audit futures from the teacher's original 0..11 seeds.
        future_rng = random.Random(f"task9:{args.seed}:{point['seed']}:{point['decision']}")
        futures = [future_rng.randrange(2**32) for _ in range(args.futures)]
        stats = {}; scores = []
        for action in actions:
            handles = teacher._continuations(env,obs,root,[(action,s) for s in futures],args,stats)
            values = teacher._leaf_values(handles)
            assert all(v is not None for v in values)
            scores.append(values)
        env.restore(root)
        assert physical(_frame(env.restore(root))) == physical(frames[point['decision']-1])
        env.release_snapshot(root)
        result.update(actions=actions,scores=scores,future_seeds=futures,runner_up=runner_up,
                      archived_targets=frames[point['decision']].get('targets'),
                      archived_verification=frames[point['decision']].get('verification'),
                      archived_executed=frames[point['decision']].get('action'),stats=stats)
        return result
    finally:
        env.close()


def worker_init(args):
    import search_play as teacher
    teacher._init_worker(args.value,args.student,61,True,args.static_value)


def score(args):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing
    doc = json.loads(args.selection.read_text())
    points = doc['points']
    if args.mode == 'replay':
        points = sum([[p for p in points if p['category']==c][:10] for c in ('A','B')], [])
    args.out.mkdir(parents=True,exist_ok=True)
    pending = [p for p in points if not (args.out/f"{p['category']}_{p['seed']}_{p['decision']}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init if args.mode!='replay' else None,
                             initargs=(args,) if args.mode!='replay' else ()) as pool:
        jobs = {pool.submit(run_point,p,args,args.mode=='replay'):p for p in pending}
        for job in as_completed(jobs):
            try:
                result = job.result()
            except BaseException:
                for other in jobs:
                    other.cancel()
                raise
            p=jobs[job]
            dest = args.out/f"{p['category']}_{p['seed']}_{p['decision']}.json"
            temp = dest.with_suffix('.tmp');temp.write_text(json.dumps(result)+'\n');temp.replace(dest)
            print(json.dumps({'done':dest.name,'actions':len(result.get('actions',[]))}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['select','replay','score'])
    p.add_argument('--archives',default=str(Path.home()/'PvZ-search/artifacts/student/frames_v2'))
    p.add_argument('--label',default='z17dag32b_fast_search')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--selection',type=Path)
    p.add_argument('--points',type=int,default=300)
    p.add_argument('--seed',type=int,default=20261019)
    p.add_argument('--futures',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--replay-search-restores',action='store_true',
                   help='diagnose physical effects of the original search snapshot restoration')
    p.add_argument('--horizon',type=int,default=2500)
    p.add_argument('--student',default=str(Path.home()/'PvZ-search/artifacts/student/v30/model.pt'))
    p.add_argument('--value',default=str(Path.home()/'PvZ-search/artifacts/value_net/v3/model.pt'))
    p.add_argument('--static-value',default=str(Path.home()/'PvZ-search/artifacts/hard/value_branch1/model.pt'))
    p.add_argument('--manifest',default=str(Path.home()/'PvZ-search/artifacts/stage4/day_l7_hard.json'))
    p.add_argument('--resource-dir',default=str(Path.home()/'.cache/pvz-research-resources'))
    p.add_argument('--executable',default=str(ROOT/'build/pvz-portable'))
    args = p.parse_args()
    if not 1 <= args.workers <= 4:
        p.error('workers must be 1..4')
    if args.mode=='select':
        select(args)
    else:
        score(args)


if __name__=='__main__':
    main()
