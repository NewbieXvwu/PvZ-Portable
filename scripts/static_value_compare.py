"""Same-state, common-future comparison of frozen static value networks.

Diagnostics only: trajectories execute the student, never either search teacher.
Run under the experiment's host-memory guard; at most four simulator workers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]


def key(a):
    return ('plant',a['packet'],a['row'],a['col']) if a['type']=='plant' else (a['type'],)


def init(args):
    import search_play as t
    keep, paths = t._export_values(args.value, args.desktop)
    mac_keep, mac_paths = t._export_values(args.value, args.mac)
    t._init_worker(paths[0],args.student,61,True,paths[1])
    t._W.update(value_keep=keep, mac_keep=mac_keep,
                static_models={'desktop':paths[1], 'mac':mac_paths[1]})


def open_game(seed,args):
    from pvz_tasks import profile_for_deck
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from render_episode import load_training_task,training_task_kwargs
    import search_play as t
    kw=training_task_kwargs(load_training_task(args.manifest,'day_l7_z17_eval'),seed)
    task=TaskSpec(level=kw['level'],seed=seed,playthrough=2,
                  profile=profile_for_deck(tuple(kw['deck']),kw['first_aid_owned']),
                  **{k:v for k,v in kw['task_extra'].items() if v is not None})
    env=EventWaitEnv(args.resource_dir,executable=args.executable,headless=True)
    try:
        obs,_=env.reset(deck=tuple(kw['deck']),task=task)
        env.load_rollout_net(t._W['rollout_file'])
        env.load_value_net(t._W['value_files'][0])
        env.load_value_net(t._W['static_models']['desktop'], static=True)
        t._W['env'] = env
    except BaseException:
        env.close()
        raise
    return env,obs


def generate(seed,args):
    import search_play as t
    from pvz_value_net import observation_context
    from pvz_frame_store import write_frames
    from render_episode import _frame
    dest=args.out/'trajectories'/f'seed_{seed}'
    if (dest/'meta.json').exists():
        return json.loads((dest/'meta.json').read_text())
    env,obs=open_game(seed,args)
    try:
        t._W['context']=observation_context(obs)
        frames=[_frame(obs)];state={};eligible=[];last=-10**9
        for i in range(3000):
            action=t._fast_action(obs,state)
            legal=obs['legal_actions']['plants']
            if legal and (action['type']=='plant' or obs['tick']-last>=300):
                eligible.append({'decision':i+1,'tick':obs['tick'],'wave':obs['wave']})
                last=obs['tick']
            obs,*_,info=env.step(action)
            rejected=not info['ok']
            if rejected:obs,*_=env.step({'type':'wait','ticks':60})
            frame=_frame(obs);frame['action']=action;frame['action_rejected']=rejected
            frames.append(frame)
            if obs['terminal']:break
        chosen=sorted(random.Random(20261020+seed).sample(eligible,min(8,len(eligible))),key=lambda p:p['decision'])
        dest.mkdir(parents=True,exist_ok=True)
        path=dest/'frames.jsonl';write_frames(path,frames)
        meta={'seed':seed,'archive':str(path.resolve()),'frames':len(frames),
              'eligible_points':len(eligible),'points':chosen,'terminal':obs['terminal'],
              'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (dest/'meta.json').write_text(json.dumps(meta,indent=1)+'\n')
        return meta
    finally:env.close()


def normal(env,obs,policy,args,D,name):
    import search_play as t
    env.load_value_net(t._W['static_models'][name], static=True)
    original=t._halving_choice;captured=[]
    def record(*pos):
        captured.extend(deepcopy(pos[3]))
        return original(*pos)
    stats={'searched':0,'overrides':0,'override_kinds':Counter()}
    try:
        t._halving_choice=record
        chosen=t._search(env,obs,policy,args,D,stats)
    finally:t._halving_choice=original
    assert captured and stats['searched']==1
    return {'chosen':chosen,'candidates':captured,
            'continuations':stats['halving_rollouts']+18*bool(stats.get('last_verification'))}


def pool_actions(scored_by_model,normal_arms,legal,seed):
    """Symmetric audit pool: both choices, top2/card/model, and a random cell/card."""
    actions=[];seen=set()
    def add(a):
        k=key(a)
        if k not in seen:actions.append(a);seen.add(k)
    for name in ('mac','desktop'):
        for a in normal_arms[name]['candidates']:add(a)
        counts=Counter()
        for score,a in sorted(scored_by_model[name],key=lambda p:-p[0]):
            if counts[a['packet']]<2:add(a)
            counts[a['packet']]+=1
    rng=random.Random(seed)
    for packet in sorted({a['packet'] for a in legal}):
        rest=[a for a in legal if a['packet']==packet and key(a) not in seen]
        if rest:add(rng.choice(rest))
    # The common reference must not resolve equal rollout means in favour of the
    # model whose candidates were inserted first. This order is model-independent.
    return sorted(actions,key=key)


def evaluate(env,obs,policy,args,seed,point):
    import numpy as np
    import search_play as t
    from render_episode import _frame
    D=seed*10000+point['decision']-1
    arms={}
    for name in ('mac','desktop'):
        arms[name]=normal(env,obs,policy,args,D,name)
        arms[name+'_noise']=normal(env,obs,policy,args,D+7777777,name)
        assert arms[name]['candidates']==arms[name+'_noise']['candidates']
    root=env.snapshot()
    try:
        plants=[{'type':'plant',**{k:a[k] for k in ('packet','row','col')}} for a in obs['legal_actions']['plants']]
        static_scores={};lookup={}
        for name in ('mac','desktop'):
            env.load_value_net(t._W['static_models'][name], static=True)
            static_scores[name]=t._static_scores(env,root,plants,obs.get('sun_income_rate'))
            assert len(static_scores[name])==len(plants)
            lookup[name]={key(a):v for v,a in static_scores[name]}
            lookup[name][key(t.NOOP)]=t._static_scores(env,root,[dict(type='wait',ticks=0)],obs.get('sun_income_rate'))[0][0]
        actions=pool_actions(static_scores,arms,plants,20261020+D)
        keys={key(a) for a in actions}
        assert len(actions)==len(keys) and all(key(a) in keys for v in arms.values() for a in v['candidates'])
        truthD=D+33333333;scores=[];totals=[]
        for a in actions:
            leaves=t._continuations(env,obs,root,[(a,(truthD*1000+k)%2**32) for k in range(64)],args,{})
            values=t._leaf_values(leaves)
            assert all(v is not None for v in values) and np.isfinite(values).all()
            scores.append(values)
            # Keep future-index order in truth data; use the original terminal-first
            # accumulation for the heavy teacher's selection and anchored verification.
            ordered=[v for leaf,v in zip(leaves,values) if t._score_terminal_or_value(leaf) is not None]
            ordered += [v for leaf,v in zip(leaves,values) if t._score_terminal_or_value(leaf) is None]
            totals.append(sum(ordered)/64)
        anchor=key(policy if policy['type']=='plant' else t.NOOP)
        policy_index=next(i for i,a in enumerate(actions) if key(a)==anchor)
        heavy_args=SimpleNamespace(**vars(args));heavy_args.samples=64;heavy_args.verify_samples=18
        best=max(range(len(actions)),key=lambda i:totals[i])
        stats={'overrides':0,'override_kinds':Counter()}
        heavy=t._verified_choice(env,obs,root,actions,totals,[64]*len(actions),best,policy_index,
                                  policy,heavy_args,truthD,stats)
        return {'seed':seed,**point,'policy':policy,'actions':actions,
                'static_scores':{name:[lookup[name][key(a)] for a in actions] for name in lookup},
                'truth_scores':scores,'normal_arms':arms,'heavy_label':heavy,
                'heavy_verification':stats.get('last_verification'),
                'truth_continuations':64*len(actions),'heavy_extra_continuations':36*bool(stats.get('last_verification'))}
    finally:
        env.restore(root);env.release_snapshot(root)


def compare(meta,args,evaluator=None):
    import search_play as t
    from pvz_frame_store import read_frames
    from pvz_student_observation import StudentHistory
    from pvz_value_net import observation_context
    from teacher_counterfactual_audit import physical
    from render_episode import _frame
    path=Path(meta['archive']);assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['archive_sha256']
    frames=read_frames(path);selected={p['decision']:p for p in meta['points']}
    env,obs=open_game(meta['seed'],args)
    history=StudentHistory(t._W['fast'].obs);completed=0
    try:
        for i in range(max(selected)):
            assert physical(_frame(obs))==physical(frames[i]),(meta['seed'],i)
            if i+1 in selected:
                dest=args.out/'points'/f"{meta['seed']}_{i+1}.json"
                if not dest.exists():
                    t._W['context']=observation_context(obs)
                    state={'history':history,'action':frames[i].get('action')}
                    if i:state['frame']=frames[i-1]
                    policy=t._fast_action(obs,state);t._W['student_history']=history
                    assert policy==frames[i+1]['action'],('policy replay',meta['seed'],i)
                    row=(evaluator or evaluate)(env,obs,policy,args,meta['seed'],selected[i+1])
                    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(row)+'\n');tmp.replace(dest)
                    completed+=1
                    print(json.dumps({'point':dest.stem,'done':completed,'pool':len(row['actions'])}),flush=True)
            history.extra(frames[i])
            if i+1==max(selected):break
            obs,*_,info=env.step(frames[i+1]['action'])
            assert (not info['ok'])==bool(frames[i+1].get('action_rejected'))
            if not info['ok']:obs,*_=env.step({'type':'wait','ticks':60})
    finally:env.close()
    return {'seed':meta['seed'],'completed':completed,'strict_replay_frames':max(selected)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['generate','compare'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    base=Path.home()/'PvZ-search/artifacts'
    p.add_argument('--student',default=str(base/'student/v30/model.pt'))
    p.add_argument('--value',default=str(base/'value_net/v3/model.pt'))
    p.add_argument('--desktop',default=str(base/'hard/value_branch1/model.pt'))
    p.add_argument('--mac',default=str(Path.home()/'pvz-logs/mac_branch1/model.pt'))
    p.add_argument('--manifest',default=str(base/'stage4/day_l7_hard.json'))
    p.add_argument('--resource-dir',default=str(Path.home()/'.cache/pvz-research-resources'))
    p.add_argument('--executable',default=str(ROOT/'build-task12/pvz-portable'))
    args=p.parse_args()
    if not 1<=args.workers<=4:p.error('workers must be1..4')
    args.samples=3;args.horizon=2500;args.verify_samples=9;args.per_card=1
    args.student_candidates=2;args.shovel_candidates=0;args.min_gap=300;args.halving=True;args.margin=.02
    args.out.mkdir(parents=True,exist_ok=True);(args.out/'points').mkdir(exist_ok=True)
    if args.mode=='generate':jobs=list(range(30000,30032));fn=generate
    else:
        jobs=[json.loads(path.read_text()) for path in sorted((args.out/'trajectories').glob('seed_*/meta.json'))]
        assert len(jobs)==32 and all(m['points'] for m in jobs)
        fn=compare
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(args,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(fn,v,args) for v in jobs]
        results=[]
        for future in as_completed(futures):
            row=future.result();results.append(row)
            print(json.dumps({'game_complete':row}),flush=True)
    (args.out/(args.mode+'_summary.json')).write_text(json.dumps(results,indent=1)+'\n')


if __name__=='__main__':main()
