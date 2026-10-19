"""Reproduce original desktop31 teacher labels and expose halving budgets for audit cases."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]


def init(args):
    import search_play as teacher
    teacher._init_worker(args.value,args.student,61,True,args.static_value)


def trace(point,args):
    import search_play as teacher
    from teacher_counterfactual_audit import physical
    from pvz_frame_store import read_frames
    from pvz_event_env import EventWaitEnv
    from pvz_env import TaskSpec
    from pvz_student_observation import StudentHistory
    from pvz_value_net import observation_context
    from render_episode import _frame,load_training_task,training_task_kwargs
    from pvz_tasks import profile_for_deck
    frames=read_frames(Path(point['archive'])/'frames.npz');d=point['decision']
    kw=training_task_kwargs(load_training_task(args.manifest,'day_l7_z17_train'),point['seed'])
    task=TaskSpec(level=kw['level'],seed=kw['seed'],playthrough=2,
                 profile=profile_for_deck(tuple(kw['deck']),kw['first_aid_owned']),
                 **{k:v for k,v in kw['task_extra'].items() if v is not None})
    env=EventWaitEnv(args.resource_dir,executable=args.executable,headless=True)
    original=teacher._verified_choice
    captured={}
    def record(*pos,**kwargs):
        captured.update(candidates=deepcopy(pos[3]),before_totals=list(pos[4]),counts=list(pos[5]),
                        best_index=pos[6],policy_index=pos[7],policy_action=deepcopy(pos[8]))
        return original(*pos,**kwargs)
    try:
        obs,_=env.reset(deck=tuple(kw['deck']),task=task)
        for i in range(d):
            assert physical(_frame(obs))==physical(frames[i]),(point,i)
            if i==d-1:break
            obs,*_,info=env.step(frames[i+1]['action'])
            assert (not info['ok'])==bool(frames[i+1].get('action_rejected'))
            if not info['ok']:obs,*_=env.step({'type':'wait','ticks':60})
        env.load_rollout_net(teacher._W['rollout_file'])
        teacher._W['context']=observation_context(obs)
        history=StudentHistory(teacher._W['fast'].obs)
        for frame in frames[:d-1]:history.extra(frame)
        state={'history':history,'action':frames[d-1].get('action')}
        if d>=2:state['frame']=frames[d-2]
        policy=teacher._fast_action(obs,state)
        teacher._W['student_history']=state['history']
        stats={'searched':0,'overrides':0,'override_kinds':Counter()}
        teacher._verified_choice=record
        chosen=teacher._search(env,obs,policy,args,point['seed']*10000+d-1,stats)
        reproduced=[[a.get('packet',-1),a.get('row',-1),a.get('col',-1),round(v,4)]
                    for a,v in stats['last_targets'] if abs(v)!=float('inf') and a['type']!='shovel']
        return {**point,**captured,'chosen':chosen,'label_reproduced':chosen==point['label'],
                'targets_reproduced':reproduced==frames[d].get('targets'),
                'reproduced_targets':reproduced,'archived_targets':frames[d].get('targets'),
                'verification':stats.get('last_verification'),
                'halving_rollouts':stats.get('halving_rollouts'),
                'archived_verification':frames[d].get('verification')}
    finally:
        teacher._verified_choice=original
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    base=Path.home()/'PvZ-search/artifacts'
    p.add_argument('--student',default=str(base/'student/v29/model.pt'))
    p.add_argument('--value',default=str(base/'value_net/v3/model.pt'))
    p.add_argument('--static-value',default=str(base/'hard/value_branch1/model.pt'))
    p.add_argument('--manifest',default=str(base/'stage4/day_l7_hard.json'))
    p.add_argument('--resource-dir',default=str(Path.home()/'.cache/pvz-research-resources'))
    p.add_argument('--executable',default=str(ROOT/'build/pvz-portable'))
    args=p.parse_args()
    if not 1<=args.workers<=4:p.error('workers must be1..4')
    args.samples=3;args.horizon=2500;args.verify_samples=9;args.per_card=1
    args.student_candidates=2;args.shovel_candidates=0;args.min_gap=300;args.halving=True;args.margin=.02
    points=json.loads(args.selection.read_text())['points'];args.out.mkdir(parents=True,exist_ok=True)
    points=[v for v in points if not (args.out/f"{v['category']}_{v['seed']}_{v['decision']}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(args,),
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs={pool.submit(trace,v,args):v for v in points}
        for job in as_completed(jobs):
            try:row=job.result()
            except BaseException:
                for future in jobs:future.cancel()
                raise
            v=jobs[job];dest=args.out/f"{v['category']}_{v['seed']}_{v['decision']}.json"
            tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(row)+'\n');tmp.replace(dest)
            print(json.dumps({'case':dest.name,'label':row['label_reproduced'],'targets':row['targets_reproduced']}),flush=True)


if __name__=='__main__':main()
