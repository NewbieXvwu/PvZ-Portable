"""Compare frozen/default/sparse native continuations, full observations and private state."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--candidate',default=str(ROOT/'build-sparse/pvz-portable'))
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(1)
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_frame_store import read_frames
    from pvz_rollout_policy import load_student,export_student
    from pvz_student_observation import StudentHistory
    from render_episode import _frame,load_training_task,training_task_kwargs
    from scripted_baseline import profile_for_deck
    from teacher_counterfactual_audit import alternatives,physical
    artifacts=Path.home()/'PvZ-search/artifacts'
    student=load_student(artifacts/'student/v29/model.pt');net=args.out/'student.bin';export_student(student,net)
    task_def=load_training_task(artifacts/'stage4/day_l7_hard.json','day_l7_z17_train')
    rows=json.loads(args.selection.read_text())['points']
    points=sum([[r for r in rows if r['category']==c][:10] for c in ('A','B')],[])
    arms=[(str(ROOT/'build/pvz-portable'),61),(args.candidate,61),(args.candidate,317)]
    envs=[];verified=portable=0;records=[]
    def stable(data):return json.dumps(data,sort_keys=True,separators=(',',':'))
    try:
        for binary,_ in arms:
            env=EventWaitEnv(Path.home()/'.cache/pvz-research-resources',executable=binary,headless=True)
            env.load_rollout_net(net);envs.append(env)
        for number,point in enumerate(points):
            frames=read_frames(Path(point['archive'])/'frames.npz');kw=training_task_kwargs(task_def,point['seed'])
            task=TaskSpec(level=kw['level'],seed=kw['seed'],playthrough=2,
                         profile=profile_for_deck(tuple(kw['deck']),kw['first_aid_owned']),
                         **{k:v for k,v in kw['task_extra'].items() if v is not None})
            observations=[e.reset(deck=tuple(kw['deck']),task=task)[0] for e in envs]
            for index in range(point['decision']):
                for obs in observations:assert physical(_frame(obs))==physical(frames[index]),(point,index)
                if index==point['decision']-1:break
                for j,env in enumerate(envs):
                    observations[j],*_,info=env.step(frames[index+1]['action'])
                    assert (not info['ok'])==bool(frames[index+1].get('action_rejected'))
                    if not info['ok']:observations[j],*_=env.step({'type':'wait','ticks':60})
            history=StudentHistory(student.obs)
            for frame in frames[:point['decision']]:history.extra(frame)
            choices,_=alternatives(observations[0],point,frames[point['decision']])
            choices=[choices[i] for i in dict.fromkeys([0,len(choices)//2,len(choices)-1])]
            roots=[env.snapshot() for env in envs]
            try:
                for action in choices:
                    for k in range(8):
                        seed=(point['seed']*10000+point['decision']*17+k)%2**32
                        outputs=[]
                        for j,(env,(_,mask)) in enumerate(zip(envs,arms)):
                            leaf=env.rollout(action,observations[j]['tick']+2500,sim_skip=mask,
                                             snapshot_id=roots[j],seed=seed,student_history=deepcopy(history))
                            private=env.privileged_state();state_hash=None
                            if not leaf.get('terminal'):
                                tip=env.snapshot()
                                branch=env.branch_snapshot(tip,[{'type':'wait','ticks':0}])[0]
                                assert branch['ok'] and 'state_hash' in branch
                                state_hash=branch['state_hash'];env.release_snapshot(branch['snapshot_id']);env.release_snapshot(tip)
                            outputs.append({'leaf':stable(leaf),'private':stable(private),'portable_hash':state_hash})
                        if not outputs[0]==outputs[1]==outputs[2]:
                            failure={'point':point,'action':action,'future':seed,'outputs':outputs}
                            (args.out/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
                            raise AssertionError(f'sparse/default/frozen divergence at {point},future{seed}')
                        verified+=1;portable+=outputs[0]['portable_hash'] is not None
                        records.append({'seed':point['seed'],'decision':point['decision'],'action':action,
                                        'future':seed,'leaf_sha256':hashlib.sha256(outputs[0]['leaf'].encode()).hexdigest(),
                                        'private_sha256':hashlib.sha256(outputs[0]['private'].encode()).hexdigest(),
                                        'portable_hash':outputs[0]['portable_hash']})
            finally:
                for env,root in zip(envs,roots):env.restore(root);env.release_snapshot(root)
            print(json.dumps({'points':number+1,'triple_matched_rollouts':verified,'portable_hashes':portable}),flush=True)
        result={'points':len(points),'triple_matched_rollouts':verified,'portable_hashes':portable,
                'arms':arms,'exact':True,'records':records}
        (args.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        for env in envs:env.close()


if __name__=='__main__':main()
