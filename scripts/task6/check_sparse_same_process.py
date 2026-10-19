"""Isolate the sparse bit within one engine/root, with repeated full-mode hash controls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(1)
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_frame_store import read_frames
    from pvz_rollout_policy import load_student,export_student
    from pvz_student_observation import StudentHistory
    from render_episode import _frame,load_training_task,training_task_kwargs
    from pvz_tasks import profile_for_deck
    from teacher_counterfactual_audit import alternatives,physical
    artifacts=Path.home()/'PvZ-search/artifacts';student=load_student(artifacts/'student/v29/model.pt')
    net=args.out/'student.bin';export_student(student,net)
    task_def=load_training_task(artifacts/'stage4/day_l7_hard.json','day_l7_z17_train')
    rows=json.loads(args.selection.read_text())['points']
    points=sum([[r for r in rows if r['category']==c][:10] for c in ('A','B')],[])
    env=EventWaitEnv(Path.home()/'.cache/pvz-research-resources',executable=ROOT/'build-sparse/pvz-portable',headless=True)
    env.load_rollout_net(net);records=[]
    def stable(d):return json.dumps(d,sort_keys=True,separators=(',',':'))
    try:
        for number,point in enumerate(points):
            frames=read_frames(Path(point['archive'])/'frames.npz');kw=training_task_kwargs(task_def,point['seed'])
            task=TaskSpec(level=kw['level'],seed=kw['seed'],playthrough=2,
                         profile=profile_for_deck(tuple(kw['deck']),kw['first_aid_owned']),
                         **{k:v for k,v in kw['task_extra'].items() if v is not None})
            obs,_=env.reset(deck=tuple(kw['deck']),task=task)
            for i in range(point['decision']):
                assert physical(_frame(obs))==physical(frames[i])
                if i==point['decision']-1:break
                obs,*_,info=env.step(frames[i+1]['action'])
                assert (not info['ok'])==bool(frames[i+1].get('action_rejected'))
                if not info['ok']:obs,*_=env.step({'type':'wait','ticks':60})
            history=StudentHistory(student.obs)
            for frame in frames[:point['decision']]:history.extra(frame)
            choices,_=alternatives(obs,point,frames[point['decision']])
            choices=[choices[i] for i in dict.fromkeys([0,len(choices)//2,len(choices)-1])]
            root=env.snapshot()
            try:
                for action in choices:
                    for k in range(8):
                        seed=(point['seed']*10000+point['decision']*17+k)%2**32
                        outputs=[]
                        for mask in [61,61,317,317,61]:
                            leaf=env.rollout(action,obs['tick']+2500,sim_skip=mask,snapshot_id=root,
                                             seed=seed,student_history=deepcopy(history))
                            private=env.privileged_state();state_hash=None
                            if not leaf.get('terminal'):
                                tip=env.snapshot();branch=env.branch_snapshot(tip,[{'type':'wait','ticks':0}])[0]
                                assert branch['ok'] and branch['observation']['tick']==leaf['tick']
                                state_hash=branch['state_hash'];env.release_snapshot(branch['snapshot_id']);env.release_snapshot(tip)
                            outputs.append({'mask':mask,'leaf':stable(leaf),'private':stable(private),'hash':state_hash})
                        public_private=all(o['leaf']==outputs[0]['leaf'] and o['private']==outputs[0]['private'] for o in outputs)
                        hashes=[o['hash'] for o in outputs]
                        record={'seed':point['seed'],'decision':point['decision'],'future':seed,'action':action,
                                'hashes':hashes,'public_private_equal':public_private,
                                'full_mode_repeat_equal':hashes[0]==hashes[1]==hashes[4],
                                'all_hashes_equal':len(set(hashes))==1}
                        records.append(record)
                        if not public_private:
                            (args.out/'failure.json').write_text(json.dumps({'point':point,'outputs':outputs},indent=2)+'\n')
                            raise AssertionError('same-process full leaf/private divergence')
            finally:
                env.restore(root);env.release_snapshot(root)
            print(json.dumps({'points':number+1,'comparisons':len(records),
                              'hash_mismatches':sum(not r['all_hashes_equal'] for r in records),
                              'repeat_full_mismatches':sum(not r['full_mode_repeat_equal'] for r in records)}),flush=True)
        (args.out/'result.json').write_text(json.dumps({'records':records,'points':len(points),
                                                     'public_private_equal':True,
                                                     'all_hashes_equal':all(r['all_hashes_equal'] for r in records)},indent=2)+'\n')
    finally:
        env.close()


if __name__=='__main__':main()
