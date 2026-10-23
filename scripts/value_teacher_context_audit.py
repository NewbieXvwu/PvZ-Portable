"""Replay saved task43 actions/short leaves, verify states, and retain exact sun-income history.

Pass absolute round2 plan paths. One owned engine at a time; never re-run terminal teachers.
"""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from value_teacher_feasibility import (ROOT, cpu, digest, history_record, load_plan,
                                       process_sample, write_json)
import search_play as teacher
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_student_observation import StudentHistory
from pvz_tasks import profile_for_deck
from pvz_value_net import observation_context
from render_episode import _frame, load_training_task, training_task_kwargs


def income(env):
    return dict(start_tick=env._sun_history_start_tick,
                production=[list(x) for x in env._sun_production_history])


def json_value(value):
    # _frame keeps plants/zombies as tuples; archives encode them as JSON arrays.
    return json.loads(json.dumps(value))


def keep_income(path, record, value):
    if 'sun_income_history' in record and record['sun_income_history'] != value:
        raise ValueError(f'existing income history differs: {path}')
    record['sun_income_history'] = value
    write_json(path, record)


def audit(plan_paths):
    sample = process_sample()
    if sample['own_engines']:
        raise RuntimeError('wait for all collection engines to finish before context audit')
    results=[]
    for plan_path in plan_paths:
        plan=load_plan(plan_path); args=Namespace(**plan['teacher'])
        teacher._load_nets(args.value,args.rollout_policy,args.static_value)
        teacher._init_worker(args.value,args.rollout_policy,args.rollout_sim_skip,args.static_value)
        for seed in plan['seeds']:
            directory=Path(plan['out'])/f'seed_{seed}'
            saved=json.loads((directory/'result.json').read_text())
            actions=[json.loads(x) for x in (directory/'root_trajectory.jsonl').read_text().splitlines()]
            roots={}
            for path in directory.glob('wave_*/root.json'):
                record=json.loads(path.read_text()); roots[record['decision_seed']%10000]=(path,record)
            start=cpu(); wall=time.monotonic()
            with EventWaitEnv(args.resource_dir,executable=args.executable,headless=True) as env:
                env.load_rollout_net(teacher._W['rollout_file'])
                kw=training_task_kwargs(load_training_task(Path(args.manifest),args.task_id),seed)
                task=TaskSpec(level=kw['level'],seed=seed,playthrough=2,
                    profile=profile_for_deck(tuple(kw['deck']),kw.get('first_aid_owned',False)),
                    **{k:v for k,v in kw['task_extra'].items() if v is not None})
                obs,_=env.reset(deck=tuple(kw['deck']),task=task)
                teacher._W['context']=observation_context(obs)
                checked_roots=checked_leaves=0
                for index,action in enumerate(actions):
                    if index!=action['decision'] or obs['tick']!=action['tick'] or obs['wave']!=action['wave']:
                        raise ValueError('root action replay identity differs')
                    if index in roots:
                        path,record=roots[index]
                        actual_frame = json_value(_frame(obs))
                        frame_differences = [k for k in record['root_frame']
                                             if actual_frame.get(k) != record['root_frame'][k]]
                        actual_hash = env._debug_state_sha256()
                        if actual_hash != record['root_state_sha256'] or frame_differences:
                            raise ValueError(f'replayed root differs: frame={frame_differences}, '
                                             f'hash_equal={actual_hash == record["root_state_sha256"]}')
                        keep_income(path,record,income(env))
                        h=record['history']['history']; history=StudentHistory(h['obs'])
                        history.wave=h['wave'];history.wave_tick=h['wave_tick']
                        history.plants={tuple(x[:3]):x[3] for x in h['plants']}
                        root=env.snapshot()
                        try:
                            for leaf_path in sorted(path.parent.glob('candidate_[0-9]*.json')):
                                leaf_record=json.loads(leaf_path.read_text())
                                leaf=env.rollout(leaf_record['action'],obs['tick']+args.horizon,
                                    sim_skip=args.rollout_sim_skip,snapshot_id=root,
                                    seed=record['common_future'],student_history=history)
                                if json_value(_frame(leaf))!=leaf_record['leaf_frame'] or env._debug_state_sha256()!=leaf_record['leaf_state_sha256']:
                                    raise ValueError('replayed original native leaf differs')
                                keep_income(leaf_path,leaf_record,income(env))
                                # Validate the stored recurrent history itself, not only frame identity.
                                teacher._W['student_history']=deepcopy(history); recovered={}
                                teacher._python_rollout_reference(env,obs,leaf_record['action'],
                                    obs['tick']+args.horizon,root,record['common_future'],
                                    fast_state_out=recovered)
                                if json_value(history_record(recovered))!=leaf_record['leaf_history']:
                                    raise ValueError('stored student history differs from replay')
                                checked_leaves+=1
                        finally:
                            env.restore(root);env.release_snapshot(root)
                        checked_roots+=1
                    obs,_,_,_,info=env.step(action['chosen'])
                    if not info.get('ok'):obs,*_=env.step({'type':'wait','ticks':60})
                if json_value(_frame(obs))!=saved['final_frame']:
                    raise ValueError('complete replayed root final frame differs')
                measured=cpu(env)-start
            result=dict(seed=seed,roots=checked_roots,leaves=checked_leaves,
                        cpu_seconds=measured,wall_seconds=time.monotonic()-wall)
            results.append(result); print(json.dumps(result),flush=True)
    output=plan_paths[0].parent/'context_audit.json'
    value=dict(status='pass',seeds=results,root_states=sum(r['roots'] for r in results),
        native_leaf_states_and_student_histories=sum(r['leaves'] for r in results),
        cpu_seconds=sum(r['cpu_seconds'] for r in results),
        max_own_engines=1,collection_engines_at_start=sample['own_engines'],
        source_sha256=digest(Path(__file__)))
    write_json(output,value)


if __name__=='__main__':
    paths=[Path(x) for x in sys.argv[1:]]
    if not paths or not all(p.is_absolute() for p in paths):raise ValueError('absolute plan paths required')
    audit(paths)
