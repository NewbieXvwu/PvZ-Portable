"""Fresh isolated archive replays of fixed task53 branches, after the experiment finishes."""
from __future__ import annotations
import argparse
from copy import deepcopy
import gzip
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as f
ORIGINAL_REPLAY=f.replay


def checked_replay(job):
    """An observer only: verify every beta=0 archived action against today's frozen student."""
    from pvz_frame_store import frame_path,read_frames
    frames=read_frames(frame_path(Path(job['archive'])))
    original=f.TEACHER._fast_action;index=0
    def predict(obs,state):
        nonlocal index
        action=original(obs,state)
        if job.get('student_executed') and action!=frames[index+1]['action']:
            raise ValueError(f'archived beta=0 student differs at decision {index+1}')
        index+=1
        return action
    f.TEACHER._fast_action=predict
    try:result=ORIGINAL_REPLAY(job)
    finally:f.TEACHER._fast_action=original
    f.save(Path(f.PLAN['out'])/'roots'/job['id']/'prefix-actor.json',dict(
        archive=job['archive'],decision_count=index,all_student_actions_exact=True,
        all_public_prefix_frames_exact=True,source=job['source']))
    return result


def semantic_compare(original,repeated,a,b):
    fields=('identity','outcome','v4_short','short_leaf','post_action_frame','teacher_leaf',
            'terminal_engine_sha256','checks')
    changed=[name for name in fields if original[name]!=repeated[name]]
    with gzip.open(a,'rb') as stream:x=stream.read()
    with gzip.open(b,'rb') as stream:y=stream.read()
    if x!=y:changed.append('uncompressed_teacher_trace')
    if changed:raise ValueError('replay differs: '+','.join(changed))
    return dict(fields_equal=list(fields),full_teacher_trace_equal=True)


def live_games():
    processes={};owners=set()
    scripts={str(ROOT/'scripts'/name).encode() for name in
             ('critical_forks.py','task53_pipeline.py','task53_evaluate.py','task53_repro.py')}
    for p in Path('/proc').glob('[0-9]*'):
        try:
            stat=(p/'stat').read_text().rsplit(')',1)[1].split();args=(p/'cmdline').read_bytes().split(b'\0')
            if stat[0] in ('Z','X'):continue
            processes[int(p.name)]=(int(stat[1]),args)
            if len(args)>1 and args[1] in scripts:owners.add(int(p.name))
        except OSError:continue
    def below(pid):
        seen=set()
        while pid in processes and pid not in seen:
            if pid in owners:return True
            seen.add(pid);pid=processes[pid][0]
        return False
    return [pid for pid,(_,args) in processes.items() if args and args[0].endswith(b'/pvz-portable') and below(pid)]


def select_cases(plan):
    out=Path(plan['out']);cases=[]
    for terrain in ('day','night','pool'):
        job=next(j for j in plan['jobs'] if j['terrain']==terrain)
        cases.append(dict(job=job,index=0,reason='first frozen root per terrain; wait'))
    for kind in ('plant','shovel'):
        for job in plan['jobs']:
            root=f.load(out/'roots'/job['id']/'root.json')
            candidates=[i for i,a in enumerate(root['candidates']) if a['type']==kind]
            if candidates:
                cases.append(dict(job=job,index=candidates[0],reason='first legal '+kind+' action in frozen order'))
                break
    return cases


def run(plan):
    actual=Path(plan['out'])
    if not (actual/'pipeline-complete.json').exists() or live_games():
        raise RuntimeError('wait for existing collector/confirmation/evaluation games')
    cases=select_cases(plan);replica=actual/'repro'/str(int(time.time()))
    if replica.exists():raise RuntimeError('fresh replay directory required')
    repeat=deepcopy(plan);repeat['out']=str(replica)
    grouped={}
    for case in cases:
        job=grouped.setdefault(case['job']['id'],dict(case['job'],indices=[],samples=1,stage='screen'))
        if case['index'] not in job['indices']:job['indices'].append(case['index'])
    pool_prefixes={}
    for job in plan['jobs']:
        if job['terrain']=='pool' and (job['archive'] not in pool_prefixes or job['decision']>pool_prefixes[job['archive']]['decision']):
            pool_prefixes[job['archive']]=job
    for job in pool_prefixes.values():
        if job['id'] not in grouped:
            grouped[job['id']]=dict(job,indices=[],samples=1,stage='screen',inspect_only=True)
    repeat['jobs']=[dict(job,indices=sorted(job['indices'])) for job in grouped.values()]
    repeat['cpu_budget_seconds']=plan['cpu_budget_seconds']-f.spent_cpu(actual)-250
    if repeat['cpu_budget_seconds']<=0:raise RuntimeError('no reserved CPU for final reproducibility')
    # Charge the complete subprocess tree to the real experiment ledger. Nested replica
    # receipts are not added a second time by spent_cpu(actual).
    ledger=actual/'runs'/f'{int(time.time())}-repro';ledger.mkdir(parents=True)
    before=f.cpu()
    try:
        f.replay=checked_replay
        f.run(repeat,repeat['jobs'],'repro')
    finally:
        f.replay=ORIGINAL_REPLAY
        f.save(ledger/'complete.json',dict(cpu_seconds=f.cpu()-before))
    proof=[]
    for case in cases:
        name=f"screen_{case['index']}_0.json";source=actual/'roots'/case['job']['id']/name
        reproduced=replica/'roots'/case['job']['id']/name
        checks=semantic_compare(f.load(source),f.load(reproduced),source.with_suffix('.jsonl.gz'),reproduced.with_suffix('.jsonl.gz'))
        proof.append(dict(root=case['job']['id'],candidate=case['index'],reason=case['reason'],checks=checks,
                          original_sha256=f.sha(source),reproduced_sha256=f.sha(reproduced)))
    value=dict(status='complete',cases=proof,case_count=len(proof),fresh_directory=str(replica),
               comparison='exact input/physics/leaf/outcome/full teacher actions; exclude CPU and gzip timestamps',
               original_evidence_read_only=True,inputs=plan['inputs'],cpu_seconds_metered_after=f.spent_cpu(actual))
    value['pool_actor_prefixes']=[f.load(replica/'roots'/job['id']/'prefix-actor.json') for job in pool_prefixes.values()]
    assert len(value['pool_actor_prefixes'])==8
    f.save(ROOT/'MEMORY/docs/task53-critical-forks-repro.json',value)
    print('REPRODUCED',len(proof),'fresh isolated cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path)
    a=p.parse_args();run(f.check_plan(a.plan))
