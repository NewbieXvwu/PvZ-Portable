"""Repeat a fixed small subset of appendix E labels in isolated directories, never overwrite evidence."""
from __future__ import annotations
import argparse
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import fog_terminal_diagnostic as d


def select_cases(selection,root_loader):
    cases=[]
    for level in (31,32,33,34):
        for mode in ('human','xray'):
            candidates=[j for j in selection['jobs'] if j['cell']['level']==level and j['cell']['observation']==mode]
            seed=min(j['cell']['seed'] for j in candidates)
            candidates=[j for j in candidates if j['cell']['seed']==seed]
            chosen=min(candidates,key=lambda j:j['point']['decision']) if mode=='human' else max(candidates,key=lambda j:j['point']['decision'])
            root=root_loader(chosen)
            student=root['student_action']
            index=0 if mode=='xray' or student['type']=='wait' else root['candidates'].index(student)
            cases.append(dict(job=chosen,candidate=index,reason='first seed; human initial/student, xray late/wait'))
    # Probe the first root where a wall-nut really is legal, without choosing by terminal outcome.
    for job in selection['jobs']:
        root=root_loader(job);packets={p[0]:p[1] for p in root['root_frame']['packets']}
        nuts=[i for i,a in enumerate(root['candidates']) if a['type']=='plant' and packets[a['packet']]==3]
        if nuts:
            cases.append(dict(job=job,candidate=nuts[0],reason='first legal wall-nut root/cell'))
            break
    return cases


def semantic_compare(original,repeated,original_trace,repeated_trace):
    fields=('identity','outcome','v4_short_value','short_leaf','teacher_leaf','terminal_state_sha256','checks')
    mismatched=[name for name in fields if original[name]!=repeated[name]]
    with gzip.open(original_trace,'rb') as f:a=f.read()
    with gzip.open(repeated_trace,'rb') as f:b=f.read()
    if a!=b:mismatched.append('uncompressed_teacher_trace')
    if mismatched:raise ValueError('repeated label differs: '+','.join(mismatched))
    return dict(fields_equal=list(fields),teacher_trace_equal=True)


def run():
    actual=d.OUT
    if d.task45_engine_pids():raise RuntimeError('wait for existing task45 games; do not use a fifth slot')
    if not (actual/'pipeline-complete.json').exists():raise RuntimeError('finish screening and independent confirmation first')
    plan=d.check_plan();selection=d.load(actual/'selection-active.json')
    if d.spent_cpu()+480>=71400:raise RuntimeError('insufficient reserved CPU for reproducibility check')
    def folder(job):return actual/'episodes'/job['cell']['id']/f"root_{job['point']['decision']}"
    cases=select_cases(selection,lambda job:d.load(folder(job)/'root.json'))
    run_dir=actual/'runs'/f'{int(time.time())}-repro-check';run_dir.mkdir(parents=True)
    d.init_worker(plan,str(run_dir),mp.get_context('fork').Event())
    result=[]
    try:
        for n,case in enumerate(cases):
            if d.spent_cpu()+480>=71400:raise RuntimeError('CPU reserve reached')
            job=case['job'];index=case['candidate'];original_dir=folder(job)
            replica=actual/'repro-check'/f'case_{n}'
            target=replica/'episodes'/job['cell']['id'];target.mkdir(parents=True,exist_ok=True)
            trajectory=actual/'episodes'/job['cell']['id']/'trajectory.json'
            link=target/'trajectory.json'
            if not link.exists():link.symlink_to(trajectory)
            d.OUT=replica
            try:d.labels_job(dict(**job,candidate_indices=[index],samples=1))
            finally:d.OUT=actual
            repeated_dir=target/f"root_{job['point']['decision']}"
            original=d.load(original_dir/f'screen_{index}_0.json')
            repeated=d.load(repeated_dir/f'screen_{index}_0.json')
            checks=semantic_compare(original,repeated,original_dir/f'screen_{index}_0.jsonl.gz',repeated_dir/f'screen_{index}_0.jsonl.gz')
            result.append(dict(cell=job['cell'],point=job['point'],candidate=index,reason=case['reason'],
                action=original['identity']['action'],outcome=original['outcome'],checks=checks,
                original_label_sha256=d.sha(original_dir/f'screen_{index}_0.json'),
                repeated_label_sha256=d.sha(repeated_dir/f'screen_{index}_0.json')))
            print('REPRODUCED',job['cell']['id'],job['point']['decision'],index,flush=True)
    finally:
        d.OUT=actual;d.teacher._close_worker_env();d.progress(force=True)
    value=dict(status='complete',cases=result,case_count=len(result),
        selection='first seed per level/mode; human initial/student action, xray late/wait; first legal wall-nut as extra',
        comparison='exact physics, outcome, public leaves, v4 values and full uncompressed teacher action chain; wall/CPU time and gzip timestamps excluded',
        inputs_sha256=plan['sha256'],cpu_seconds_metered_after=d.spent_cpu())
    d.save(ROOT/'MEMORY/docs/task45-fog-terminal-repro.json',value)
    d.save(run_dir/'repro-complete.json',dict(case_count=len(result)))
    print('REPRO CHECK COMPLETE',len(result),flush=True)

if __name__=='__main__':run()
