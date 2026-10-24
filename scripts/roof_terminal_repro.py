"""Independent same-future label replay after the full frozen roof panel completes."""
from __future__ import annotations
import gzip
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import terminal_candidate_audit as d


def compare(original,repeated,a_trace,b_trace):
    fields=('identity','outcome','v4_short_value','short_leaf','teacher_leaf','terminal_state_sha256','checks')
    differences=[field for field in fields if original[field]!=repeated[field]]
    with gzip.open(a_trace,'rb') as stream:a=stream.read()
    with gzip.open(b_trace,'rb') as stream:b=stream.read()
    if a!=b:differences.append('full_teacher_action_chain')
    if differences:raise ValueError('independent repeat differs: '+','.join(differences))
    return dict(fields_equal=list(fields),teacher_action_chain_equal=True)


def run():
    actual=d.OUT;plan=d.check_plan();selection=d.load(actual/'selection-active.json')
    if d.own_engine_pids():raise RuntimeError('existing owned engines; wait for screen to finish')
    # Validate the entire panel before calling any repeated result a final audit.
    import terminal_candidate_report as report
    report.report()
    cases=[]
    for level in plan['spec']['levels']:
        eligible=[j for j in selection['jobs'] if j['cell']['level']==level and not j['cell'].get('archive')]
        first=min(eligible,key=lambda j:(j['cell']['seed'],j['point']['decision']))
        folder=actual/'episodes'/first['cell']['id']/f"root_{first['point']['decision']}"
        root=d.load(folder/'root.json')
        index=0 if level%2 else (0 if root['student_action']['type']=='wait' else root['candidates'].index(root['student_action']))
        cases.append(dict(job=first,index=index,reason='first seed/earliest selected root; odd levels wait, even levels student'))
    job=selection['jobs'][0]
    folder=actual/'episodes'/job['cell']['id']/f"root_{job['point']['decision']}"
    root=d.load(folder/'root.json')
    shovel=next((i for i,a in enumerate(root['candidates']) if a['type']=='shovel'),None)
    if shovel is not None:cases.append(dict(job=job,index=shovel,reason='first selected root first legal shovel'))
    run_dir=actual/'runs'/f'{int(time.time())}-repro';run_dir.mkdir(parents=True)
    d.init_worker(plan,str(run_dir),mp.get_context('fork').Event())
    results=[]
    try:
        for number,case in enumerate(cases):
            if d.spent_cpu()+600>=plan['spec']['cpu_stop_seconds']:raise RuntimeError('CPU reserve reached during independent repeat')
            job=case['job'];index=case['index'];cell=job['cell'];point=job['point']
            original_folder=actual/'episodes'/cell['id']/f"root_{point['decision']}"
            replica=actual/'repro'/f'case_{number}'
            target=replica/'episodes'/cell['id'];target.mkdir(parents=True,exist_ok=True)
            trajectory=target/'trajectory.json'
            if not trajectory.exists():trajectory.symlink_to(actual/'episodes'/cell['id']/'trajectory.json')
            d.OUT=replica
            try:d.labels_job(dict(cell=cell,point=point,stage='screen',samples=1,candidate_indices=[index]))
            finally:d.OUT=actual
            repeated_folder=target/f"root_{point['decision']}"
            name=f'screen_{index}_0'
            original=d.load(original_folder/f'{name}.json');repeated=d.load(repeated_folder/f'{name}.json')
            checks=compare(original,repeated,original_folder/f'{name}.jsonl.gz',repeated_folder/f'{name}.jsonl.gz')
            results.append(dict(cell=cell,point=point,candidate=index,reason=case['reason'],
                outcome=original['outcome'],checks=checks,original_sha256=d.sha(original_folder/f'{name}.json'),
                repeated_sha256=d.sha(repeated_folder/f'{name}.json')))
            print('REPRODUCED',cell['id'],point['decision'],index,flush=True)
    finally:
        d.OUT=actual;d.teacher._close_worker_env();d.progress(force=True)
    value=dict(status='complete',cases=results,cpu_seconds_after=d.spent_cpu(),
        comparison='exact root, history, bank context, public leaves, v4 values, terminal and full decompressed action chain; elapsed CPU/wall and gzip timestamps excluded')
    d.save(ROOT/'MEMORY/docs/task46-roof-terminal-repro.json',value)
    d.save(run_dir/'complete.json',dict(cases=len(results)))
    return value


if __name__=='__main__':run()
