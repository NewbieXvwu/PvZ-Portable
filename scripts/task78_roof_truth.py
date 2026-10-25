"""Independent roof terminal panel: 24 teacher worlds, 48 roots, three common futures.

Stages: plan, trajectories, sources. The frozen source manifests then use the same
task78_replay/task78_collect channel, each in its own k0/k1/k2 label directory.
This panel is evaluation data and must never be exported to the training corpus.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from task78_ab import frozen_panel
from task78_data import MAIN,sha,row_for
from value_teacher_feasibility import write_json

OUT=ROOT/'artifacts/task78/roof_truth'


def inputs():
    paths=dict(value=MAIN/'artifacts/value_net/v5/model.pt',
               student=MAIN/'artifacts/student/dw4i_s1/model.pt',
               static=MAIN/'artifacts/hard/value_branch1/model.pt',
               executable=ROOT/'build/pvz-portable')
    return {k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()}


def code():
    paths=list((ROOT/'python').glob('pvz_*.py'))+[ROOT/'scripts'/name for name in
        ('search_play.py','render_episode.py','pvz_tasks.py','task78_roof_truth.py','task78_data.py')]
    return {str(p):sha(p) for p in sorted(paths)}


def plan():
    panel=[c for c in frozen_panel(ROOT/'artifacts/task78/ab') if c['terrain']=='roof']
    worlds=[];tasks=[]
    for cell in panel:
        task={k:v for k,v in cell.items() if k not in ('terrain','pairs','cell_id')}
        task['seeds']=cell['seeds'][:8]
        tasks.append(task)
        worlds.extend(dict(level=cell['level'],task_id=cell['task_id'],seed=s) for s in task['seeds'])
    expected_baseline=json.loads((ROOT/'MEMORY/docs/task78-v5-corpus-requirements.json').read_text())['baseline_sha256']
    frozen_inputs=inputs()
    if frozen_inputs['value']['sha256']!=expected_baseline:
        raise ValueError('roof truth continuation must use the adopted v5 baseline')
    document=dict(worlds=worlds,tasks=tasks,inputs=frozen_inputs,code=code(),roots_per_world=2,horizon=2500,
                  common_futures=3,seed_bases=[78400000,78410000,78420000],workers=2,
                  roots='midpoint quantiles of actual teacher searches, before reading terminal branch labels',
                  continuation='frozen v5 search teacher from full student leaves to terminal',
                  training_eligible=False)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'plan.json'
    manifest=OUT/'tasks.json'
    if path.exists():
        if json.loads(path.read_text())!=document or json.loads(manifest.read_text())!={'tasks':tasks}:
            raise ValueError('roof terminal panel differs from its frozen plan')
    else:
        write_json(path,document);write_json(manifest,dict(tasks=tasks))
    print(json.dumps(dict(plan=str(path),sha256=sha(path),worlds=len(worlds),roots=48,futures=3),indent=1))


def frozen():
    data=json.loads((OUT/'plan.json').read_text())
    if (data['inputs']!=inputs() or data.get('training_eligible') is not False
            or data['code']!=code() or json.loads((OUT/'tasks.json').read_text())!={'tasks':data['tasks']}):
        raise ValueError('roof terminal panel inputs changed')
    return data


def no_games():
    executable=str(ROOT/'build/pvz-portable').encode()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if path.read_bytes().split(b'\0')[0]==executable:
                raise RuntimeError('another task78 game is live; wait before opening roof panel workers')
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            continue


def trajectories():
    import fcntl
    with (ROOT/'artifacts/task78/games-controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        data=frozen()
        tasks=json.loads((OUT/'tasks.json').read_text())['tasks']
        for task in tasks:
            no_games();frozen()
            output=OUT/'trajectories'/task['task_id'];output.mkdir(parents=True,exist_ok=True)
            args=[sys.executable,str(ROOT/'scripts/search_play.py'),
                  '--manifest',str(OUT/'tasks.json'),'--task-id',task['task_id'],
                  '--seed-list',','.join(map(str,task['seeds'])),'--workers','2',
                  '--out',str(output),'--frames',str(OUT/'frames'),
                  '--frames-label',task['task_id'],
                  '--value',data['inputs']['value']['path'],
                  '--rollout-policy',data['inputs']['student']['path'],
                  '--static-value',data['inputs']['static']['path'],
                  '--executable',data['inputs']['executable']['path'],
                  '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources',
                  '--verify-samples','9','--student-candidates','2','--samples','3','--horizon','2500',
                  '--min-gap','300','--per-card','1','--halving','--rollout-sim-skip','61','--dagger-beta','1']
            receipt=dict(plan_sha256=sha(OUT/'plan.json'),tasks_sha256=sha(OUT/'tasks.json'),command=args)
            target=output/'run_identity.json'
            if target.exists() and json.loads(target.read_text())!=receipt:
                raise ValueError('roof trajectories belong to another runtime')
            write_json(target,receipt)
            print(task['task_id'],flush=True)
            with (output/'run.log').open('a') as log:
                subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,
                    env=os.environ|dict(PVZ_NO_POPUP='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))


def sources():
    data=frozen();archives=[]
    for task in json.loads((OUT/'tasks.json').read_text())['tasks']:
        output=OUT/'trajectories'/task['task_id']
        records=[json.loads(line) for line in (output/'games.jsonl').read_text().splitlines()]
        if (len(records)!=8 or {r['seed'] for r in records}!=set(task['seeds']) or
                any(r.get('terminal') is not True or type(r.get('won')) is not bool for r in records)):
            raise ValueError('missing, duplicate or nonterminal roof trajectories')
        for seed in task['seeds']:
            archive=OUT/'frames'/f'seed_{seed}'/(task['task_id']+'_fast_search')
            row=row_for(f"roof_l{task['level']}",archive)
            if len(row['roots'])!=2:
                raise ValueError('roof trajectory has fewer than two actual search roots')
            archives.append(row)
    if len(archives)!=24:
        raise ValueError('exactly 24 independent roof worlds required')
    for k,seed_base in enumerate(data['seed_bases']):
        directory=OUT/f'k{k}';directory.mkdir(exist_ok=True)
        document=dict(archives=archives,seed_base=seed_base,training_eligible=False,
                      common_future_index=k,plan_sha256=sha(OUT/'plan.json'))
        path=directory/'sources.json'
        if path.exists() and json.loads(path.read_text())!=document:
            raise ValueError('roof source manifest changed after root selection')
        write_json(path,document)
    print(json.dumps(dict(worlds=24,roots=48,label_jobs=72,training_eligible=False),indent=1))


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('plan','trajectories','sources'):
        raise SystemExit('stage must be plan, trajectories, or sources')
    globals()[sys.argv[1]]()
