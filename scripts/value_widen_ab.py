"""Paired v3/v4 teacher evaluation; only leaf value changes, never defaults.

Runs one task at a time, at most four game workers. Independent phases permit
baseline collection while the GPU trains v4. The saved plan freezes tasks/seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/newbiexvwu/PvZ-search')


def paired(old,new):
    seeds=sorted(set(old)&set(new));gain=sum(new[s] and not old[s] for s in seeds)
    harm=sum(old[s] and not new[s] for s in seeds);n=gain+harm
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(gain,harm)+1))/2**n) if n else 1.
    return dict(pairs=len(seeds),old_wins=sum(old[s] for s in seeds),new_wins=sum(new[s] for s in seeds),
                gain=gain,harm=harm,delta_wins=gain-harm,delta_rate=(gain-harm)/len(seeds) if seeds else None,
                mcnemar_exact_two_sided=p,missing_old=sorted(set(new)-set(old)),missing_new=sorted(set(old)-set(new)))


def plan(args):
    target=args.out/'plan.json'
    if target.is_file():return json.loads(target.read_text())
    tasks=[]
    # --task MANIFEST:task_id replaces the default six-task set, so a follow-up validation can
    # freeze its own plan (e.g. the day-pressure pair) without touching the original plan.json.
    sources=([(Path(spec.split(':',1)[0]),[spec.split(':',1)[1]]) for spec in args.task] if args.task else
             [(MAIN/'artifacts/stage4/night_panel.json',['night_l14_eval_n10','night_l17_eval_n10','night_l20_eval_n10_orig']),
              (MAIN/'artifacts/stage4/pool_panel.json',['pool_l21_eval_p9k','pool_l24_eval_p9k']),
              (MAIN/'artifacts/stage4/day_l7_hard.json',['day_l7_z15_eval'])])
    for manifest,ids in sources:
        data=json.loads(manifest.read_text())
        for task_id in ids:
            t=next(t for t in data['tasks'] if t['task_id']==task_id)
            tasks.append({**t,'seeds':t['seeds'][args.seed_offset:args.seed_offset+args.games]})
    if any(len(t['seeds'])!=args.games for t in tasks):raise ValueError('not enough evaluation seeds')
    args.out.mkdir(parents=True,exist_ok=True)
    frozen=args.out/'manifest.json';frozen.write_text(json.dumps(dict(tasks=tasks),indent=1))
    result=dict(manifest=str(frozen),tasks=[t['task_id'] for t in tasks],games=args.games,workers=args.workers,
                seed_offset=args.seed_offset,student=str(args.student),static=str(args.static),
                v3=str(args.baseline),v4=str(args.v4),executable=str(args.executable),
                student_sha256=hashlib.sha256(args.student.read_bytes()).hexdigest(),
                static_sha256=hashlib.sha256(args.static.read_bytes()).hexdigest(),
                control='queue103 run(), beta=1, only leaf value checkpoint changes')
    target.write_text(json.dumps(result,indent=1));return result


def run(args,arm,p):
    for key in ('student','static'):
        if hashlib.sha256(Path(p[key]).read_bytes()).hexdigest()!=p[key+'_sha256']:
            raise ValueError(f'{key} changed since the frozen paired plan')
    for task in p['tasks']:
        out=args.out/arm/task;out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(ROOT/'scripts/search_play.py'),'--rollout-policy',p['student'],
             '--static-value',p['static'],'--value',p[arm],'--verify-samples','9','--student-candidates','2',
             '--manifest',p['manifest'],'--task-id',task,'--seeds',str(p['games']),
             '--workers',str(args.workers),'--executable',str(p['executable']),
             '--resource-dir','/home/newbiexvwu/.cache/pvz-research-resources','--dagger-beta','1','--out',str(out)]
        print(f'{arm} {task}',flush=True)
        with (out/'run.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        print((out/'summary.json').read_text(),flush=True)


def report(args,p):
    target=args.out/'paired.json'
    target.unlink(missing_ok=True) # A failed revalidation must not leave an old success report.
    frozen={t['task_id']:t for t in json.loads(Path(p['manifest']).read_text())['tasks']}
    out={}
    for task in p['tasks']:
        planned=frozen[task]['seeds'];expected=set(map(int,planned))
        if len(planned)!=p['games'] or len(expected)!=p['games']:
            raise ValueError(f'{task}: frozen plan does not contain {p["games"]} unique seeds')
        results=[];missing={}
        for arm in ('v3','v4'):
            path=args.out/arm/task/'games.jsonl'
            rows=[json.loads(s) for s in path.read_text().splitlines()] if path.is_file() else []
            seeds=[int(r['seed']) for r in rows]
            if len(seeds)!=len(set(seeds)):raise ValueError(f'{task} {arm}: duplicate game seeds')
            unexpected=set(seeds)-expected
            if unexpected:raise ValueError(f'{task} {arm}: unexpected seeds {sorted(unexpected)}')
            missing[arm]=sorted(expected-set(seeds))
            results.append({int(r['seed']):bool(r['won']) for r in rows})
        out[task]=paired(*results)
        out[task]['missing_expected_seeds']=missing
    doc=dict(plan=p,complete=all(r['pairs']==p['games'] and not any(r['missing_expected_seeds'].values()) for r in out.values()),
             results=out,note='A nonsignificant difference with 32 pairs does not establish noninferiority. No defaults are changed.')
    target.write_text(json.dumps(doc,indent=1));print(json.dumps(doc,indent=1))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['plan','v3','v4','report'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--v4',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--games',type=int,default=32)
    p.add_argument('--seed-offset',type=int,default=0)
    p.add_argument('--baseline',type=Path,default=MAIN/'artifacts/value_net/v3/model.pt',
                   help='the arm named v3; pass v4 to compare a new net against v4')
    # The arm names stay v3/v4 for compatibility with value_sparse_finish.py and the tests; the
    # student/static pair is what makes a run comparable, so a later comparison can pin its own
    # (e.g. the production r36 student instead of the v4-era dw4i_s1) without editing the file.
    p.add_argument('--student',type=Path,default=MAIN/'artifacts/student/dw4i_s1/model.pt')
    p.add_argument('--static',type=Path,default=MAIN/'artifacts/hard/value_branch1/model.pt')
    p.add_argument('--executable',type=Path,default=ROOT/'build/pvz-portable')
    p.add_argument('--task',action='append',help='MANIFEST:task_id; replaces the default six tasks')
    a=p.parse_args()
    if not 1<=a.workers<=4:p.error('--workers must be 1..4')
    q=plan(a)
    if a.phase in ('v3','v4'):run(a,a.phase,q)
    if a.phase=='report':report(a,q)
if __name__=='__main__':main()
