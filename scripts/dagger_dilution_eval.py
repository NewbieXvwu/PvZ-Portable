"""Evaluate every frozen task15 checkpoint on the same six full held-out tiers."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from train_rollout_policy_dilution import foreign_trainers


def game_processes():
    result=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if args and Path(args[0].decode(errors='replace')).name=='pvz-portable':result.append(int(p.name))
    return result


def available_kib():
    return int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plan=json.loads(args.plan.read_text());contract=plan['eval_contract']
    # runnerlib is the same already-validated private memory guard used throughout
    # tasks8..15; keep logs/raw games ignored rather than publishing them in git.
    sys.path.insert(0,str(ROOT/'artifacts/cuda/task8'));import runnerlib as r
    r.OUT=args.out;r.OUT.mkdir(parents=True,exist_ok=True)
    completed=json.loads((args.plan.parent/'training_complete.json').read_text())
    by_name={v['name']:v for v in completed};assert set(by_name)=={j['name'] for j in plan['jobs']}
    snapshot=args.plan.parent/'frozen_eval_manifest.json'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==contract['manifest_sha256']
    assert hashlib.sha256(Path(contract['executable']).read_bytes()).hexdigest()==contract['executable_sha256']
    # A common order is fixed before reading any evaluation output.
    jobs=sorted(plan['jobs'],key=lambda j:(j['seed'],j['arm']))
    for job in jobs:
        model=Path(job['out'])/'model.pt';expected=by_name[job['name']]['model_sha256']
        assert hashlib.sha256(model.read_bytes()).hexdigest()==expected
        for task in contract['task_ids']:
            tag='eval_'+job['name']+'_'+task.split('_')[-2]
            dest=args.out/job['name']/task
            if (dest/'summary.json').exists():
                rows=[json.loads(v) for v in (dest/'games.jsonl').read_text().splitlines()]
                if len(rows)==contract['seeds_each']:continue
            while True:
                busy=foreign_trainers();games=game_processes();avail=available_kib()
                if not busy and len(games)<=4 and avail>=9*1024*1024:break
                print(json.dumps({'eval_admission_wait':job['name'],'task':task,'foreign_trainers':busy,
                                  'foreign_games':len(games),'available_kib':avail}),flush=True)
                time.sleep(10)
            for path,digest in plan['source_files'].items():
                assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
            cmd=[r.PYTHON,'scripts/search_play.py','--rollout-policy',str(model),
                 '--value',str(Path.home()/'PvZ-search/artifacts/value_net/v3/model.pt'),
                 '--manifest',str(snapshot),'--task-id',task,'--seeds',str(contract['seeds_each']),
                 '--arms','fast','--workers',str(contract['workers']),'--executable',contract['executable'],
                 '--resource-dir',str(Path.home()/'.cache/pvz-research-resources'),'--out',str(dest)]
            result=r.run(cmd,tag,extra={'CUDA_VISIBLE_DEVICES':'','PVZ_EXECUTABLE':contract['executable']})
            assert result['returncode']==0,(tag,result)
            rows=[json.loads(v) for v in (dest/'games.jsonl').read_text().splitlines()]
            assert len(rows)==contract['seeds_each'] and len({v['seed'] for v in rows})==len(rows)
            assert hashlib.sha256(model.read_bytes()).hexdigest()==expected
            print(json.dumps({'evaluation_complete':job['name'],'task':task,'games':len(rows)}),flush=True)
    (args.out/'evaluation_complete.json').write_text(json.dumps({'models':len(jobs),'tiers':len(contract['task_ids']),
        'games':len(jobs)*len(contract['task_ids'])*contract['seeds_each']},indent=2)+'\n')


if __name__=='__main__':main()
