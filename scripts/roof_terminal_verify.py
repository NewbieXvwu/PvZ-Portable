"""Independent whole-panel check of the frozen roof terminal labels.

Reconstructs every label identity from the stored root, the frozen plan and the
preregistered common-future seeds, then revalidates the label, its teacher action
chain hash and the full decision chain. Read-only: opens no game process.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import terminal_candidate_audit as d


def expected_identity(cell,point,index,action,k,plan_sha):
    return dict(cell=cell['id'],root=point,index=index,action=action,
        future=d.future_seed(cell['id'],point['decision'],k,'screen'),
        search_seed_base=d.future_seed(cell['id'],point['decision'],k,'screen-teacher-search'),
        stage='screen',k=k,inputs_sha256=plan_sha)


def verify():
    plan=d.check_plan();d.PLAN=plan  # future_seed hashes the preregistered namespace
    selection=d.load(d.OUT/'selection-active.json')
    if selection['jobs']!=d.load(d.OUT/'selection-active.json')['jobs']:raise ValueError('unstable selection')
    failures=[];roots=[];labels=0;wins=0;futures=defaultdict(set)
    for job in selection['jobs']:
        cell,point=job['cell'],job['point']
        folder=d.OUT/'episodes'/cell['id']/f"root_{point['decision']}"
        root=d.load(folder/'root.json')
        complete=d.load(folder/'screen-complete.json')
        initial=d.ready(d.trajectory_first_frame(cell))
        if root['cell']!=cell or root['point']!=point:failures.append(dict(cell=cell['id'],root=point['decision'],error='root header differs'))
        if root['inputs_sha256']!=plan['sha256']:failures.append(dict(cell=cell['id'],root=point['decision'],error='root inputs differ'))
        if complete['labels']!=len(root['candidates'])*job['samples']:failures.append(dict(cell=cell['id'],root=point['decision'],error='complete label count differs'))
        if complete['root_restored'] is not True:failures.append(dict(cell=cell['id'],root=point['decision'],error='root not restored'))
        got=0;root_wins=0;mower_max=0
        for index,action in enumerate(root['candidates']):
            for k in range(job['samples']):
                path=folder/f'screen_{index}_{k}.json';trace=folder/f'screen_{index}_{k}.jsonl.gz'
                if not path.exists():
                    failures.append(dict(cell=cell['id'],root=point['decision'],index=index,k=k,error='missing label'));continue
                row=d.load(path)
                try:d.validate_label(row,expected_identity(cell,point,index,action,k,plan['sha256']),trace)
                except ValueError as error:
                    failures.append(dict(cell=cell['id'],root=point['decision'],index=index,k=k,error=str(error)));continue
                outcome=row['outcome']
                if outcome['tick']<point['tick']:failures.append(dict(cell=cell['id'],root=point['decision'],index=index,k=k,error='terminal before root'))
                if not 0<=outcome['mowers_used']<=initial:failures.append(dict(cell=cell['id'],root=point['decision'],index=index,k=k,error='mowers outside reset defenses'))
                futures[(cell['id'],point['decision'],k)].add(row['identity']['future'])
                got+=1;labels+=1;mower_max=max(mower_max,outcome['mowers_used'])
                if outcome['won']:wins+=1;root_wins+=1
        roots.append(dict(cell=cell['id'],root=point['decision'],tick=point['tick'],labels=got,
            expected=len(root['candidates'])*job['samples'],wins=root_wins,initial_ready_mowers=initial,mowers_used_max=mower_max))
    for key,values in futures.items():
        if len(values)!=1:failures.append(dict(cell=key[0],root=key[1],k=key[2],error='common future not shared across candidates'))
    value=dict(status='complete' if not failures else 'failed',roots=len(roots),labels=labels,wins=wins,
        sampling_sha256=d.sha(d.OUT/'selection-active.json'),plan_sha256=d.sha(d.OUT/'plan.json'),
        checks='identity rebuilt from stored root/frozen plan/preregistered seeds; label, teacher action chain hash, full decision chain, terminal tick, mowers vs actual reset defenses, shared common future',
        failures=failures,by_root=roots)
    d.save(ROOT/'MEMORY/docs/task46-roof-terminal-verification.json',value)
    return value


if __name__=='__main__':
    value=verify();print(json.dumps({k:v for k,v in value.items() if k!='by_root'},indent=2))
    if value['failures']:raise SystemExit(1)
