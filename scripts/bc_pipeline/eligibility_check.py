from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT=Path('/Users/newbiexvwu/PvZAgent')
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from pvz_env import PvZEnv,TaskSpec
from pvz_event_env import EventWaitEnv
OUT=Path('/tmp/pvz_unblock_bc_2b')
TEACHER_PATH=Path('/tmp/pvz_deck_v2/scripted_baseline_d1.py')
RESOURCE='/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN'
SEEDS=tuple(range(30000,30256))
DECK=(0,1,2,3,4,5,7)
WORKERS=8
MAX_ACTIONS=4000
_ENV=None
_CHOOSE=None
_PROFILE=None
_MODE=None


def init_worker(mode):
    global _ENV,_CHOOSE,_PROFILE,_MODE
    spec=importlib.util.spec_from_file_location('teacher_d1',TEACHER_PATH)
    teacher=importlib.util.module_from_spec(spec); spec.loader.exec_module(teacher)
    _CHOOSE=teacher.choose
    _PROFILE=teacher.profile_for_deck
    _MODE=mode
    env_cls=PvZEnv if mode=='E2' else EventWaitEnv
    _ENV=env_cls(RESOURCE,headless=True)


def run_seed(seed):
    task=TaskSpec(level=7,seed=seed,playthrough=2,profile=_PROFILE(DECK),
                  zombie_count_multiplier=1.0,wave_cap=None,preplanted=())
    obs,_=_ENV.reset(deck=list(DECK),task=task)
    actions=0; mowers=0; rejected=0; info={}
    while not obs['terminal'] and actions<MAX_ACTIONS:
        raw=_CHOOSE(obs)
        if _MODE=='E1' and raw.get('type')=='wait':
            action={'type':'wait','ticks':raw.get('ticks',60),'until':'timeout'}
        else:
            action=raw
        obs,_,done,_,info=_ENV.step(action)
        if not info.get('ok'):
            rejected+=1
            obs,_,done,_,info=_ENV.step({'type':'wait','ticks':60})
        mowers+=int((info.get('events') or {}).get('mower_triggered',0) or 0)
        actions+=1
        if done: break
    return {'seed':seed,'won':int(obs['result'])==1,'wave':obs['wave'],'tick':obs['tick'],
            'sun_left':obs['sun'],'actions':actions,'mowers':mowers,'rejected':rejected}


def run_variant(mode):
    start=time.time(); rows=[]
    print(f'START {mode} workers={WORKERS}',flush=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS,initializer=init_worker,
                                                  initargs=(mode,)) as pool:
        futures=[pool.submit(run_seed,seed) for seed in SEEDS]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            rows.append(future.result())
            if i%32==0 or i==len(futures):
                print(f'{mode} progress={i}/{len(futures)} wins={sum(r["won"] for r in rows)}',flush=True)
    rows.sort(key=lambda x:x['seed'])
    return {'wins':sum(r['won'] for r in rows),'episodes':len(rows),
            'win_rate':sum(r['won'] for r in rows)/len(rows),
            'rejected':sum(r['rejected'] for r in rows),
            'elapsed_seconds':round(time.time()-start,1),'episodes_detail':rows}


def exact_mcnemar(e2_only, variant_only):
    n=e2_only+variant_only
    if n==0: return 1.0
    tail=sum(math.comb(n,k) for k in range(min(e2_only,variant_only)+1))/(2**n)
    return min(1.0,2*tail)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results={mode:run_variant(mode) for mode in ('E2','E0','E1')}
    base={r['seed']:bool(r['won']) for r in results['E2']['episodes_detail']}
    paired={}
    for mode in ('E0','E1'):
        alt={r['seed']:bool(r['won']) for r in results[mode]['episodes_detail']}
        e2_only=sum(base[s] and not alt[s] for s in base)
        variant_only=sum(not base[s] and alt[s] for s in base)
        paired[mode]={'E2_only_wins':e2_only,'variant_only_wins':variant_only,
                      'net_variant_wins':variant_only-e2_only,
                      'mcnemar_exact_two_sided_p':exact_mcnemar(e2_only,variant_only)}
    payload={'schema_version':1,'purpose':'E2/E0/E1 wait semantics comparison on the same D1 policy and seeds.',
             'scope':{'deck':list(DECK),'policy_revision':'v4-2026-10-05-double-pea',
                      'seeds':[SEEDS[0],SEEDS[-1]],'episodes_per_variant':len(SEEDS),'workers':WORKERS},
             'variants':results,'paired_vs_E2':paired}
    path=OUT/'eligibility_results.json'
    path.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\\n')
    print(json.dumps({'report':str(path),'wins':{k:v['wins'] for k,v in results.items()},
                      'paired':paired},ensure_ascii=False),flush=True)


if __name__=='__main__': main()

