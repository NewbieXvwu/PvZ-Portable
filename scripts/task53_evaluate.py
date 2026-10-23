"""Frozen dw4i_s1 vs one small finetune on the preregistered paired task53 resets."""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as f
CONTRACT=None


def init_worker(contract):
    global CONTRACT
    CONTRACT=contract


def game(job):
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    start=f.cpu();reset=deepcopy(job['reset']);deck=tuple(reset.pop('deck'))
    first_aid=reset.pop('first_aid_owned',False)
    reset['preplanted']=tuple(tuple(p) for p in reset.get('preplanted',[]))
    with EventWaitEnv(CONTRACT['resource_dir'],executable=CONTRACT['executable'],headless=True) as env:
        env.load_rollout_net(CONTRACT['binaries'][job['arm']])
        obs,_=env.reset(deck=deck,task=TaskSpec(profile=profile_for_deck(deck,first_aid),**reset))
        initial=sum(d['state']==1 for d in obs['defenses'])
        leaf=env.rollout(None,0)
        if not leaf['terminal'] or leaf['result'] not in (1,2):
            raise ValueError('nonterminal evaluation is not a loss')
        result=dict(id=job['id'],arm=job['arm'],terrain=job['terrain'],seed=reset['seed'],
            won=leaf['result']==1,result=leaf['result'],tick=leaf['tick'],wave=leaf['wave'],
            mowers_used=initial-env._last_response['mowers_left'],
            terminal_state_sha256=env._debug_state_sha256(),model_sha256=CONTRACT['models'][job['arm']]['sha256'])
    result['cpu_seconds']=f.cpu()-start
    return result


def summary(rows,evaluation):
    from pvz_promotion import summarize
    keyed={(r['id'],r['arm']):r for r in rows}
    expected={(j['id'],arm) for j in evaluation for arm in ('incumbent','candidate')}
    if set(keyed)!=expected or len(rows)!=len(expected):
        raise ValueError('paired evaluation incomplete or duplicated')
    pairs=[dict(id=j['id'],terrain=j['terrain'],seed=j['reset']['seed'],
                incumbent=keyed[(j['id'],'incumbent')]['won'],candidate=keyed[(j['id'],'candidate')]['won'])
           for j in evaluation]
    by_terrain={t:summarize([p for p in pairs if p['terrain']==t]) for t in ('day','night','pool')}
    result=dict(status='complete',overall=summarize(pairs),by_terrain=by_terrain,paired=pairs,
                interval='existing pvz_promotion paired Newcombe method 10, 95%; retains uncertainty on unanimous outcomes',
                deployment_note='Feasibility experiment only; no promotion certificate, default or qc changes.')
    return result


def run(plan,model):
    import torch
    from pvz_rollout_policy import export_student,load_student
    torch.set_num_threads(1)
    out=Path(plan['out']);directory=out/'evaluation';directory.mkdir(parents=True,exist_ok=True)
    registration=f.load(out/'training-registration.json');evaluation=registration['evaluation']
    models={a:dict(path=str(p),sha256=f.sha(p)) for a,p in
            [('incumbent',Path(plan['teacher']['rollout_policy'])),('candidate',Path(model))]}
    contract=dict(models=models,executable=plan['teacher']['executable'],
                  executable_sha256=f.sha(plan['teacher']['executable']),resource_dir=plan['teacher']['resource_dir'],
                  registration_sha256=f.sha(out/'training-registration.json'),binaries={})
    for arm,record in models.items():
        path=directory/f'{arm}.bin';export_student(load_student(record['path']),path)
        contract['binaries'][arm]=str(path)
    path=directory/'contract.json'
    if path.exists() and f.load(path)!=contract:raise ValueError('evaluation inputs changed')
    f.save(path,contract)
    output=directory/'games.jsonl';rows=[json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    done={(r['id'],r['arm']) for r in rows}
    jobs=[dict(j,arm=arm) for j in evaluation for arm in ('incumbent','candidate') if (j['id'],arm) not in done]
    with (out/'collector.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run=out/'runs'/f'{int(time.time())}-paired-eval';run.mkdir(parents=True)
        started=f.cpu()
        f.save(run/'controller.json',dict(pid=os.getpid(),phase='paired-eval',jobs=len(jobs)))
        with output.open('a') as stream,mp.get_context('fork').Pool(4,initializer=init_worker,initargs=(contract,)) as pool:
            for row in pool.imap_unordered(game,jobs):
                rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                f.save(run/'progress.json',dict(completed=len(rows),total=len(evaluation)*2,
                                               worker_game_cpu_seconds=sum(r['cpu_seconds'] for r in rows)))
            pool.close();pool.join()
        f.save(run/'complete.json',dict(cpu_seconds=f.cpu()-started,games=len(rows)))
    result=summary(rows,evaluation);result['contract']=contract
    f.save(ROOT/'MEMORY/docs/task53-critical-forks-evaluation.json',result)
    print(json.dumps({k:result[k] for k in ('overall','by_terrain')},ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path);p.add_argument('model',type=Path)
    a=p.parse_args();run(f.check_plan(a.plan),a.model)


if __name__=='__main__':main()
