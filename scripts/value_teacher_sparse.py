"""Approved task43 sparse teacher labels. Absolute plan, then roots8/labels8/gate8/
roots64/labels64/report/export. Reuses the frozen teacher and the round2 leaf channel.
"""
from __future__ import annotations
from argparse import Namespace
from collections import Counter
from copy import deepcopy
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

from value_teacher_feasibility import (ROOT, action_step, cpu, digest, group_summary,
    process_sample, sample_root, stats_new, validate_existing_candidate, write_json)
import search_play as teacher

_STOP = None
_PLAN = None
_PHASE = None
_RUN = None


class CollectionStop(RuntimeError):
    pass


def json_value(value):
    return json.loads(json.dumps(value))


def spaced_roots(roots, count=12):
    """Midpoint quantiles of actual search ordinals; terminal candidate labels are not inputs."""
    n = len(roots)
    if n <= count:
        return list(roots)
    return [roots[(2*i+1)*n//(2*count)] for i in range(count)]


def artifact_bytes(directory):
    total = 0
    for parent, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(parent)/d).is_symlink()]
        for name in files:
            p = Path(parent)/name
            try:
                if not p.is_symlink(): total += p.stat().st_size
            except FileNotFoundError:
                pass
    return total


def load_plan(filename):
    p = Path(filename)
    if not p.is_absolute(): raise ValueError('absolute plan path required')
    plan = json.loads(p.read_text())
    if len(plan['seeds']) != 64 or len(set(plan['seeds'])) != 64:
        raise ValueError('exactly 64 unique frozen seeds required')
    if not 1 <= plan['workers'] <= 4 or plan['roots_per_seed'] != 12:
        raise ValueError('approved design: at most 4 engines, 12 roots per seed')
    if len(plan['bootstrap_seeds']) != 8 or len(set(plan['bootstrap_seeds'])) != 8 or not set(plan['bootstrap_seeds']) <= set(plan['seeds']):
        raise ValueError('eight planned bootstrap seeds required')
    if len(plan['held_out_seeds']) < 8 or len(set(plan['held_out_seeds'])) != len(plan['held_out_seeds']) or not set(plan['held_out_seeds']) <= set(plan['seeds']):
        raise ValueError('at least eight held-out seeds required')
    for name in ('value','static_value','rollout_policy','executable','manifest'):
        f = Path(plan['teacher'][name])
        if not f.is_absolute() or digest(f) != plan['sha256'][name]:
            raise ValueError(f'changed or relative frozen input: {name}')
    if digest(ROOT/'scripts/search_play.py') != plan['search_source_sha256']:
        raise ValueError('formal teacher source changed')
    for key in ('out','artifact_root'):
        if not Path(plan[key]).is_absolute(): raise ValueError('absolute output/artifact root required')
    return plan


def open_game(args, seed):
    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_tasks import profile_for_deck
    from pvz_value_net import observation_context
    from render_episode import load_training_task, training_task_kwargs
    env = teacher._W['env'] = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(teacher._W['rollout_file'])
    kw = training_task_kwargs(load_training_task(Path(args.manifest),args.task_id),seed)
    task = TaskSpec(level=kw['level'],seed=seed,playthrough=2,
        profile=profile_for_deck(tuple(kw['deck']),kw.get('first_aid_owned',False)),
        **{k:v for k,v in kw['task_extra'].items() if v is not None})
    obs,_ = env.reset(deck=tuple(kw['deck']),task=task)
    teacher._W['context'] = observation_context(obs)
    teacher._W['student_history'] = None
    return env,obs


def verify_selection(directory, seed):
    frozen = json.loads((directory/'selection.json').read_text())
    original = json.loads((directory/'trajectory_result.json').read_text())
    if original['teacher_sha256'] != _PLAN['sha256']:
        raise ValueError('trajectory used different frozen teacher inputs')
    if frozen['seed'] != seed or frozen['trajectory_sha256'] != digest(directory/'root_trajectory.jsonl'):
        raise ValueError('trajectory/selection changed')
    expected = spaced_roots(original['search_roots'],_PLAN['roots_per_seed'])
    if frozen['roots'] != expected:
        raise ValueError('selected roots differ from predeclared midpoint quantiles')
    return frozen,original


def root_trajectory(seed, directory, env, obs, args, progress):
    from render_episode import _frame
    state={};stats=stats_new();roots=[]
    trace_tmp=directory/'root_trajectory.jsonl.tmp'
    with trace_tmp.open('w') as trace:
        for decision in range(teacher.MAX_ACTIONS):
            if obs.get('terminal'): break
            progress()
            proposed = teacher._fast_action(obs,state)
            teacher._W['student_history'] = state.get('history')
            eligible = bool((obs.get('legal_actions') or {}).get('plants')) and not (
                args.min_gap and proposed['type'] != 'plant' and
                obs['tick']-stats.get('last_search_tick',-10**9) < args.min_gap)
            root_hash = env._debug_state_sha256() if eligible else None
            before=stats['searched']
            chosen=teacher._search(env,obs,proposed,args,seed*10000+decision,stats)
            if stats['searched'] != before:
                if not eligible: raise RuntimeError('search schedule differs from teacher')
                roots.append(dict(decision=decision,tick=obs['tick'],wave=obs['wave'],
                                  state_sha256=root_hash))
            trace.write(json.dumps(dict(decision=decision,tick=obs['tick'],wave=obs['wave'],
                                       student=proposed,chosen=chosen))+'\n')
            state['action']=chosen
            obs=action_step(env,chosen)
        else:
            raise RuntimeError('root trajectory hit action cap')
    trace_tmp.replace(directory/'root_trajectory.jsonl')
    result=dict(seed=seed,terminal=True,won=obs.get('result')==1,decisions=decision,
        searched=stats['searched'],overrides=stats['overrides'],search_roots=roots,
        final_frame=_frame(obs),teacher_sha256=_PLAN['sha256'])
    write_json(directory/'trajectory_result.json',result)
    # This file is frozen before any candidate terminal labels are read or generated.
    write_json(directory/'selection.json',dict(seed=seed,
        roots=spaced_roots(roots,_PLAN['roots_per_seed']),
        trajectory_sha256=digest(directory/'root_trajectory.jsonl'),
        method='midpoint quantiles of complete actual search order; no candidate outcomes'))
    return dict(seed=seed,phase='roots',roots=len(roots),selected=min(len(roots),12),status='complete')


def collect(seed, directory, env, obs, args, progress):
    from render_episode import _frame
    selected,original=verify_selection(directory,seed)
    chosen_roots={r['decision']:r for r in selected['roots']}
    all_roots={r['decision']:r for r in original['search_roots']}
    rows=[json.loads(x) for x in (directory/'root_trajectory.jsonl').read_text().splitlines()]
    stats=stats_new();state={};groups=[]
    for index,row in enumerate(rows):
        progress()
        if row['decision'] != index or obs['tick'] != row['tick'] or obs['wave'] != row['wave']:
            raise RuntimeError('saved root trajectory identity differs')
        proposed=teacher._fast_action(obs,state)
        teacher._W['student_history']=state.get('history')
        if proposed != row['student']: raise RuntimeError('replayed student context differs')
        if index in chosen_roots:
            if env._debug_state_sha256() != chosen_roots[index]['state_sha256']:
                raise RuntimeError('replayed selected root state differs')
            chosen,summary=sample_root(env,obs,state,args,seed*10000+index,stats,
                                      directory/f'root_{index}')
            if chosen != row['chosen']: raise RuntimeError('replayed teacher choice differs')
            groups.append(summary)
            progress(force=True)
        # Reconstruct the actual teacher search gap even for roots not labelled on this pass.
        if index in all_roots: stats['last_search_tick']=obs['tick']
        state['action']=row['chosen']
        obs=action_step(env,row['chosen'])
    if json_value(_frame(obs)) != original['final_frame']:
        raise RuntimeError('replayed full root final frame differs')
    result=dict(seed=seed,status='complete',groups=groups,
        selection_sha256=digest(directory/'selection.json'),root_final_state_equal=True)
    write_json(directory/'labels_result.json',result)
    return dict(seed=seed,phase='labels',status='complete',groups=len(groups))


def worker(seed):
    if _STOP.is_set(): return dict(seed=seed,status='stopped',reason='controller stopped')
    args=Namespace(**_PLAN['teacher'])
    directory=Path(_PLAN['out'])/f'seed_{seed}';directory.mkdir(parents=True,exist_ok=True)
    completed=directory/('trajectory_result.json' if _PHASE=='roots' else 'labels_result.json')
    if completed.exists():
        verify_selection(directory,seed)
        if _PHASE=='labels': audit_seed(directory,seed)
        return dict(seed=seed,status='complete',reused=True,phase=_PHASE)
    env=None;start=cpu();wall=time.monotonic();last=0;last_cpu=0
    teacher._init_worker(args.value,args.rollout_policy,args.rollout_sim_skip,args.static_value)
    usage_path=directory/'usage'/f'{_PHASE}_{_RUN}.json'
    def progress(force=False, stopping=True):
        nonlocal last,last_cpu
        now=time.monotonic()
        if force or now-last>=5:
            try:measured=cpu(env)-start
            except (FileNotFoundError,ProcessLookupError):measured=max(last_cpu,cpu()-start)
            last_cpu=measured
            write_json(usage_path,dict(seed=seed,phase=_PHASE,run=_RUN,pid=os.getpid(),
                engine_pid=env._process.pid if env is not None and env._process is not None else None,
                cpu_seconds=measured,wall_seconds=now-wall,updated=time.time()))
            last=now
        if stopping and _STOP.is_set():raise CollectionStop('CPU, disk, or controller error stop')
    def terminate(_signum,_frame):
        try: progress(force=True,stopping=False)
        finally: teacher._close_worker_env()
        raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    try:
        env,obs=open_game(args,seed)
        teacher._W['diagnostic_progress']=progress
        progress(force=True)
        return (root_trajectory if _PHASE=='roots' else collect)(seed,directory,env,obs,args,progress)
    except BaseException as exc:
        _STOP.set()
        error=dict(seed=seed,status='stopped' if isinstance(exc,CollectionStop) else 'error',
                   reason=str(exc),traceback=traceback.format_exc())
        write_json(directory/f'{_PHASE}_error_{_RUN}.json',error)
        return error
    finally:
        if env is None:env=teacher._W.get('env')
        if env is not None:progress(force=True,stopping=False)
        teacher._W.pop('diagnostic_progress',None)
        teacher._close_worker_env()


def used_cpu(out,seeds=None):
    total=sum(json.loads(p.read_text())['cpu_seconds'] for p in out.glob('seed_*/usage/*.json')
              if seeds is None or int(p.parent.parent.name[5:]) in seeds)
    for p in out.glob('runs/*/controller_usage.json'):
        item=json.loads(p.read_text())
        if seeds is None or set(item['seeds'])<=seeds:total+=item['cpu_seconds']
    return total


def run_phase(plan, phase, seeds):
    global _PLAN,_PHASE,_STOP,_RUN
    controller_start=cpu()
    _PLAN=plan;_PHASE=phase;_RUN=f'{int(time.time())}_{uuid.uuid4().hex[:8]}'
    _STOP=mp.get_context('fork').Event()
    out=Path(plan['out']);out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if process_sample()['own_engines']:
            raise RuntimeError('another owned collector has live engines; preserve four-engine cap')
        if artifact_bytes(plan['artifact_root'])>plan['artifact_budget_bytes']:
            raise CollectionStop('artifact size exceeds approved 5 GB limit')
        run_path=out/'runs'/_RUN;run_path.mkdir(parents=True)
        write_json(run_path/'run.json',dict(pid=os.getpid(),started=time.time(),phase=phase,
            seeds=seeds,plan=plan,source_sha256=digest(Path(__file__))))
        teacher._load_nets(plan['teacher']['value'],plan['teacher']['rollout_policy'],plan['teacher']['static_value'])
        teacher._freeze_startup();results=[];stop_reason=None
        with mp.get_context('fork').Pool(plan['workers']) as pool:
            pending=pool.imap_unordered(worker,seeds)
            with (run_path/'resources.jsonl').open('w') as monitor:
                while len(results)<len(seeds):
                    write_json(run_path/'controller_usage.json',dict(seeds=seeds,cpu_seconds=cpu()-controller_start))
                    sample=process_sample();sample['cpu_seconds']=used_cpu(out)
                    sample['artifact_bytes']=artifact_bytes(plan['artifact_root'])
                    monitor.write(json.dumps(sample)+'\n');monitor.flush()
                    if sample['own_engines']>4:stop_reason='own engine count exceeds four'
                    # Stop early enough to absorb two checkpoint intervals on four single-CPU lanes.
                    if sample['cpu_seconds']>=plan['cpu_budget_seconds']-40:stop_reason='60 CPU-hour budget reached'
                    if sample['artifact_bytes']>plan['artifact_budget_bytes']:stop_reason='5 GB artifact budget exceeded'
                    if stop_reason:_STOP.set()
                    try:result=pending.next(timeout=5)
                    except mp.TimeoutError:continue
                    results.append(result)
                    if result['status']!='complete':_STOP.set();stop_reason=stop_reason or result.get('reason','worker stopped')
                    write_json(run_path/'progress.json',dict(results=results,stop_reason=stop_reason))
                pool.close();pool.join()
        write_json(run_path/'controller_usage.json',dict(seeds=seeds,cpu_seconds=cpu()-controller_start))
        completed=not stop_reason and all(r['status']=='complete' for r in results)
        write_json(run_path/'result.json',dict(status='complete' if completed else 'stopped',
            phase=phase,seeds=seeds,results=results,reason=stop_reason,cpu_seconds=used_cpu(out),
            artifact_bytes=artifact_bytes(plan['artifact_root']),finished=time.time()))
        if not completed:raise CollectionStop(stop_reason)
        print(json.dumps(dict(phase=phase,seeds=len(seeds),status='complete',cpu_seconds=used_cpu(out))),flush=True)


def audit_seed(directory,seed):
    selected,original=verify_selection(directory,seed)
    done=json.loads((directory/'labels_result.json').read_text())
    if done['seed']!=seed:raise ValueError('completed seed identity differs')
    if done['selection_sha256']!=digest(directory/'selection.json') or not done['root_final_state_equal']:
        raise ValueError('completed seed does not match frozen root selection')
    directories={int(p.name[5:]) for p in directory.glob('root_*') if p.is_dir()}
    if directories!={x['decision'] for x in selected['roots']}:
        raise ValueError('label roots do not exactly cover frozen selection')
    groups=[]
    for selected_root in selected['roots']:
        d=directory/f"root_{selected_root['decision']}";root=json.loads((d/'root.json').read_text())
        if root['decision_seed']!=seed*10000+selected_root['decision'] or root['common_future']!=(root['decision_seed']*1000)%2**32:
            raise ValueError('root decision/common future identity differs')
        if root['root_state_sha256']!=selected_root['state_sha256']:
            raise ValueError('selected root hash differs')
        paths=list(d.glob('candidate_[0-9]*.json'));rows=[json.loads(p.read_text()) for p in paths]
        n=len(root['candidates'])
        if len(rows)!=n or {r['candidate'] for r in rows}!=set(range(n)):
            raise ValueError('missing or duplicate candidate labels')
        for row in rows:
            from value_teacher_feasibility_report import validate_income
            validate_income(root,root['root_frame']['tick']);validate_income(row,row['leaf_frame']['tick'])
            expected=dict(action=root['candidates'][row['candidate']],common_future=root['common_future'],
                continuation_seed_base=root['decision_seed']+5000,
                history_checks=dict(frame=True,engine_state=True,income=True,income_start=True))
            validate_existing_candidate(row,expected)
            trace=d/f"candidate_{row['candidate']}_teacher.jsonl"
            actions=[json.loads(line) for line in trace.read_text().splitlines()]
            if [a['decision'] for a in actions]!=list(range(row['outcome']['decisions'])):
                raise ValueError('teacher continuation trace is incomplete')
        rows.sort(key=lambda x:x['candidate'])
        summary=group_summary(rows);saved=json.loads((d/'summary.json').read_text())
        if any(saved[k]!=v for k,v in summary.items()):raise ValueError('summary/raw candidate mismatch')
        nt=[r for r in rows if not r['leaf_frame']['terminal']];w=sum(r['outcome']['won'] for r in nt)
        summary.update(seed=seed,decision=selected_root['decision'],wave=root['root_frame']['wave'],
            nonterminal_labels=len(nt),nonterminal_wins=w,nonterminal_pairs=w*(len(nt)-w),
            nonterminal_mixed=0<w<len(nt))
        groups.append(summary)
    return groups


def report(plan,seeds,gate=False):
    global _PLAN
    _PLAN=plan;out=Path(plan['out']);groups=[]
    for seed in seeds:groups.extend(audit_seed(out/f'seed_{seed}',seed))
    pairs=sum(g['nonterminal_pairs'] for g in groups);used=used_cpu(out,set(seeds))
    forecast=used*64/len(seeds)
    value=dict(status='complete',seeds=seeds,root_groups=len(groups),labels=sum(g['labels'] for g in groups),
        wins=sum(g['wins'] for g in groups),mixed_groups=sum(g['mixed'] for g in groups),
        effective_pairs=sum(g['effective_pairs'] for g in groups),nonterminal_effective_pairs=pairs,
        nonterminal_labels=sum(g['nonterminal_labels'] for g in groups),
        nonterminal_wins=sum(g['nonterminal_wins'] for g in groups),cpu_seconds=used,
        projected_64_cpu_hours=forecast/3600,artifact_bytes=artifact_bytes(plan['artifact_root']),groups=groups)
    if gate:
        value['approved_to_continue']=(pairs>=plan['bootstrap_min_pairs'] and
            forecast<=plan['cpu_budget_seconds'] and value['artifact_bytes']<=plan['artifact_budget_bytes'])
        write_json(out/'gate8.json',value)
    else:write_json(out/'report.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='groups'},indent=1),flush=True)
    return value


def export(plan):
    import numpy as np
    import torch
    from pvz_value_net import encode_frame
    report(plan,plan['seeds'])
    out=Path(plan['out']);target=out/'data';target.mkdir(exist_ok=True)
    info=[]
    for seed in plan['seeds']:
        boards=[];scalars=[];legacy=[];won=[];positions=[];candidates=[];packets=[]
        for root_path in sorted((out/f'seed_{seed}').glob('root_*/root.json')):
            root=json.loads(root_path.read_text());context=torch.tensor(root['context'],dtype=torch.float32)
            for p in sorted(root_path.parent.glob('candidate_[0-9]*.json')):
                row=json.loads(p.read_text());frame=row['leaf_frame']
                if frame['terminal']:continue # These states bypass the value net in the real teacher.
                b,s=encode_frame(frame,2);_,s1=encode_frame(frame,1)
                boards.append(b.half());scalars.append(torch.cat([s,context]));legacy.append(torch.cat([s1,context]))
                won.append(float(row['outcome']['won']));positions.append(root['decision_seed']%10000)
                candidates.append(row['candidate']);packets.append(row['action'].get('packet',-1))
        if not boards:raise ValueError(f'no nonterminal value rows for seed {seed}')
        n=len(boards);path=target/f'seed_{seed}.pt';tmp=path.with_suffix('.pt.tmp')
        torch.save(dict(boards=torch.stack(boards),scalars=torch.stack(legacy),scalars_v2=torch.stack(scalars),
            won=torch.tensor(won),seed=torch.full((n,),seed,dtype=torch.long),
            position=torch.tensor(positions),candidate=torch.tensor(candidates),packet=torch.tensor(packets),
            kind=torch.zeros(n,dtype=torch.long),
            config=dict(task_id=plan['teacher']['task_id'],root_policy='frozen search teacher',
                        continuation_policy='same frozen search teacher',samples=1,horizon=plan['teacher']['horizon'])),tmp)
        tmp.replace(path);info.append(dict(seed=seed,rows=n,wins=sum(won),sha256=digest(path)))
    write_json(out/'export.json',dict(status='complete',seeds=info,held_out_seeds=plan['held_out_seeds']))


def main():
    plan=load_plan(sys.argv[1]);stage=sys.argv[2];out=Path(plan['out'])
    bootstrap=plan['bootstrap_seeds']
    if stage in ('roots64','labels64','report','export'):
        gate=json.loads((out/'gate8.json').read_text())
        if not gate.get('approved_to_continue'):raise CollectionStop('eight-seed gate did not pass')
    if stage in ('roots8','labels8','roots64','labels64'):
        seeds=bootstrap if stage.endswith('8') else plan['seeds']
        run_phase(plan,'roots' if stage.startswith('roots') else 'labels',seeds)
    elif stage=='gate8':report(plan,bootstrap,gate=True)
    elif stage=='report':report(plan,plan['seeds'])
    elif stage=='export':export(plan)
    else:raise ValueError('unknown stage')


if __name__=='__main__':main()
