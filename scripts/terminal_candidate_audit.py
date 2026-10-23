"""Shared terminal candidate audit adapted from task45 E and frozen task43 teacher.

Task-specific plans configure paths, windows and roots; production search is untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import fcntl
import gzip
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
MAIN=Path('/home/newbiexvwu/PvZ-search')
OUT=ROOT/'artifacts/task46/phase3'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import search_play as teacher
from render_episode import _frame, load_training_task, training_task_kwargs
from pvz_event_env import EventWaitEnv
from pvz_env import TaskSpec
from pvz_tasks import profile_for_deck
from pvz_value_net import observation_context
def own_engine_pids():
    result=[]
    executable=str(ROOT/'build/pvz-portable').encode()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:argv=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if argv and argv[0]==executable:result.append(int(p.name))
    return result


PLAN=None
STOP=None
RUN=None
REF=None
LAST=0


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def jkey(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def digest_value(value):return hashlib.sha256(jkey(value).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=1,allow_nan=False)+'\n');tmp.replace(path)


def future_seed(cell,decision,k,stage='screen'):
    namespace=PLAN['spec']['namespace'] if PLAN else 'terminal-audit'
    return int.from_bytes(hashlib.sha256(f'{namespace}:{cell}:{decision}:{stage}:{k}'.encode()).digest()[:4],'little')


def all_candidates(obs):
    legal=(obs.get('legal_actions') or {}).get('plants') or []
    actions=[dict(type='plant',packet=p['packet'],row=p['row'],col=p['col']) for p in legal]
    actions.sort(key=lambda x:(x['packet'],x['row'],x['col']))
    shovels=(obs.get('legal_actions') or {}).get('shovels') or []
    actions += sorted([dict(type='shovel',row=a['row'] if isinstance(a,dict) else a[1],
                           col=a['col'] if isinstance(a,dict) else a[0]) for a in shovels],
                      key=lambda a:(a['row'],a['col']))
    if len({jkey(x) for x in actions})!=len(actions):raise ValueError('duplicate engine legal action')
    return [dict(teacher.NOOP)]+actions


def audit_search(env,obs,state,args,point,seed):
    """Observe the unchanged production search, retaining static and halving evidence."""
    import math
    trace={}
    static_frames=teacher._static_frames
    values=teacher._values
    verified=teacher._verified_choice
    def record_static(*pos,**kw):
        result=static_frames(*pos,**kw)
        trace['static_actions']=[deepcopy(a) for a,_ in result]
        return result
    def record_values(frames,static=False):
        result=values(frames,static=static)
        if static:trace['static_values']=list(map(float,result))
        return result
    def finite(numbers):return [float(v) if math.isfinite(v) else None for v in numbers]
    def record_verified(*pos,**kw):
        trace.update(candidates=deepcopy(pos[3]),before_verification_values=finite(pos[4]),
            halving_counts=list(pos[5]),best_index=pos[6],student_index=pos[7])
        result=verified(*pos,**kw)
        trace.update(after_verification_values=finite(pos[4]),chosen=deepcopy(result))
        return result
    before=env._debug_state_sha256()
    income=list(env._sun_production_history);income_start=env._sun_history_start_tick
    baseline=dict(teacher._W)
    stats=REF.stats_new();stats['last_search_tick']=point.get('schedule_last_search_tick',-10**9)
    try:
        teacher._static_frames=record_static;teacher._values=record_values;teacher._verified_choice=record_verified
        chosen=teacher._search(env,obs,state['action'],args,
            seed*10000+point['decision'],stats)
        if not stats['searched']:raise ValueError('frozen selected root was not a scheduled search')
        trace.update(chosen=chosen,verification=stats.get('last_verification'),
            margin=teacher.MARGIN,required_halving_samples=args.samples,
            samples=args.samples,verify_samples=args.verify_samples,per_card=args.per_card,
            student_candidates=args.student_candidates,root_state_preserved=env._debug_state_sha256()==before)
        if not trace['root_state_preserved'] or list(env._sun_production_history)!=income or env._sun_history_start_tick!=income_start:
            raise ValueError('search instrumentation changed physical/income root')
        return trace
    finally:
        teacher._static_frames=static_frames;teacher._values=values;teacher._verified_choice=verified
        teacher._W.clear();teacher._W.update(baseline)


def ready(obs):return sum(d['state']==1 for d in obs.get('defenses',[]))
def process_cpu(env=None):
    own=resource.getrusage(resource.RUSAGE_SELF);children=resource.getrusage(resource.RUSAGE_CHILDREN)
    value=own.ru_utime+own.ru_stime+children.ru_utime+children.ru_stime
    if env is not None and env._process is not None and env._process.poll() is None:
        fields=Path(f'/proc/{env._process.pid}/stat').read_text().rsplit(')',1)[1].split()
        value+=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')
    return value


def spent_cpu():
    total=0.
    for folder in (OUT/'runs').glob('*'):
        receipts=sum(load(p)['cpu_seconds'] for p in folder.glob('usage-*.json'))
        final=folder/'complete.json'
        if final.exists():receipts=max(receipts,load(final)['controller_children_cpu_seconds'])
        total+=receipts
    return total


def artifact_bytes():
    return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())


def progress(env=None,force=False):
    global LAST
    now=time.monotonic()
    if force or now-LAST>=2:
        save(RUN/f'usage-{os.getpid()}.json',dict(pid=os.getpid(),cpu_seconds=process_cpu(env),time=time.time()))
        LAST=now
    if STOP is not None and STOP.is_set():raise RuntimeError('controller budget/error stop; no terminal label saved')


def reference(plan):
    # Bind the pinned task43 helper to this worktree's formal teacher.
    path=Path(plan['reference_path'])
    if sha(path)!=plan['reference_sha256']:raise ValueError('changed pinned task43 helper')
    paths=list(sys.path)
    try:
        spec=importlib.util.spec_from_file_location('task43_frozen_finish',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    finally:sys.path[:]=paths
    if module.teacher is not teacher:raise RuntimeError('task43 helper bound another teacher')
    return module


def teacher_args(plan,cell):
    from curriculum_queue import command_for
    cfg=SimpleNamespace(python=Path(sys.executable),**{k:Path(v) for k,v in plan['paths'].items() if k!='manifest'},
                        workers=1,frames=OUT/'unused',out=OUT/'unused')
    cmd=command_for(cfg,Path(plan['paths']['manifest']),cell['task_id'],0,1,'terminal')
    args=teacher.parser().parse_args(cmd[2:]);args.dagger_beta=1
    return args


def init_worker(plan,run_dir,stop):
    global PLAN,RUN,STOP,REF
    PLAN=plan;RUN=Path(run_dir);STOP=stop
    teacher._init_worker(plan['paths']['value'],plan['paths']['rollout_policy'],61,plan['paths']['static_value'])
    REF=reference(plan)
    progress(force=True)


def open_game(cell):
    args=teacher_args(PLAN,cell)
    env=teacher._W['env']=EventWaitEnv(args.resource_dir,executable=args.executable,headless=True)
    env.load_rollout_net(teacher._W['rollout_file'])
    kw=training_task_kwargs(load_training_task(PLAN['paths']['manifest'],cell['task_id']),cell['seed'])
    task=TaskSpec(level=kw['level'],seed=kw['seed'],playthrough=2,profile=profile_for_deck(tuple(kw['deck'])),
                  **{k:v for k,v in kw['task_extra'].items() if v is not None})
    obs,_=env.reset(deck=tuple(kw['deck']),task=task)
    teacher._W['context']=observation_context(obs);teacher._W['student_history']=None
    return env,obs,args


def root_identity(obs,state):
    return dict(frame=_frame(obs),student_history=REF.history_record(state))


def roots_job(cell):
    folder=OUT/'episodes'/cell['id'];folder.mkdir(parents=True,exist_ok=True)
    dest=folder/'trajectory.json'
    if dest.exists():
        saved=load(dest)
        if saved['cell']!=cell or saved['inputs_sha256']!=PLAN['sha256']:raise ValueError('changed trajectory inputs')
        return dict(cell=cell['id'],roots=len(saved['eligible']),reused=True)
    env=None
    try:
        env,obs,args=open_game(cell);state={};eligible=[];tape=[];frames=[];proposals=[]
        archived=None
        if cell.get('archive'):
            from episode_query import load as load_archive
            _,archived=load_archive(Path(cell['archive']))
        last_search_tick=-10**9
        for decision in range(teacher.MAX_ACTIONS):
            window=cell.get('window_end_tick',PLAN['spec']['window_end_tick'])
            if obs['terminal'] or obs['tick']>window:break
            progress(env)
            # Root history is the one just read by the formal student before executing action.
            action=teacher._fast_action(obs,state)
            teacher._W['student_history']=state.get('history')
            frame=_frame(obs);frame['action']=tape[-1] if tape else None;frames.append(frame)
            if archived is not None:
                exclude={'action','eaten','targets','label','verification','action_rejected'}
                expected={k:v for k,v in archived[decision].items() if k not in exclude}
                actual={k:v for k,v in frame.items() if k not in exclude}
                if jkey(expected)!=jkey(actual):raise ValueError('archive prefix physics changed')
            scheduled=not (args.min_gap and action['type']!='plant' and obs['tick']-last_search_tick<args.min_gap)
            if (obs.get('legal_actions') or {}).get('plants') and scheduled:
                cands=all_candidates(obs)
                eligible.append(dict(decision=decision,tick=obs['tick'],wave=obs['wave'],sun=obs['sun'],
                    candidate_count=len(cands),candidate_sha256=digest_value(cands),state_sha256=env._debug_state_sha256(),
                    public_history_sha256=digest_value(root_identity(obs,state)),
                    schedule_last_search_tick=last_search_tick,prefix='teacher_archive' if archived else 'student'))
                last_search_tick=obs['tick']
            proposals.append(action)
            if archived is not None:action=archived[decision+1]['action']
            tape.append(action);state['action']=action
            obs,*_,info=env.step(action)
            if not info.get('ok'):raise RuntimeError('student trajectory illegal action')
        save(dest,dict(cell=cell,inputs_sha256=PLAN['sha256'],eligible=eligible,tape=tape,proposals=proposals,frames=frames,
                      last_tick=obs['tick'],terminal=obs['terminal']))
        progress(env,True)
        return dict(cell=cell['id'],roots=len(eligible))
    finally:
        teacher._close_worker_env();progress(force=True)


def pick_roots(eligible,count=3,end_tick=10000):
    if not eligible:return []
    # Include initial legal root; remaining roots spread over later elapsed ticks, no outcome inputs.
    targets=[0,end_tick//2,end_tick] if count==3 else [0,end_tick] if count==2 else [end_tick//2]
    selected=[]
    for tick in targets:
        best=min(eligible,key=lambda x:(abs(x['tick']-tick),x['decision']))
        if best not in selected:selected.append(best)
    return sorted(selected,key=lambda x:x['decision'])


def replay_root(cell,point):
    trajectory=load(OUT/'episodes'/cell['id']/'trajectory.json')
    env,obs,args=open_game(cell);state={}
    for decision in range(point['decision']+1):
        progress(env)
        action=teacher._fast_action(obs,state);teacher._W['student_history']=state.get('history')
        if action!=trajectory['proposals'][decision]:raise ValueError('student proposal changed')
        if jkey(_frame(obs))!=jkey({k:v for k,v in trajectory['frames'][decision].items() if k!='action'}):
            raise ValueError('physical prefix differs from stored student trajectory')
        if decision==point['decision']:break
        executed=trajectory['tape'][decision]
        state['action']=executed;obs,*_,info=env.step(executed)
        if not info.get('ok'):raise ValueError('replayed prefix rejected')
    if env._debug_state_sha256()!=point['state_sha256']:raise ValueError('root engine identity differs')
    if digest_value(root_identity(obs,state))!=point['public_history_sha256']:raise ValueError('root history differs')
    return env,obs,state,args


def validate_label(row,identity,trace):
    outcome=row['outcome']
    if (row['identity']!=identity or outcome.get('terminal') is not True or
            outcome.get('result') not in (1,2) or outcome.get('won')!=(outcome['result']==1)):
        raise ValueError('incompatible/incomplete label')
    if not trace.is_file() or sha(trace)!=row['trace_sha256']:raise ValueError('missing or changed teacher action chain')
    required={'root_engine','root_history','root_income','legal_candidate','terminal','no_rejected_actions'}
    if set(row['checks'])!=required or any(v is not True for v in row['checks'].values()):
        raise ValueError('failed or incomplete root/terminal checks')
    with gzip.open(trace,'rt') as stream:records=[json.loads(line) for line in stream]
    if [r['decision'] for r in records]!=list(range(outcome['decisions'])):
        raise ValueError('incomplete teacher decision chain')
    return row


def inspect_job(job):
    """Materialize exact candidate identities for timing forecasts, without terminal labels."""
    cell,point=job['cell'],job['point'];env=None
    folder=OUT/'episodes'/cell['id']/f"root_{point['decision']}";folder.mkdir(parents=True,exist_ok=True)
    try:
        env,obs,state,args=replay_root(cell,point)
        cands=all_candidates(obs)
        if digest_value(cands)!=point['candidate_sha256']:raise ValueError('candidate set changed')
        value=dict(cell=cell,point=point,candidates=cands,student_action=state['action'],
            root_frame=_frame(obs),history=REF.history_record(state),income=list(env._sun_production_history),
            income_start=env._sun_history_start_tick,context=teacher._W['context'].tolist(),inputs_sha256=PLAN['sha256'])
        path=folder/'root.json'
        if path.exists() and load(path)!=json.loads(json.dumps(value)):raise ValueError('inspected root changed')
        save(path,value)
        if not (folder/'search-trace.json').exists():save(folder/'search-trace.json',audit_search(env,obs,state,args,point,cell['seed']))
        return dict(cell=cell['id'],root=point['decision'],candidates=len(cands))
    finally:
        teacher._close_worker_env();progress(force=True)


def labels_job(job):
    cell,point=job['cell'],job['point'];env=None
    folder=OUT/'episodes'/cell['id']/f"root_{point['decision']}";folder.mkdir(parents=True,exist_ok=True)
    results=[]
    try:
        env,obs,state,args=replay_root(cell,point)
        root=env.snapshot();root_hash=env._debug_state_sha256();income=list(env._sun_production_history)
        history_start=env._sun_history_start_tick;baseline=dict(teacher._W)
        cands=all_candidates(obs)
        if len(cands)!=point['candidate_count'] or digest_value(cands)!=point['candidate_sha256']:raise ValueError('root candidate coverage differs')
        initial_mowers=ready(trajectory_first_frame(cell))
        stage=job.get('stage','screen');samples=job.get('samples',3)
        indices=job.get('candidate_indices',list(range(len(cands))))
        if job.get('representative_packets'):
            indices=[0]
            for packet in sorted({a['packet'] for a in cands if a['type']=='plant'}):
                ids=[i for i,a in enumerate(cands) if a.get('packet')==packet]
                indices+=sorted({ids[0],ids[len(ids)//2],ids[-1]})
            shovels=[i for i,a in enumerate(cands) if a['type']=='shovel']
            if shovels:indices.append(shovels[0])
        root_info=dict(cell=cell,point=point,candidates=cands,student_action=state['action'],
            root_frame=_frame(obs),history=REF.history_record(state),income=income,income_start=history_start,
            context=teacher._W['context'].tolist(),inputs_sha256=PLAN['sha256'])
        rp=folder/'root.json'
        if rp.exists() and load(rp)!=json.loads(json.dumps(root_info)):raise ValueError('existing root differs')
        save(rp,root_info)
        search_path=folder/'search-trace.json'
        if not search_path.exists():
            save(search_path,audit_search(env,obs,state,args,point,cell['seed']))
        teacher._W.clear();teacher._W.update(baseline)
        for index in indices:
            cand=cands[index]
            for k in range(samples):
                progress(env)
                future=future_seed(cell['id'],point['decision'],k,stage)
                noise=future_seed(cell['id'],point['decision'],k,stage+'-teacher-search')
                identity=dict(cell=cell['id'],root=point,index=index,action=cand,future=future,
                              search_seed_base=noise,stage=stage,k=k,inputs_sha256=PLAN['sha256'])
                path=folder/f'{stage}_{index}_{k}.json';trace=folder/f'{stage}_{index}_{k}.jsonl.gz'
                if path.exists():
                    results.append(validate_label(load(path),identity,trace));continue
                started=process_cpu(env);wall=time.monotonic()
                env.restore(root);teacher._W.clear();teacher._W.update(baseline)
                teacher._W['student_history']=deepcopy(state.get('history'))
                if env._debug_state_sha256()!=root_hash or list(env._sun_production_history)!=income or env._sun_history_start_tick!=history_start:
                    raise ValueError('snapshot does not restore engine/income context')
                short=teacher._rollout_policy(env,obs,cand,obs['tick']+2500,root,future)
                short_frame=_frame(short)
                v4=teacher._score_terminal_or_value(short)
                if v4 is None:v4=teacher._values([short_frame])[0]
                env.restore(root);env.reseed(future)
                teacher._W.clear();teacher._W.update(baseline)
                actual_state=deepcopy(state);actual_state['action']=cand
                mowers_left=ready(obs);rejected=[];teacher_leaf={}
                def step(environment,action):
                    nonlocal mowers_left
                    new,*_,info=environment.step(action)
                    if not info.get('ok'):
                        rejected.append(action);raise RuntimeError('actual teacher action rejected')
                    if new.get('result')!=1:mowers_left=ready(new)
                    return new
                def tick_progress():
                    progress(env)
                    if not teacher_leaf and env._tick>=obs['tick']+2500:
                        f=_frame(env.observe())
                        teacher_leaf.update(tick=f['tick'],frame=f,value=teacher._values([f])[0])
                original_step=REF.action_step;REF.action_step=step
                temporary=folder/f'{stage}_{index}_{k}.jsonl.tmp'
                try:
                    teacher._W['diagnostic_progress']=tick_progress
                    after=step(env,cand)
                    outcome=REF.finish_teacher(env,after,actual_state,args,noise,temporary)
                finally:
                    REF.action_step=original_step;teacher._W.pop('diagnostic_progress',None)
                if not outcome['terminal']:raise RuntimeError('nonterminal candidate result')
                final=env.observe();outcome['mowers_used']=initial_mowers-mowers_left
                outcome['final_sun']=final['sun']
                if not teacher_leaf:
                    teacher_leaf=dict(tick=final['tick'],frame=_frame(final),value=float(outcome['won']))
                with temporary.open('rb') as src,gzip.open(trace,'wb',compresslevel=1) as dst:dst.write(src.read())
                temporary.unlink()
                row=dict(identity=identity,outcome=outcome,v4_short_value=v4,short_leaf=short_frame,
                    teacher_leaf=teacher_leaf,terminal_state_sha256=env._debug_state_sha256(),
                    checks=dict(root_engine=True,root_history=True,root_income=True,legal_candidate=True,
                                terminal=True,no_rejected_actions=not rejected),trace_sha256=sha(trace),
                    cpu_seconds=process_cpu(env)-started,wall_seconds=time.monotonic()-wall)
                save(path,row);results.append(row);progress(env,True)
                print(json.dumps(dict(cell=cell['id'],root=point['decision'],candidate=index,k=k,
                                     cpu=round(row['cpu_seconds'],3),wave=outcome['wave'],won=outcome['won'])),flush=True)
        env.restore(root)
        if env._debug_state_sha256()!=root_hash:raise ValueError('candidate labels changed root')
        env.release_snapshot(root)
        save(folder/f'{stage}-complete.json',dict(indices=indices,samples=samples,labels=len(results),root_restored=True))
        return dict(cell=cell['id'],root=point['decision'],labels=len(results),cpu_seconds=sum(x['cpu_seconds'] for x in results))
    finally:
        teacher._close_worker_env();progress(force=True)


def trajectory_first_frame(cell):
    f=load(OUT/'episodes'/cell['id']/'trajectory.json')['frames'][0]
    return {'defenses':[dict(row=r,state=s) for r,s in f['mowers']]}


def check_plan():
    plan=load(OUT/'plan.json')
    for k,value in plan['sha256'].items():
        p=ROOT/'scripts/search_play.py' if k=='search_source' else plan['paths'][k]
        if sha(p)!=value:raise ValueError('changed frozen input '+k)
    return plan


def run_jobs(phase,jobs):
    plan=check_plan();OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'stage.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if own_engine_pids():raise RuntimeError('other owned game process live')
        run=OUT/'runs'/f'{int(time.time())}-{phase}';run.mkdir(parents=True)
        stop=mp.get_context('fork').Event()
        started=process_cpu()
        save(run/'controller.json',dict(pid=os.getpid(),phase=phase,jobs=len(jobs),started=time.time(),source_sha256=sha(Path(__file__))))
        with mp.get_context('fork').Pool(4,initializer=init_worker,initargs=(plan,str(run),stop)) as pool:
            function=roots_job if phase=='roots' else inspect_job if phase=='inspect' else labels_job
            pending=pool.imap_unordered(function,jobs)
            results=[]
            try:
                while len(results)<len(jobs):
                    own=resource.getrusage(resource.RUSAGE_SELF)
                    save(run/'usage-controller.json',dict(cpu_seconds=own.ru_utime+own.ru_stime,pid=os.getpid(),time=time.time()))
                    spent=spent_cpu();pids=own_engine_pids()
                    if len(pids)>4 or spent>=plan['spec']['cpu_stop_seconds']:stop.set()
                    if int(time.time())%30<5 and artifact_bytes()>plan['spec']['artifact_budget_bytes']:stop.set()
                    save(run/'progress.json',dict(completed=len(results),jobs=len(jobs),cpu_seconds=spent,engine_pids=pids,stopping=stop.is_set()))
                    try:row=pending.next(timeout=5)
                    except mp.TimeoutError:continue
                    results.append(row);save(run/'results.json',results)
                pool.close();pool.join()
            except BaseException:
                stop.set();pool.close();pool.join();raise
        # Resource receipts count each worker and its waited engine children once.
        save(run/'complete.json',dict(results=results,cpu_receipts=spent_cpu(),controller_children_cpu_seconds=process_cpu()-started))


def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['roots','pilot','screen','inspect'])
    p.add_argument('--plan',type=Path,default=OUT/'plan.json')
    p.add_argument('--jobs',type=Path)
    a=p.parse_args()
    if not a.plan.is_absolute():p.error('absolute plan path required')
    OUT=a.plan.parent
    plan=check_plan()
    if a.phase=='roots':run_jobs('roots',plan['cells'])
    else:
        jobs=load(a.jobs)['jobs']
        run_jobs(a.phase,jobs)

if __name__=='__main__':main()
