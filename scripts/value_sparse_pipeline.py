"""Continue an approved sparse-label run from its verified live labels8 handle.

Absolute plan and labels8 run.json paths. Advances only after complete processes/data
and the frozen eight-seed gate. Each milestone is recorded/committed/pushed. Stops
at exported data; training/acceptance remain the active goal's subsequent work.
"""
from __future__ import annotations
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from value_teacher_feasibility import ROOT, write_json
from value_teacher_sparse import artifact_bytes, load_plan


class PipelineStop(RuntimeError):
    pass


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return fields[19] # /proc stat field22: start time, including after PID reuse.
    except (FileNotFoundError,ProcessLookupError):return None


def wait_prior(run_path,plan,status):
    run=json.loads(run_path.read_text())
    if run['phase']!='labels' or run['seeds']!=plan['bootstrap_seeds']:
        raise PipelineStop('prior handle is not the frozen labels8 run')
    pid=run['pid'];identity=process_identity(pid)
    if identity is not None:
        arguments=[part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
        if str(ROOT/'scripts/value_teacher_sparse.py') not in arguments or arguments[-1]!='labels8':
            raise PipelineStop('prior PID was reused or is not the labels8 controller')
    status('waiting_labels8',dict(watched_pid=pid,watched_start_ticks=identity,run_path=str(run_path)))
    while identity is not None and process_identity(pid)==identity:
        time.sleep(5)
    result_path=run_path.parent/'result.json'
    if not result_path.exists():raise PipelineStop('labels8 process ended without a completion result')
    result=json.loads(result_path.read_text())
    if result['status']!='complete':raise PipelineStop(result.get('reason') or 'labels8 did not complete')


def git(*args,input=None):
    result=subprocess.run(['git','-C',str(ROOT),*args],input=input,text=True,capture_output=True)
    if result.returncode not in (0,):
        raise PipelineStop(f'git {args[0]} failed: {result.stderr.strip()}')
    return result.stdout


def publish(stage,details,problem=None):
    """Only this task's text conclusion paths may be staged; never include another edit."""
    relative=('MEMORY/docs/task43-value-v5-pool.md','MEMORY/docs/value-v5-sparse-progress.json',
              'MEMORY/'+time.strftime('%Y-%m-%d')+'.md')
    existing=set(git('diff','--cached','--name-only').splitlines())
    if existing-set(relative):raise PipelineStop('unrelated staged files; refuse to commit them')
    doc=ROOT/relative[0];progress=ROOT/relative[1];daily=ROOT/relative[2]
    data=json.loads(progress.read_text()) if progress.exists() else {}
    data.update(stage=stage,task_complete=False,training_complete=False,acceptance_complete=False)
    data.setdefault('milestones',[]).append(dict(stage=stage,details=details))
    write_json(progress,data)
    text=doc.read_text();n=max(map(int,re.findall(r'^### (\d+)\.',text,re.M)))+1
    printable=json.dumps(details,ensure_ascii=False,sort_keys=True)
    entry=f'### {n}. 稀疏扩量阶段 `{stage}`\n\n{printable}\n\n'
    if '## 问题\n' not in text:raise PipelineStop('task result insertion marker missing')
    text=text.replace('## 问题\n',entry+'## 问题\n',1)
    if problem:text=text.replace('## 问题\n',f'## 问题\n\n### 稀疏扩量阶段停止：{stage}\n\n{problem}\n',1)
    doc.write_text(text)
    daily.parent.mkdir(exist_ok=True)
    with daily.open('a') as stream:stream.write(f'\n- task43 稀疏扩量 {stage}：{printable}\n')
    git('add',*[str(ROOT/f) for f in relative])
    changed=git('diff','--cached','--name-only').splitlines()
    if set(changed)-set(relative):raise PipelineStop('staging unexpectedly included other files')
    print(git('diff','--cached','--numstat'),flush=True)
    print(git('diff','--cached','--stat'),flush=True)
    tracked=subprocess.run(['git','-C',str(ROOT),'ls-files','-z',str(ROOT/'artifacts'),str(ROOT/'logs')],capture_output=True,check=True).stdout
    ignored=subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],input=tracked,capture_output=True)
    if ignored.returncode not in (0,1) or ignored.stdout:
        raise PipelineStop('tracked artifact ignore audit failed')
    git('diff','--cached','--check')
    print(git('commit','-m',f'task43：稀疏扩量阶段 {stage}'),flush=True)
    print(git('push','origin','v5'),flush=True)


def advance(plan_path,call,publish_result,status):
    plan=load_plan(plan_path);out=Path(plan['out'])
    call('gate8')
    gate=json.loads((out/'gate8.json').read_text());brief={k:v for k,v in gate.items() if k!='groups'}
    passed=gate.get('approved_to_continue') is True
    publish_result('bootstrap_gate_passed' if passed else 'bootstrap_gate_failed',brief,
                   None if passed else '未满足获批的首8种子门禁；不扩到64，不用不足的数据自动训练。保留全部原始标签，待分析信号/成本来源。')
    if not passed:
        status('gated_stop',brief);return
    for command,name in [('roots64','full_trajectories_complete'),('labels64','full_labels_complete')]:
        status(command,{})
        call(command)
        if command=='roots64':
            results=[json.loads((out/f'seed_{s}'/'trajectory_result.json').read_text()) for s in plan['seeds']]
            selections=[json.loads((out/f'seed_{s}'/'selection.json').read_text()) for s in plan['seeds']]
            details=dict(seeds=len(results),root_games_won=sum(x['won'] for x in results),
                search_roots=sum(x['searched'] for x in results),selected_roots=sum(len(x['roots']) for x in selections))
        else:
            call('report');details={k:v for k,v in json.loads((out/'report.json').read_text()).items() if k!='groups'}
        publish_result(name,details,None)
    status('export',{});call('export')
    data=json.loads((out/'export.json').read_text())
    publish_result('teacher_data_exported',dict(seeds=len(data['seeds']),
        nonterminal_rows=sum(x['rows'] for x in data['seeds']),wins=sum(x['wins'] for x in data['seeds']),
        held_out_seeds=data['held_out_seeds'],artifact_bytes=artifact_bytes(plan['artifact_root'])),None)
    status('data_complete_training_pending',dict(training_complete=False,acceptance_complete=False))


def main():
    plan_path=Path(sys.argv[1]);run_path=Path(sys.argv[2])
    if not plan_path.is_absolute() or not run_path.is_absolute():raise ValueError('absolute paths required')
    plan=load_plan(plan_path);out=Path(plan['out'])
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def status(stage,details):
            write_json(out/'pipeline_status.json',dict(pid=__import__('os').getpid(),stage=stage,
                updated=time.time(),details=details,plan=str(plan_path),watched_run=str(run_path)))
        def call(stage):
            with (out/f'pipeline_{stage}.log').open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/value_teacher_sparse.py'),str(plan_path),stage],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            wait_prior(run_path,plan,status)
            advance(plan_path,call,publish,status)
        except BaseException as exc:
            status('stopped',dict(reason=str(exc)))
            # An active collector enforces stop/cleanup itself. No foreign processes are signalled.
            reason=str(exc)
            try:publish('pipeline_stopped',dict(reason=reason),reason)
            except BaseException as secondary:print(f'publication also stopped: {secondary}',flush=True)
            raise


if __name__=='__main__':main()
