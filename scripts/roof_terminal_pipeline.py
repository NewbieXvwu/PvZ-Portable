"""Freeze roof sampling from CPU timing only, then run the shared terminal audit."""
from __future__ import annotations
from collections import defaultdict
import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import terminal_candidate_audit as d


def selections(plan,count,samples):
    jobs=[]
    for cell in plan['cells']:
        if cell.get('archive'):continue
        trajectory=d.load(d.OUT/'episodes'/cell['id']/'trajectory.json')
        for point in d.pick_roots(trajectory['eligible'],count):
            jobs.append(dict(cell=cell,point=point,samples=samples,stage='screen'))
    for cell in plan['cells']:
        if not cell.get('archive'):continue
        trajectory=d.load(d.OUT/'episodes'/cell['id']/'trajectory.json')
        for decision in plan['spec']['marked_decisions_1based'][str(cell['level'])]:
            point=next(x for x in trajectory['eligible'] if x['decision']==decision-1)
            duplicate=next((j for j in jobs if j['cell']['level']==cell['level'] and j['cell']['seed']==cell['seed']
                and j['point']['state_sha256']==point['state_sha256']
                and j['point']['public_history_sha256']==point['public_history_sha256']),None)
            if duplicate is not None:
                duplicate.setdefault('marked_point_aliases',[]).append(dict(cell=cell['id'],decision=decision))
            else:jobs.append(dict(cell=cell,point=point,samples=samples,stage='screen',marked_point=True))
    return jobs


def action_kind(action,root):
    if action['type']!='plant':return action['type']
    packets={p[0]:p[1] for p in root['root_frame']['packets']}
    return 'plant:'+str(packets[action['packet']])


def timing_forecast(jobs,times):
    forecast=0.
    for job in jobs:
        level=job['cell']['level']
        folder=d.OUT/'episodes'/job['cell']['id']/f"root_{job['point']['decision']}"
        root=d.load(folder/'root.json')
        for action in root['candidates']:
            kind=action_kind(action,root)
            available=[(abs(tick-job['point']['tick']),v) for (l,tick,k),v in times.items() if l==level and k==kind]
            if available:
                distance=min(t[0] for t in available)
                values=[statistics.mean(v) for delta,v in available if delta==distance]
            else:
                values=[statistics.mean(v) for (l,tick,k),v in times.items() if k==kind]
            if not values:raise ValueError('missing timing for legal action kind '+kind)
            forecast+=statistics.mean(values)*job['samples']
    return forecast


def select():
    plan=d.check_plan();times=defaultdict(list);pilot=d.load(d.OUT/'pilot-jobs.json')
    for job in pilot['jobs']:
        folder=d.OUT/'episodes'/job['cell']['id']/f"root_{job['point']['decision']}"
        if not (folder/'pilot-complete.json').exists():raise ValueError('pilot is incomplete')
        root=d.load(folder/'root.json')
        for path in folder.glob('pilot_[0-9]*_0.json'):
            row=d.load(path)
            times[(job['cell']['level'],job['point']['tick'],action_kind(row['identity']['action'],root))].append(row['cpu_seconds'])
    options=[];selected=None
    for count,samples in [(2,3),(2,2),(1,3),(1,2)]:
        jobs=selections(plan,count,samples);forecast=timing_forecast(jobs,times)
        row=dict(student_roots_per_seed=count,samples=samples,roots=len(jobs),
            labels=sum(j['point']['candidate_count']*samples for j in jobs),
            projected_label_cpu_seconds=forecast,
            conservative_total_cpu_seconds=d.spent_cpu()+1.2*forecast+1200)
        options.append(row)
        if selected is None and row['conservative_total_cpu_seconds']<=72000:selected=(row,jobs)
    if selected is None:
        d.save(d.OUT/'sampling-budget-failed.json',dict(options=options,cpu_seconds=d.spent_cpu()))
        raise RuntimeError('all complete-panel choices exceed CPU budget; no partial panel substituted')
    row,jobs=selected
    value=dict(jobs=jobs,choice=row,method='CPU timing only; no candidate outcome or value used by selector',
        options=options,root_trajectory_sha256={c['id']:d.sha(d.OUT/'episodes'/c['id']/'trajectory.json') for c in plan['cells']})
    destination=d.OUT/'selection-active.json'
    if destination.exists() and d.load(destination)!=value:raise ValueError('sampling is already frozen')
    d.save(destination,value)
    summary=dict(choice=row,options=options,root_selection_sha256=d.sha(destination),
        cpu_seconds_before_screen=d.spent_cpu(),pilot_root_mean_cpu_seconds={f'l{l}-tick{tick}-{kind}':statistics.mean(v) for (l,tick,kind),v in times.items()},
        selection_rule=value['method'],marked_points='six non-L45 points; exact duplicate physical/history roots aliased, no marked point removed')
    d.save(ROOT/'MEMORY/docs/task46-roof-terminal-sampling.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def publish(stage,note,paths):
    today=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
    book=ROOT/'MEMORY/docs/task46-roof-zone.md';daily=ROOT/f'MEMORY/{today}.md'
    marker=f'<!-- task46-B-{stage} -->'
    if marker not in book.read_text():
        book.write_text(book.read_text().replace('## 问题',marker+'\n- '+note+'\n\n## 问题',1))
        with daily.open('a') as stream:stream.write('\n- task46追加B：'+note+'\n')
    paths=[str(ROOT/name) for name in paths]+[str(book),str(daily)]
    def git(*args):return subprocess.run(['git','-C',str(ROOT),*args],check=True,capture_output=True,text=True)
    if git('branch','--show-current').stdout.strip()!='roof':raise RuntimeError('wrong branch')
    staged=git('diff','--cached','--name-only').stdout
    if staged:raise RuntimeError('staged work already present')
    tracked=git('ls-files','-z',str(ROOT/'artifacts'),str(ROOT/'logs')).stdout
    ignored=subprocess.run(['git','-C',str(ROOT),'check-ignore','-z','--stdin','--no-index'],input=tracked,capture_output=True,text=True)
    if ignored.returncode not in (0,1) or ignored.stdout:raise RuntimeError('tracked ignored artifacts exist')
    git('add',*paths)
    stat=git('diff','--cached','--numstat').stdout
    if any(line.startswith('-') for line in stat.splitlines()):raise RuntimeError('binary staged')
    print(git('diff','--cached','--stat').stdout,flush=True)
    print(git('commit','-m',f'task46: {stage} full-legal roof terminal audit').stdout,flush=True)
    print(git('push','origin','roof').stderr,flush=True)


def run():
    plan=d.check_plan();inspect={}
    for count in (1,2):
        for job in selections(plan,count,2):inspect[(job['cell']['id'],job['point']['decision'])]=job
    d.run_jobs('inspect',list(inspect.values()))
    summary=select()
    publish('sampling-frozen',f"第三阶段CPU计时取点已冻结：{summary['choice']['roots']}根、{summary['choice']['samples']}共同未来、{summary['choice']['labels']}条完整候选终局标签；每关2seed，保留全部合法种植/铲子/等待和六个非L45旧点。按计时保守预测共{summary['choice']['conservative_total_cpu_seconds']/3600:.3f}CPU小时≤20；未根据终局结果选根。源工具复用关系和CPU门禁见task46-roof-terminal-plan.json / sampling.json。",
        ['scripts/terminal_candidate_audit.py','scripts/terminal_candidate_report.py','scripts/roof_terminal_pipeline.py','python/test_terminal_candidate_report.py','MEMORY/docs/task46-roof-terminal-sampling.json'])
    selection=d.load(d.OUT/'selection-active.json')
    d.run_jobs('screen',selection['jobs'])
    import terminal_candidate_report as report
    value=report.report()
    publish('screen-complete',f"第三阶段全候选筛查完成：{value['overall']['roots']}根、{value['overall']['labels']}条真实老师终局标签逐根核验齐全；候选覆盖/裁剪/叶值排序/锚定分解见task46-roof-terminal-results.json。当前计量{d.spent_cpu()/3600:.3f}CPU小时。尚需独立复现与终验，未改门槛、生产参数或qc。",
        ['MEMORY/docs/task46-roof-terminal-results.json'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['run','select'],default='run',nargs='?')
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--wait-run',type=Path)
    args=parser.parse_args()
    if args.wait_pid:
        if not args.wait_run or not args.wait_run.is_absolute():parser.error('absolute --wait-run required')
        receipt=d.load(args.wait_run/'controller.json')
        if receipt['pid']!=args.wait_pid or receipt['phase']!='pilot':raise RuntimeError('pilot handle mismatch')
        def identity():
            try:
                fields=Path(f'/proc/{args.wait_pid}/stat').read_text().rsplit(')',1)[1].split()
                return fields[19] if fields[0] not in ('Z','X') else None
            except (FileNotFoundError,ProcessLookupError):return None
        start=identity()
        if start is None and not (args.wait_run/'complete.json').exists():raise RuntimeError('pilot is not live and has no completion receipt')
        while start is not None and identity()==start:time.sleep(5)
        if not (args.wait_run/'complete.json').exists():raise RuntimeError('pilot ended without success; never restart automatically')
    if args.phase=='select':select()
    else:run()
