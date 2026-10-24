"""Outcome-blind legacy timing/selection adapter; no game loop or terminal implementation.

Source plans/trajectories may be copied into an isolated absolute directory. Sampling
writes only there; pass selection-active.json to critical_forks import-legacy.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as d
OUT=None


def check_source_plan():
    plan=d.load(OUT/'plan.json')
    for name,path in plan['paths'].items():
        if name in plan['sha256'] and d.sha(path)!=plan['sha256'][name]:
            raise ValueError('changed frozen input '+name)
    return plan


def selections(plan,count,samples):
    jobs=[]
    for cell in plan['cells']:
        if cell.get('archive'):continue
        trajectory=d.load(OUT/'episodes'/cell['id']/'trajectory.json')
        for point in d.pick_roots(trajectory['eligible'],count,plan['spec']['window_end_tick']):
            jobs.append(dict(cell=cell,point=point,samples=samples,stage='screen'))
    for cell in plan['cells']:
        if not cell.get('archive'):continue
        trajectory=d.load(OUT/'episodes'/cell['id']/'trajectory.json')
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
        folder=OUT/'episodes'/job['cell']['id']/f"root_{job['point']['decision']}"
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


def timing_samples(plan):
    times=defaultdict(list);pilot_jobs=[]
    for manifest in plan['spec'].get('timing_job_manifests',['pilot-jobs.json']):
        pilot_jobs.extend(d.load(OUT/manifest)['jobs'])
    for job in pilot_jobs:
        stage=job.get('stage','pilot')
        folder=OUT/'episodes'/job['cell']['id']/f"root_{job['point']['decision']}"
        if not (folder/f'{stage}-complete.json').exists():raise ValueError('pilot is incomplete')
        root=d.load(folder/'root.json')
        for path in folder.glob(f'{stage}_[0-9]*_0.json'):
            row=d.load(path)
            times[(job['cell']['level'],job['point']['tick'],action_kind(row['identity']['action'],root))].append(row['cpu_seconds'])
    return times


def budget_selections(plan,times,samples=2):
    """Sparse fallback: named roots cover their reset seed; CPU alone picks others."""
    named=[j for j in selections(plan,1,samples) if j['cell'].get('archive')]
    covered={(j['cell']['level'],j['cell']['seed']) for j in named}
    jobs=list(named)
    for cell in plan['cells']:
        if cell.get('archive') or (cell['level'],cell['seed']) in covered:continue
        trajectory=d.load(OUT/'episodes'/cell['id']/'trajectory.json')
        def proxy(point):
            rows=[(abs(tick-point['tick']),values) for (level,tick,kind),values in times.items()
                  if level==cell['level']]
            distance=min(delta for delta,values in rows)
            return point['candidate_count']*statistics.mean(v for delta,values in rows if delta==distance for v in values)
        eligible=[point for point in trajectory['eligible'] if 0<=point['tick']<=plan['spec']['window_end_tick']]
        point=min(eligible,key=lambda point:(proxy(point),point['decision']))
        jobs.append(dict(cell=cell,point=point,samples=samples,stage='screen',cpu_sparse=True))
    expected={(c['level'],c['seed']) for c in plan['cells'] if not c.get('archive')}
    if {(j['cell']['level'],j['cell']['seed']) for j in jobs}!=expected:
        raise ValueError('sparse fallback lost level/reset-seed coverage')
    return jobs


def select():
    destination=OUT/'selection-active.json'
    if destination.exists():
        value=d.load(destination)
        summary=d.load(OUT/'sampling.json')
        if d.sha(destination)!=summary['root_selection_sha256']:raise ValueError('frozen sampling hash changed')
        for cell,digest in value['root_trajectory_sha256'].items():
            if d.sha(OUT/'episodes'/cell/'trajectory.json')!=digest:raise ValueError('frozen root trajectory changed')
        check_source_plan()
        return summary
    plan=check_source_plan();times=timing_samples(plan)
    options=[];selected=None
    for count,samples in [(2,3),(2,2),(1,3),(1,2)]:
        jobs=selections(plan,count,samples);forecast=timing_forecast(jobs,times)
        row=dict(student_roots_per_seed=count,samples=samples,roots=len(jobs),
            labels=sum(j['point']['candidate_count']*samples for j in jobs),
            projected_label_cpu_seconds=forecast,
            conservative_total_cpu_seconds=d.spent_cpu(OUT)+1.2*forecast+1200)
        options.append(row)
        if selected is None and row['conservative_total_cpu_seconds']<=72000:selected=(row,jobs)
    if plan['spec'].get('cpu_sparse_fallback'):
        jobs=budget_selections(plan,times);forecast=timing_forecast(jobs,times)
        row=dict(selection='CPU sparse; named roots cover their reset seed',samples=2,roots=len(jobs),
            labels=sum(j['point']['candidate_count']*2 for j in jobs),projected_label_cpu_seconds=forecast,
            conservative_total_cpu_seconds=d.spent_cpu(OUT)+1.2*forecast+1200)
        options.append(row)
        if selected is None and row['conservative_total_cpu_seconds']<=72000:selected=(row,jobs)
    if selected is None:
        d.save(OUT/'sampling-budget-failed.json',dict(options=options,cpu_seconds=d.spent_cpu(OUT)))
        raise RuntimeError('all complete-panel choices exceed CPU budget; no partial panel substituted')
    row,jobs=selected
    value=dict(jobs=jobs,choice=row,method='CPU timing only; no candidate outcome or value used by selector',
        options=options,root_trajectory_sha256={c['id']:d.sha(OUT/'episodes'/c['id']/'trajectory.json') for c in plan['cells']})
    d.save(destination,value)
    summary=dict(choice=row,options=options,root_selection_sha256=d.sha(destination),
        cpu_seconds_before_screen=d.spent_cpu(OUT),pilot_root_mean_cpu_seconds={f'l{l}-tick{tick}-{kind}':statistics.mean(v) for (l,tick,kind),v in times.items()},
        selection_rule=value['method'],marked_points='six non-L45 points; exact duplicate physical/history roots aliased, no marked point removed')
    d.save(OUT/'sampling.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan',type=Path)
    args=parser.parse_args()
    if not args.plan.is_absolute():parser.error('absolute plan required')
    OUT=args.plan.parent
    select()
