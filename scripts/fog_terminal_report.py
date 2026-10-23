"""Audit full terminal panels and compare ranking without treating candidate pairs as episodes."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import fog_terminal_diagnostic as diag
OUT=diag.OUT
THRESHOLDS={'won':1/3-1e-12,'wave':1.,'tick':600.}


def pair_counts(truth,values,gap=0.,eligible=None):
    pairs=correct=ties=0
    for i in range(len(truth)):
        if eligible is not None and not eligible[i]:continue
        for j in range(i):
            if eligible is not None and not eligible[j]:continue
            delta=truth[i]-truth[j]
            if abs(delta)<=1e-12 or abs(delta)+1e-12<gap:continue
            pairs+=1;prediction=values[i]-values[j]
            if abs(prediction)<=1e-10:ties+=1;correct+=.5
            elif delta*prediction>0:correct+=1
    return dict(pairs=pairs,correct=correct,value_ties=ties,accuracy=correct/pairs if pairs else None)


def bootstrap_counts(groups,metric,field='v4',iterations=2000):
    by_seed=defaultdict(lambda:[0.,0])
    for g in groups:
        counts=g['pair_metrics'][metric][field]
        unit=(g['cell']['level'],g['cell']['seed'])
        by_seed[unit][0]+=counts['correct'];by_seed[unit][1]+=counts['pairs']
    units=list(by_seed.values())
    if not units or not sum(v[1] for v in units):return None
    rng=random.Random(4505);values=[]
    for _ in range(iterations):
        sample=[units[rng.randrange(len(units))] for _ in units]
        n=sum(x[1] for x in sample)
        if n:values.append(sum(x[0] for x in sample)/n)
    values.sort()
    return [values[int((len(values)-1)*q)] for q in (.025,.975)]


def aggregate(groups):
    metrics={}
    for metric in THRESHOLDS:
        result={}
        for field in ('v4','v4_strict','v4_nonterminal','teacher_path_v4'):
            n=sum(g['pair_metrics'][metric][field]['pairs'] for g in groups)
            correct=sum(g['pair_metrics'][metric][field]['correct'] for g in groups)
            ties=sum(g['pair_metrics'][metric][field]['value_ties'] for g in groups)
            result[field]=dict(pairs=n,accuracy=correct/n if n else None,value_ties=ties,
                              cluster_bootstrap_95=bootstrap_counts(groups,metric,field))
        metrics[metric]=result
    return dict(roots=len(groups),episodes=len({g['cell']['id'] for g in groups}),
        base_seed_units=len({(g['cell']['level'],g['cell']['seed']) for g in groups}),
        candidates=sum(len(g['candidates']) for g in groups),labels=sum(g['labels'] for g in groups),
        terminal_wins=sum(g['terminal_wins'] for g in groups),
        separable_roots={m:sum(max(c['means'][m] for c in g['candidates'])-min(c['means'][m] for c in g['candidates'])>=threshold for g in groups) for m,threshold in THRESHOLDS.items()},
        available_root_types=dict(Counter(str(kind) for g in groups for kind in
             {c['plant_type'] for c in g['candidates'] if c['plant_type'] is not None})),
        v4_favourite_types=dict(Counter(str(g['candidates'][g['v4_best']]['plant_type']) for g in groups)),
        wave_favourite_types=dict(Counter(str(g['candidates'][g['best']['wave']]['plant_type']) for g in groups)),
        metrics=metrics)


def group(job,stage='screen',samples=3,indices=None):
    cell,point=job['cell'],job['point'];folder=OUT/'episodes'/cell['id']/f"root_{point['decision']}"
    root=diag.load(folder/'root.json');plan=diag.check_plan()
    if root['point']!=point or root['cell']!=cell or root['inputs_sha256']!=plan['sha256']:
        raise ValueError('root identity/input mismatch')
    all_actions=root['candidates'];indices=list(range(len(all_actions))) if indices is None else indices
    expected={(i,k) for i in indices for k in range(samples)}
    files=list(folder.glob(f'{stage}_[0-9]*_[0-9]*.json'))
    actual={(int(p.stem.split('_')[1]),int(p.stem.split('_')[2])) for p in files}
    if actual!=expected:raise ValueError(f'incomplete/unexpected candidate futures {cell["id"]} root{point["decision"]}: {len(actual)}/{len(expected)}')
    complete=diag.load(folder/f'{stage}-complete.json')
    if complete!=dict(indices=indices,samples=samples,labels=len(expected),root_restored=True):
        raise ValueError('completion marker does not match exact panel')
    packets={p[0]:p[1] for p in root['root_frame']['packets']};candidates=[];run_rows=[]
    for index in indices:
        rows=[]
        for k in range(samples):
            row=diag.load(folder/f'{stage}_{index}_{k}.json')
            identity=dict(cell=cell['id'],root=point,index=index,action=all_actions[index],
                future=diag.future_seed(cell['id'],point['decision'],k,stage),
                search_seed_base=diag.future_seed(cell['id'],point['decision'],k,stage+'-teacher-search'),
                stage=stage,k=k,inputs_sha256=plan['sha256'])
            diag.validate_label(row,identity,folder/f'{stage}_{index}_{k}.jsonl.gz')
            if row['outcome']['tick']<point['tick'] or not 0<=row['outcome']['mowers_used']<=6:
                raise ValueError('invalid terminal tick or mowers')
            if not all(math.isfinite(row[key]) and 0<=row[key]<=1 for key in ['v4_short_value']):
                raise ValueError('invalid v4 probability')
            rows.append(row)
        action=all_actions[index]
        candidates.append(dict(index=index,action=action,plant_type=packets[action['packet']] if action['type']=='plant' else None,
            means={m:statistics.mean(r['outcome'][m] for r in rows) for m in ('won','wave','tick','mowers_used')},
            v4=statistics.mean(r['v4_short_value'] for r in rows),
            teacher_path_v4=statistics.mean(r['teacher_leaf']['value'] for r in rows),
            short_all_nonterminal=all(not r['short_leaf']['terminal'] for r in rows),
            per_future=[{k:r['outcome'][k] for k in ('won','wave','tick','mowers_used')} for r in rows],
            cpu_seconds=sum(r['cpu_seconds'] for r in rows)))
        run_rows.extend(rows)
    student=root['student_action'];student_index=0 if student['type']=='wait' else all_actions.index(student)
    candidate_positions={c['index']:i for i,c in enumerate(candidates)}
    best={m:max(range(len(candidates)),key=lambda i:candidates[i]['means'][m]) for m in THRESHOLDS}
    v4_best=max(range(len(candidates)),key=lambda i:candidates[i]['v4'])
    metrics={}
    for metric,threshold in THRESHOLDS.items():
        truth=[c['means'][metric] for c in candidates];v4=[c['v4'] for c in candidates]
        metrics[metric]=dict(v4=pair_counts(truth,v4,threshold),v4_strict=pair_counts(truth,v4),
             v4_nonterminal=pair_counts(truth,v4,threshold,[c['short_all_nonterminal'] for c in candidates]),
             teacher_path_v4=pair_counts(truth,[c['teacher_path_v4'] for c in candidates],threshold))
    return dict(cell=cell,point=point,stage=stage,samples=samples,labels=len(run_rows),candidates=candidates,
         terminal_wins=sum(r['outcome']['won'] for r in run_rows),student_index=student_index,
         student_position=candidate_positions[student_index],best=best,v4_best=v4_best,pair_metrics=metrics)


def screen():
    frozen=diag.load(OUT/'selection-active.json')
    sampling=diag.load(ROOT/'MEMORY/docs/task45-fog-terminal-sampling.json')
    if diag.sha(OUT/'selection-active.json')!=sampling['root_selection_sha256']:raise ValueError('changed selection')
    groups=[group(job) for job in frozen['jobs']]
    report=dict(status='complete',sampling=sampling,overall=aggregate(groups),
                by_level_observation={},groups=groups)
    for level in (31,32,33,34):
        for mode in ('human','xray'):
            subset=[g for g in groups if g['cell']['level']==level and g['cell']['observation']==mode]
            report['by_level_observation'][f'l{level}-{mode}']=aggregate(subset)
    diag.save(OUT/'screen-report.json',report)
    return report


def confirmation_plan():
    report=screen();jobs=[];selection=[]
    for level in (31,32,33,34):
        for mode in ('human','xray'):
            eligible=[g for g in report['groups'] if g['cell']['level']==level and g['cell']['observation']==mode]
            def improvement(g):
                return g['candidates'][g['best']['wave']]['means']['wave']-g['candidates'][g['student_position']]['means']['wave']
            g=min(eligible,key=lambda g:(-improvement(g),g['cell']['seed'],g['point']['decision']))
            indices=sorted({0,g['student_index'],g['candidates'][g['v4_best']]['index']}|
                           {g['candidates'][g['best'][m]]['index'] for m in THRESHOLDS})
            jobs.append(dict(cell=g['cell'],point=g['point'],stage='confirm',samples=16,candidate_indices=indices))
            selection.append(dict(cell=g['cell']['id'],point=g['point'],indices=indices,
                 screening_wave_gain=improvement(g)))
    labels=sum(len(j['candidate_indices'])*16 for j in jobs)
    times=[c['cpu_seconds']/3 for g in report['groups'] for c in g['candidates']]
    forecast=labels*statistics.mean(times)
    budget_available=71400-diag.spent_cpu()
    value=dict(jobs=jobs,selection=selection,labels=labels,forecast_cpu_seconds=forecast,
               cpu_seconds_before=diag.spent_cpu(),budget_available=budget_available,
               approved_by_predeclared_budget=forecast*1.5+480<budget_available,
               screening_report_sha256=diag.sha(OUT/'screen-report.json'))
    diag.save(OUT/'confirmation-plan.json',value)
    return value


def sign_p(differences):
    nonzero=[d for d in differences if abs(d)>1e-12]
    if not nonzero:return 1.
    n=len(nonzero);positive=sum(d>0 for d in nonzero)
    return sum(math.comb(n,k) for k in range(positive,n+1))/2**n


def holm(tests):
    order=sorted(range(len(tests)),key=lambda i:tests[i]['p']);highest=0.
    for rank,index in enumerate(order):
        highest=max(highest,min(1.,tests[index]['p']*(len(tests)-rank)))
        tests[index]['p_holm']=highest
    return tests


def confirmation_report():
    plan=diag.load(OUT/'confirmation-plan.json')
    if not plan['approved_by_predeclared_budget']:return dict(status='not_run_budget_gate',plan=plan)
    groups=[group(j,'confirm',16,j['candidate_indices']) for j in plan['jobs']]
    tests=[]
    for g in groups:
        student=g['candidates'][g['student_position']]
        for c in g['candidates']:
            if c['index']==student['index']:continue
            for metric in THRESHOLDS:
                diff=[float(a[metric])-float(b[metric]) for a,b in zip(c['per_future'],student['per_future'])]
                tests.append(dict(cell=g['cell']['id'],root=g['point']['decision'],candidate=c['index'],
                    action=c['action'],plant_type=c['plant_type'],metric=metric,mean_gain=statistics.mean(diff),
                    paired_differences=diff,positive=sum(d>0 for d in diff),negative=sum(d<0 for d in diff),p=sign_p(diff)))
    holm(tests)
    return dict(status='complete',groups=groups,tests=tests,
        significant_better=[t for t in tests if t['mean_gain']>0 and t['p_holm']<=.05],
        significance='one-sided paired sign test; zeros omitted; Holm across all candidates/metrics/eight selected roots',
        seed_independence='fresh 16 futures disjoint from 3 screening futures; selected roots/candidates remain a diagnostic subset')


def final_report():
    screening=screen();confirmation=confirmation_report()
    results=dict(schema_version=1,status='complete',frozen_inputs=diag.check_plan(),
        sampling=screening['sampling'],overall=screening['overall'],by_level_observation=screening['by_level_observation'],
        confirmation=confirmation,
        cpu_seconds_metered=diag.spent_cpu(),cpu_seconds_setup_test_reporting_allowance=480,
        artifact_bytes=diag.artifact_bytes(),raw_screen_report_sha256=diag.sha(OUT/'screen-report.json'),
        ranking_note='Reported agreements compare noisy 3-future candidate means; strict/thresholded/true-nonterminal leaf pairs separated. Cluster bootstrap samples paired base seeds, never individual candidate pairs.',
        raw_data='artifacts/task45/phase-e/episodes/<level-observation-seed>/root_<decision>/{root.json,screen|confirm_<candidate>_<future>.json,.jsonl.gz}',
        policy_note='Original root trajectories are student decisions. All branch terminal labels execute candidate immediately then the pinned frozen search teacher; no v4 terminal cutoff and no 2500-tick student prefix on terminal path.')
    if results['cpu_seconds_metered']+480>72000:raise RuntimeError('20 CPU-hour budget exceeded')
    compact_confirmation={k:v for k,v in confirmation.items() if k!='groups'}
    results['confirmation']=compact_confirmation
    results['screen_root_summaries']=[dict(cell=g['cell'],point=g['point'],labels=g['labels'],
        student=g['candidates'][g['student_position']],v4_favourite=g['candidates'][g['v4_best']],
        best={m:g['candidates'][g['best'][m]] for m in THRESHOLDS},pair_metrics=g['pair_metrics']) for g in screening['groups']]
    diag.save(ROOT/'MEMORY/docs/task45-fog-terminal-results.json',results)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['screen','plan-confirm','confirm','report']);a=p.parse_args()
    if a.mode=='screen':r=screen();print(json.dumps(r['overall'],indent=2))
    elif a.mode=='plan-confirm':print(json.dumps({k:v for k,v in confirmation_plan().items() if k!='jobs'},indent=2))
    elif a.mode=='confirm':
        plan=diag.load(OUT/'confirmation-plan.json')
        if not plan['approved_by_predeclared_budget']:raise RuntimeError('confirmation exceeds remaining budget')
        diag.run_jobs('confirm',plan['jobs'])
    else:r=final_report();print(json.dumps({k:v for k,v in r.items() if k in ('status','cpu_seconds_metered','artifact_bytes')},indent=2))

if __name__=='__main__':main()
