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
import terminal_candidate_audit as diag
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
    if len(all_actions)!=point['candidate_count'] or diag.digest_value(all_actions)!=point['candidate_sha256']:
        raise ValueError('root does not cover frozen engine legal actions exactly')
    diag.PLAN=plan
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


def attribution(group):
    folder=OUT/'episodes'/group['cell']['id']/f"root_{group['point']['decision']}"
    trace=diag.load(folder/'search-trace.json')
    if len(trace['static_values'])!=len(trace['static_actions']):raise ValueError('incomplete static scoring trace')
    per_card=defaultdict(list)
    for value,action in sorted(zip(trace['static_values'],trace['static_actions']),key=lambda x:-x[0]):
        per_card[action['packet']].append(action)
    static={diag.jkey(a) for actions in per_card.values() for a in actions[:trace['per_card']]}
    generated={diag.jkey(a):i for i,a in enumerate(trace['candidates'])}
    best=trace['best_index'];anchor=trace['student_index']
    before=trace['before_verification_values'];after=trace['after_verification_values']
    reference=group['candidates'][group['student_position']]
    stages={};details=[]
    for metric,threshold in THRESHOLDS.items():
        counts=Counter();examples=[]
        for candidate in group['candidates']:
            gain=candidate['means'][metric]-reference['means'][metric]
            if gain+1e-12<threshold:continue
            action=candidate['action'];key=diag.jkey(action);index=generated.get(key)
            counts['observed_improvers']+=1
            if action['type']=='plant':counts['improving_plant_actions']+=1
            if action['type']!='shovel':counts['production_representable_improvers']+=1
            if key in static:counts['static_retained']+=1
            if index is None:
                cause='outside_production_action_space' if action['type']=='shovel' else 'candidate_generation_omission'
            elif trace['halving_counts'][index]<trace['required_halving_samples']:
                cause='halving_pruned'
            elif index!=best:
                cause='leaf_ranking_not_top'
            elif before[best]-before[anchor]<=trace['margin']:
                cause='initial_margin_blocked'
            elif after[best]-after[anchor]<=trace['margin']:
                cause='verification_rejected'
            elif diag.jkey(trace['chosen'])==key:
                cause='executed'
            else:raise ValueError('unclassified production decision')
            counts[cause]+=1
            examples.append(dict(candidate=candidate['index'],action=action,mean_gain=gain,cause=cause,
                                 static_retained=key in static))
        stages[metric]=dict(counts)
        details.append(dict(metric=metric,improvers=examples))
    return dict(stages=stages,details=details,formal_trace=trace,
        note='Observed common-future terminal mean improvements versus student; not independent statistical confirmation. Static counts overlap final stage causes.')


def report():
    global OUT
    OUT=diag.OUT
    sampling=diag.load(OUT/'selection-active.json')
    groups=[]
    for job in sampling['jobs']:
        g=group(job,samples=job['samples']);g['attribution']=attribution(g);groups.append(g)
    value=dict(schema_version=1,status='complete',sampling_sha256=diag.sha(OUT/'selection-active.json'),
        overall=aggregate(groups),by_level={},groups=groups,
        cpu_seconds_metered=diag.spent_cpu(),cpu_allowance_seconds=600,
        frozen_inputs=diag.check_plan(),raw_directory=str(OUT/'episodes'),
        inference_limit='Only preregistered root/common-future cohort; report wins/wave/tick separately. Terminal continuation is frozen teacher, short v4 leaf is 2500-tick student.')
    for level in diag.check_plan()['spec']['levels']:
        value['by_level'][str(level)]=aggregate([g for g in groups if g['cell']['level']==level])
    causes={}
    for metric in THRESHOLDS:
        counts=Counter()
        for g in groups:counts.update(g['attribution']['stages'][metric])
        causes[metric]=dict(counts)
    value['attribution_totals']=causes
    diag.save(OUT/'screen-report.json',value)
    compact={k:v for k,v in value.items() if k!='groups'}
    compact['root_summaries']=[dict(cell=g['cell'],point=g['point'],labels=g['labels'],
        student=g['candidates'][g['student_position']],best={m:g['candidates'][g['best'][m]] for m in THRESHOLDS},
        v4_favourite=g['candidates'][g['v4_best']],pair_metrics=g['pair_metrics'],
        attribution=g['attribution']['stages']) for g in groups]
    diag.save(ROOT/'MEMORY/docs/task46-roof-terminal-results.json',compact)
    return value


if __name__=='__main__':
    print(json.dumps(report()['overall'],indent=2))
