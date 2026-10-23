"""Audit and export the common terminal-fork schema for students and task43 value training."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
import json
import gzip
import math
from pathlib import Path
import random
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import critical_forks as f
METRICS=('won','wave','tick','mowers_used')


def panel(plan,jobs,stage='screen'):
    """Require exact coverage, fixed inputs, real terminals and every teacher trace."""
    f.PLAN=plan;groups=[]
    for job in jobs:
        folder=Path(plan['out'])/'roots'/job['id'];root=f.load(folder/'root.json')
        actions=root['candidates'];indices=job.get('indices',list(range(len(actions))))
        samples=job.get('samples',plan['screen_futures'] if stage=='screen' else plan['confirm_futures'])
        expected={(i,k) for i in indices for k in range(samples)}
        actual={(int(p.stem.split('_')[1]),int(p.stem.split('_')[2])) for p in folder.glob(f'{stage}_[0-9]*.json')}
        if actual!=expected or f.load(folder/f'{stage}-complete.json')!=dict(indices=indices,samples=samples,labels=len(expected)):
            raise ValueError('incomplete/unexpected root futures: '+job['id'])
        student=actions.index(root['student_action']);candidates=[]
        for i in indices:
            rows=[]
            for k in range(samples):
                path=folder/f'{stage}_{i}_{k}.json'
                row=f.load(path)
                required={'root','income','history','legal','terminal','strict_actions'}
                if set(row['checks'])!=required or any(v is not True for v in row['checks'].values()):
                    raise ValueError('missing or failed exact root/history checks')
                if not math.isfinite(row['v4_short']) or not 0<=row['v4_short']<=1:
                    raise ValueError('invalid v4 probability')
                if row['outcome']['tick']<root['frame']['tick']:
                    raise ValueError('terminal precedes root')
                rows.append(f.validate(row,f.identity(job,actions[i],i,stage,k),path.with_suffix('.jsonl.gz')))
            action=actions[i];packets={p[0]:p[1] for p in root['frame']['packets']}
            candidates.append(dict(index=i,action=action,plant_type=packets[action['packet']] if action['type']=='plant' else None,
                means={m:statistics.mean(float(r['outcome'][m]) for r in rows) for m in METRICS},
                v4=statistics.mean(r['v4_short'] for r in rows),
                short_all_nonterminal=all(not r['short_leaf']['terminal'] for r in rows),
                outcomes=[r['outcome'] for r in rows],cpu_seconds=sum(r['cpu_seconds'] for r in rows)))
        groups.append(dict(job=job,student_index=student,candidates=candidates,labels=len(expected),samples=samples))
    return groups


def rank_counts(group,metric,gap,nonterminal=False):
    n=correct=0
    candidates=group['candidates']
    for i,a in enumerate(candidates):
        for b in candidates[:i]:
            delta=a['means'][metric]-b['means'][metric]
            if metric=='mowers_used':delta=-delta
            if abs(delta)<gap-1e-12 or abs(delta)<1e-12:
                continue
            if nonterminal and not(a['short_all_nonterminal'] and b['short_all_nonterminal']):
                continue
            prediction=a['v4']-b['v4'];n+=1
            correct+=.5 if abs(prediction)<1e-10 else int(delta*prediction>0)
    return dict(pairs=n,correct=correct,accuracy=correct/n if n else None)


def agreement(groups,metric,gap,nonterminal=False):
    counts=defaultdict(lambda:[0,0.])
    for g in groups:
        row=rank_counts(g,metric,gap,nonterminal)
        counts[g['job']['archive']][0]+=row['pairs'];counts[g['job']['archive']][1]+=row['correct']
    units=list(counts.values());n=sum(v[0] for v in units);correct=sum(v[1] for v in units)
    ci=None
    if n:
        rng=random.Random(53);values=[]
        for _ in range(2000):
            sample=[rng.choice(units) for _ in units];den=sum(v[0] for v in sample)
            if den:values.append(sum(v[1] for v in sample)/den)
        values.sort();ci=[values[int((len(values)-1)*q)] for q in (.025,.975)]
    return dict(pairs=n,accuracy=correct/n if n else None,game_cluster_bootstrap_95=ci)


def screen_summary(plan,jobs):
    groups=panel(plan,jobs);result={}
    for terrain in ('day','night','pool'):
        subset=[g for g in groups if g['job']['terrain']==terrain]
        gaps=plan['protocol']['confirmation']['mean_gap']
        result[terrain]=dict(games=len({g['job']['archive'] for g in subset}),roots=len(subset),
            labels=sum(g['labels'] for g in subset),wins=sum(o['won'] for g in subset for c in g['candidates'] for o in c['outcomes']),
            v4_agreement={m:dict(all_pairs=agreement(subset,m,gaps[m]),
                                nonterminal_v4_only=agreement(subset,m,gaps[m],True)) for m in METRICS})
    f.save(Path(plan['out'])/'screen-summary.json',dict(by_terrain=result,groups=groups))
    return groups,result


def confirmation_plan(plan,jobs):
    groups,_=screen_summary(plan,jobs);selected=[]
    for g in groups:
        candidates=g['candidates']
        best=max(candidates,key=lambda c:(c['means']['won'],c['means']['wave'],c['means']['tick'],-c['means']['mowers_used'],-c['index']))
        if best['index']==g['student_index']:
            continue
        student=next(c for c in candidates if c['index']==g['student_index'])
        gains={m:(-1 if m=='mowers_used' else 1)*(best['means'][m]-student['means'][m]) for m in METRICS}
        gaps=plan['protocol']['confirmation']['mean_gap']
        if not any(gains[m]>=gaps[m] for m in METRICS):
            continue
        selected.append(dict(g['job'],stage='confirm',indices=sorted({best['index'],g['student_index']}),
                             samples=plan['confirm_futures']))
    value=dict(jobs=selected,labels=sum(len(j['indices'])*j['samples'] for j in selected),
               selection='one lexicographic terminal win/wave/tick/mower challenger per root; screen data only',
               future_independence='confirm namespace disjoint from screen; common world/search seeds within each root')
    f.save(Path(plan['out'])/'confirmation.json',value)
    return value


def sign_p(differences):
    nz=[d for d in differences if abs(d)>1e-12];n=len(nz);positive=sum(d>0 for d in nz)
    return sum(math.comb(n,k) for k in range(positive,n+1))/2**n if n else 1.


def bootstrap_difference(differences):
    rng=random.Random(53);n=len(differences)
    values=sorted(statistics.mean(rng.choices(differences,k=n)) for _ in range(5000))
    return dict(mean_gain=statistics.mean(differences),confidence=sum(v>0 for v in values)/len(values),
                mean_gain_95=[values[int((len(values)-1)*q)] for q in (.025,.975)])


def confirm_summary(plan):
    selection=f.load(Path(plan['out'])/'confirmation.json');groups=panel(plan,selection['jobs'],'confirm');tests=[]
    for g in groups:
        student=next(c for c in g['candidates'] if c['index']==g['student_index'])
        candidate=next(c for c in g['candidates'] if c['index']!=g['student_index'])
        for m in METRICS:
            sign=-1 if m=='mowers_used' else 1
            diff=[sign*(float(a[m])-float(b[m])) for a,b in zip(candidate['outcomes'],student['outcomes'])]
            tests.append(dict(root=g['job']['id'],candidate=candidate['index'],metric=m,
                              differences=diff,p=sign_p(diff),**bootstrap_difference(diff)))
    highest=0.
    for rank,t in enumerate(sorted(tests,key=lambda t:t['p'])):
        highest=max(highest,min(1.,t['p']*(len(tests)-rank)));t['p_holm']=highest
    cfg=plan['protocol']['confirmation'];corrections=[]
    for g in groups:
        related=[t for t in tests if t['root']==g['job']['id']]
        good=[t for t in related if t['p_holm']<=cfg['alpha'] and t['mean_gain']>=cfg['mean_gap'][t['metric']]]
        # Keep terminal dimensions separate; do not trade a material regression for a scalar reward.
        if good and all(t['mean_gain']>=-cfg['mean_gap'][t['metric']] for t in related):
            evidence=min(good,key=lambda t:t['p_holm'])
            c=next(c for c in g['candidates'] if c['index']!=g['student_index'])
            corrections.append(dict(root=g['job']['id'],job=g['job'],action=c['action'],plant_type=c['plant_type'],
                confidence=evidence['confidence'],evidence=evidence,student_representable=c['action']['type'] in ('plant','wait')))
    value=dict(status='complete',root_count=len(groups),labels=sum(g['labels'] for g in groups),tests=tests,
               corrections=corrections,significance='independent futures; paired one-sided sign test; Holm across all roots and four metrics',
               confidence_note='paired bootstrap probability(mean oriented gain > 0); mowers are oriented as fewer is better',
               selection_budget=selection)
    f.save(Path(plan['out'])/'confirmation-results.json',value)
    return value


def terminal_pairs(outcomes,candidates,kinds,positions,gaps):
    """Separate terminal criteria and observation boundaries; never combine into a reward."""
    by_future=defaultdict(list)
    for index,(kind,position) in enumerate(zip(kinds,positions)):
        by_future[(kind,position)].append(index)
    for (kind,position),indices in by_future.items():
        for ordinal,i in enumerate(indices):
            for j in indices[:ordinal]:
                if candidates[i]==candidates[j]:continue
                for metric in METRICS:
                    delta=float(outcomes[i][metric])-float(outcomes[j][metric])
                    if metric=='mowers_used':delta=-delta
                    if abs(delta)<=1e-12 or abs(delta)<gaps[metric]:continue
                    winner,loser=(i,j) if delta>0 else (j,i)
                    yield dict(kind=kind,position=position,metric=metric,winner_row=winner,loser_row=loser,gap=abs(delta))


def export(plan,jobs):
    """Archives use existing label weights; value shards use task43 decision-data keys."""
    import numpy as np
    import torch
    from pvz_frame_store import frame_path,read_frames,write_frames
    from pvz_value_net import encode_frame
    out=Path(plan['out']);groups,_=screen_summary(plan,jobs);confirmation=confirm_summary(plan)
    registration=f.load(out/'training-registration.json');data=out/'training/frames';labels=set();weights={};evidence=[]
    for record in registration['replay']:
        original=Path(record['archive']);source=frame_path(original)
        if f.sha(source)!=record['frames_sha256']:raise ValueError('replay corpus changed')
        target=data/f"seed_{record['seed']}"/record['label'];target.mkdir(parents=True,exist_ok=True)
        link=target/source.name
        if not link.exists():link.symlink_to(source)
        meta=f.load(original/'meta.json');meta['training_rows']=record['rows'];f.save(target/'meta.json',meta)
        labels.add(record['label'])
    for correction in confirmation['corrections']:
        if not correction['student_representable']:continue
        job=correction['job'];source=Path(job['archive']);frames=read_frames(frame_path(source))
        # The executed action stays true; only its supervised target changes.
        i=job['decision'];frames[i].pop('targets',None);frames[i].pop('label',None)
        frames[i]['label']=correction['action']
        label='fork_'+job['id'];target=data/f"seed_{job['reset']['seed']}"/label;target.mkdir(parents=True,exist_ok=True)
        write_frames(target/'frames.npz',frames)
        meta=f.load(source/'meta.json');meta.update(training_rows=[i],critical_fork=correction)
        f.save(target/'meta.json',meta);labels.add(label);weights[label]=correction['confidence']
        evidence.append(dict(root=job['id'],label=label,row=i,confidence=correction['confidence']))
    recipe=deepcopy(registration['recipe']);recipe.update(labels=sorted(labels),label_weight=weights)
    f.save(out/'training/recipe.json',recipe)
    value_files=[];pair_counts=Counter();pair_path=out/'value-data/pairs.jsonl.gz'
    pair_path.parent.mkdir(parents=True,exist_ok=True)
    pair_stream=gzip.open(pair_path,'wt',compresslevel=1)
    for g in groups:
        job=g['job'];folder=out/'roots'/job['id'];root=f.load(folder/'root.json')
        boards=[];scalars=[];legacy=[];won=[];positions=[];candidates=[];packets=[];state_kinds=[];kinds=[];outcomes=[]
        for c in g['candidates']:
            for k in range(g['samples']):
                row=f.load(folder/f"screen_{c['index']}_{k}.json")
                states=[(row['teacher_leaf']['frame'],0,'teacher_horizon_leaf'),
                        (row['post_action_frame'],1,'post_candidate_teacher_start')]
                for frame,kind,state_kind in states:
                    if frame['terminal']:continue
                    b,s=encode_frame(frame,2);_,s1=encode_frame(frame,1)
                    context=torch.tensor(root['context'],dtype=torch.float32)
                    boards.append(b.half());scalars.append(torch.cat([s,context]));legacy.append(torch.cat([s1,context]))
                    won.append(float(row['outcome']['won']));positions.append(job['decision']*1000+k)
                    candidates.append(c['index']);packets.append(c['action'].get('packet',-1));state_kinds.append(state_kind);kinds.append(kind)
                    outcomes.append(row['outcome'])
        if boards:
            n=len(boards);path=out/'value-data'/f"{job['id']}.pt";path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(dict(boards=torch.stack(boards),scalars=torch.stack(legacy),scalars_v2=torch.stack(scalars),
                won=torch.tensor(won),seed=torch.full((n,),job['reset']['seed'],dtype=torch.long),
                position=torch.tensor(positions),candidate=torch.tensor(candidates),packet=torch.tensor(packets),
                kind=torch.tensor(kinds,dtype=torch.long),state_kind=state_kinds,
                wave=torch.tensor([o['wave'] for o in outcomes]),tick=torch.tensor([o['tick'] for o in outcomes]),
                mowers_used=torch.tensor([o['mowers_used'] for o in outcomes]),
                config=dict(task_id=job['task_id'],root=job['id'],continuation_policy='frozen search teacher',samples=g['samples'])),path)
            value_files.append(dict(path=str(path),sha256=f.sha(path),rows=n,state_kinds=dict(Counter(state_kinds))))
            for pair in terminal_pairs(outcomes,candidates,kinds,positions,plan['protocol']['confirmation']['mean_gap']):
                pair_stream.write(json.dumps(dict(shard=path.name,**pair))+'\n');pair_counts[pair['metric']]+=1
        if f.artifact_bytes(out)>plan['artifact_budget_bytes']:raise RuntimeError('5GB limit during export')
    pair_stream.close()
    value=dict(status='complete',student_corrections=evidence,labels=sorted(labels),label_weight=weights,
               recipe=str(out/'training/recipe.json'),value_files=value_files,
               ranking_pairs=dict(path=str(pair_path),sha256=f.sha(pair_path),by_metric=dict(pair_counts),
                                  note='same seed/root/future/kind; metrics kept separate; wave/tick/mower preferences are never substituted for binary won labels'),
               value_contract='task43 Decisions keys; group by kind+seed+position (root decision and common future); kind0 teacher horizon leaf, kind1 immediate candidate cell state; terminal network-bypass states excluded',
               confirmation_sha256=f.sha(out/'confirmation-results.json'))
    f.save(out/'export.json',value)
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('plan',type=Path)
    p.add_argument('stage',choices=['screen','plan-confirm','report-confirm','export']);p.add_argument('--jobs',type=Path)
    a=p.parse_args();plan=f.check_plan(a.plan);jobs=f.load(a.jobs)['jobs'] if a.jobs else plan['jobs']
    if a.stage=='screen':_,value=screen_summary(plan,jobs)
    elif a.stage=='plan-confirm':value=confirmation_plan(plan,jobs)
    elif a.stage=='report-confirm':value=confirm_summary(plan)
    else:value=export(plan,jobs)
    print(json.dumps({k:v for k,v in value.items() if k not in ('tests','corrections','value_files')},ensure_ascii=False,indent=1))


if __name__=='__main__':main()
