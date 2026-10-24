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
        stable_job={k:v for k,v in job.items() if k not in ('stage','samples','indices','inspect_only')}
        if root.get('job')!=stable_job:
            raise ValueError('root header differs from frozen job')
        if 'meta_sha256' in job:
            from pvz_frame_store import frame_path
            archive=Path(job['archive'])
            if f.sha(frame_path(archive))!=job['frames_sha256'] or f.sha(archive/'meta.json')!=job['meta_sha256']:
                raise ValueError('archive input changed')
        actions=root['candidates'];indices=job.get('indices',list(range(len(actions))))
        if 'initial_mowers' in root:
            initial_mowers=root['initial_mowers']
        else:
            from pvz_frame_store import frame_path,read_frames
            initial_mowers=sum(s==1 for _,s in read_frames(frame_path(Path(job['archive'])))[0]['mowers'])
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
                if row['outcome']['mowers_used']>initial_mowers:
                    raise ValueError('mowers exceed actual reset defenses')
                rows.append(f.validate(row,f.identity(job,actions[i],i,stage,k),path.with_suffix('.jsonl.gz')))
            action=actions[i];packets={p[0]:p[1] for p in root['frame']['packets']}
            candidates.append(dict(index=i,action=action,plant_type=packets[action['packet']] if action['type']=='plant' else None,
                means={m:statistics.mean(float(r['outcome'][m]) for r in rows) for m in METRICS},
                v4=statistics.mean(r['v4_short'] for r in rows),
                teacher_path_v4=statistics.mean(r['teacher_leaf']['value'] for r in rows) if all('teacher_leaf' in r for r in rows) else None,
                short_all_nonterminal=all(not r['short_leaf']['terminal'] for r in rows),
                outcomes=[r['outcome'] for r in rows],cpu_seconds=sum(r['cpu_seconds'] for r in rows)))
        groups.append(dict(job=job,student_index=student,candidates=candidates,labels=len(expected),samples=samples))
    return groups


def rank_counts(group,metric,gap,nonterminal=False,field='v4'):
    n=correct=ties=0
    candidates=group['candidates']
    for i,a in enumerate(candidates):
        for b in candidates[:i]:
            if metric=='tick' and (a['means']['won']>0 or b['means']['won']>0):
                continue  # Survival ordering is meaningful here only for all-loss candidates.
            delta=a['means'][metric]-b['means'][metric]
            if metric=='mowers_used':delta=-delta
            if abs(delta)<gap-1e-12 or abs(delta)<1e-12:
                continue
            if nonterminal and not(a['short_all_nonterminal'] and b['short_all_nonterminal']):
                continue
            if a.get(field) is None or b.get(field) is None:
                continue
            prediction=a[field]-b[field];n+=1
            ties += abs(prediction)<1e-10
            correct+=.5 if abs(prediction)<1e-10 else int(delta*prediction>0)
    return dict(pairs=n,correct=correct,value_ties=ties,accuracy=correct/n if n else None)


def agreement(groups,metric,gap,nonterminal=False,field='v4'):
    counts=defaultdict(lambda:[0,0.])
    for g in groups:
        row=rank_counts(g,metric,gap,nonterminal,field)
        job=g['job']
        # Paired human/xray arms and every root from one reset stay in one cluster.
        unit=(job['reset']['level'],job['reset']['seed']) if 'reset' in job else job['archive']
        counts[unit][0]+=row['pairs'];counts[unit][1]+=row['correct']
    units=list(counts.values());n=sum(v[0] for v in units);correct=sum(v[1] for v in units)
    ci=None
    if sum(v[0]>0 for v in units)>=2:
        rng=random.Random(53);values=[]
        for _ in range(2000):
            sample=[rng.choice(units) for _ in units];den=sum(v[0] for v in sample)
            if den:values.append(sum(v[1] for v in sample)/den)
        values.sort();ci=[values[int((len(values)-1)*q)] for q in (.025,.975)]
    return dict(pairs=n,accuracy=correct/n if n else None,game_cluster_bootstrap_95=ci)


def screen_summary(plan,jobs):
    groups=panel(plan,jobs);result={}
    for terrain in sorted({g['job']['terrain'] for g in groups}):
        subset=[g for g in groups if g['job']['terrain']==terrain]
        gaps=plan['protocol']['confirmation']['mean_gap']
        result[terrain]=dict(games=len({g['job']['archive'] for g in subset}),roots=len(subset),
            labels=sum(g['labels'] for g in subset),wins=sum(o['won'] for g in subset for c in g['candidates'] for o in c['outcomes']),
            v4_agreement={m:dict(all_pairs=agreement(subset,m,gaps[m]),
                                strict_pairs=agreement(subset,m,0.),
                                teacher_path_v4=agreement(subset,m,gaps[m],field='teacher_path_v4'),
                                nonterminal_v4_only=agreement(subset,m,gaps[m],True)) for m in METRICS})
    f.save(Path(plan['out'])/'screen-summary.json',dict(by_terrain=result,groups=groups))
    return groups,result


def attribution(group, trace, gaps):
    """First production stage blocking each observed terminal improver.

    Static coverage is overlapping; the final causes are mutually exclusive. Shovels
    remain engine-legal evidence outside the student's production action space.
    """
    static = {f.key(a) for a in trace.get('static_selected', [])}
    generated = {f.key(a):i for i,a in enumerate(trace.get('candidates', []))}
    reference = next(c for c in group['candidates'] if c['index']==group['student_index'])
    stages = {}; details = []
    for metric, threshold in gaps.items():
        counts = Counter(); examples = []
        for candidate in group['candidates']:
            if metric=='tick' and (candidate['means']['won'] or reference['means']['won']):
                continue
            gain = (candidate['means'][metric]-reference['means'][metric]) * (-1 if metric=='mowers_used' else 1)
            if gain+1e-12<threshold:
                continue
            action=candidate['action']; key=f.key(action); index=generated.get(key)
            counts['observed_improvers']+=1
            counts['improving_plant_actions']+=action['type']=='plant'
            counts['production_representable_improvers']+=action['type']!='shovel'
            counts['static_retained']+=key in static
            if not trace.get('searched'):
                cause='search_not_scheduled'
            elif index is None:
                cause='outside_production_action_space' if action['type']=='shovel' else 'candidate_generation_omission'
            elif trace['halving_counts'][index]<trace['required_halving_samples']:
                cause='halving_pruned'
            elif index!=trace['best_index']:
                cause='leaf_ranking_not_top'
            else:
                best, anchor = trace['best_index'], trace['student_index']
                before, after = trace['pre_verification_values'], trace['final_values']
                if before[best]-before[anchor]<=trace['margin']:
                    cause='initial_margin_blocked'
                elif after[best]-after[anchor]<=trace['margin']:
                    cause='verification_rejected'
                elif f.key(trace['chosen'])==key:
                    cause='executed'
                else:
                    raise ValueError('unclassified production decision')
            counts[cause]+=1
            examples.append(dict(candidate=candidate['index'],action=action,mean_gain=gain,cause=cause,
                                 static_retained=key in static))
        stages[metric]=dict(counts); details.append(dict(metric=metric,improvers=examples))
    return dict(stages=stages,details=details,
        note='Observed common-future means versus student; statistical confirmation uses independent futures.')


def diagnostic_report(plan, jobs):
    groups, terrains=screen_summary(plan,jobs); gaps=plan['protocol']['confirmation']['mean_gap']
    by_level_observation={}; totals={m:Counter() for m in METRICS}
    for group in groups:
        trace=f.load(Path(plan['out'])/'roots'/group['job']['id']/'search.json')
        if trace.get('audit_schema')!=2:
            raise ValueError('rerun inspect for complete candidate attribution')
        group['attribution']=attribution(group,trace,gaps)
        for metric,counts in group['attribution']['stages'].items():
            totals[metric].update(counts)
    keys=sorted({(g['job']['reset']['level'],g['job'].get('observation',g['job']['reset'].get('fog_observation','human'))) for g in groups})
    for level,mode in keys:
        subset=[g for g in groups if g['job']['reset']['level']==level and
                g['job'].get('observation',g['job']['reset'].get('fog_observation','human'))==mode]
        by_level_observation[f'l{level}-{mode}']=dict(**root_counts(subset,gaps),
            metrics={m:dict(thresholded=agreement(subset,m,gaps[m]),strict=agreement(subset,m,0.),
                nonterminal_v4_only=agreement(subset,m,gaps[m],True),
                teacher_path_v4=agreement(subset,m,gaps[m],field='teacher_path_v4')) for m in METRICS})
    value=dict(status='complete',overall=root_counts(groups,gaps),by_terrain=terrains,by_level_observation=by_level_observation,
        attribution_totals={m:dict(c) for m,c in totals.items()},groups=groups,
        observation_limit='human/xray paired reset seeds follow their own trajectories; no same-state causal claim',
        cpu_seconds=f.spent_cpu(plan['out']),artifact_bytes=f.artifact_bytes(plan['out']))
    f.save(Path(plan['out'])/'diagnostic-report.json',value)
    return value


def root_counts(groups, gaps):
    """Keep legal coverage and branch label wins separate from independent episodes."""
    return dict(roots=len(groups),base_seed_units=len({(g['job']['reset']['level'],g['job']['reset']['seed']) for g in groups}),
        labels=sum(g['labels'] for g in groups),candidates=sum(len(g['candidates']) for g in groups),
        terminal_wins=sum(o['won'] for g in groups for c in g['candidates'] for o in c['outcomes']),
        separable_roots={m:sum(max(c['means'][m] for c in g['candidates'])-min(c['means'][m] for c in g['candidates'])>=gap for g in groups) for m,gap in gaps.items()},
        available_root_types=dict(Counter(str(t) for g in groups for t in {c['plant_type'] for c in g['candidates'] if c['plant_type'] is not None})),
        v4_favourite_types=dict(Counter(str(max(g['candidates'],key=lambda c:c['v4'])['plant_type']) for g in groups)))


def confirmation_plan(plan,jobs):
    groups,_=screen_summary(plan,jobs);selected=[];unrepresented=[]
    for g in groups:
        candidates=g['candidates']
        if plan['protocol']['confirmation'].get('selection')=='metric_bests':
            # Diagnostic panels retain the best of each separate terminal metric,
            # the v4 favourite and wait, including engine-only shovel evidence.
            indices={g['student_index'],0,max(candidates,key=lambda c:c['v4'])['index']}
            for metric in METRICS:
                sign=-1 if metric=='mowers_used' else 1
                indices.add(max(candidates,key=lambda c:sign*c['means'][metric])['index'])
            selected.append(dict(g['job'],stage='confirm',indices=sorted(indices),samples=plan['confirm_futures']))
            continue
        rank=lambda c:(c['means']['won'],c['means']['wave'],c['means']['tick'],-c['means']['mowers_used'],-c['index'])
        global_best=max(candidates,key=rank)
        if global_best['action']['type'] not in ('plant','wait'):
            unrepresented.append(dict(root=g['job']['id'],candidate=global_best['index'],action=global_best['action']))
        best=max([c for c in candidates if c['action']['type'] in ('plant','wait')],key=rank)
        if best['index']==g['student_index']:
            continue
        student=next(c for c in candidates if c['index']==g['student_index'])
        gains={m:(-1 if m=='mowers_used' else 1)*(best['means'][m]-student['means'][m]) for m in METRICS}
        gaps=plan['protocol']['confirmation']['mean_gap']
        if not any(gains[m]>=gaps[m] for m in METRICS
                   if not(m=='tick' and best['means']['won']==student['means']['won']==1)):
            continue
        selected.append(dict(g['job'],stage='confirm',indices=sorted({best['index'],g['student_index']}),
                             samples=plan['confirm_futures']))
    value=dict(jobs=selected,labels=sum(len(j['indices'])*j['samples'] for j in selected),
               selection='one lexicographic terminal win/wave/tick/mower challenger among student-representable plant/wait actions per root; screen data only',
               unrepresented_global_bests=unrepresented,
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
        for candidate in g['candidates']:
            if candidate['index']==g['student_index']:continue
            for m in METRICS:
                sign=-1 if m=='mowers_used' else 1
                diff=[sign*(float(a[m])-float(b[m])) if m!='tick' or not(a['won'] or b['won']) else 0.
                      for a,b in zip(candidate['outcomes'],student['outcomes'])]
                tests.append(dict(root=g['job']['id'],candidate=candidate['index'],metric=m,
                                  differences=diff,p=sign_p(diff),**bootstrap_difference(diff)))
    highest=0.
    for rank,t in enumerate(sorted(tests,key=lambda t:t['p'])):
        highest=max(highest,min(1.,t['p']*(len(tests)-rank)));t['p_holm']=highest
    cfg=plan['protocol']['confirmation'];corrections=[]
    for g in groups:
        for c in g['candidates']:
            if c['index']==g['student_index']:continue
            related=[t for t in tests if t['root']==g['job']['id'] and t['candidate']==c['index']]
            good=[t for t in related if t['p_holm']<=cfg['alpha'] and t['mean_gain']>=cfg['mean_gap'][t['metric']]]
            # Keep terminal dimensions separate; never trade a material regression for a scalar reward.
            if (good and next(t['mean_gain'] for t in related if t['metric']=='won')>=0
                    and all(t['mean_gain']>=-cfg['mean_gap'][t['metric']] for t in related)):
                evidence=min(good,key=lambda t:t['p_holm'])
                corrections.append(dict(root=g['job']['id'],job=g['job'],action=c['action'],plant_type=c['plant_type'],
                    confidence=evidence['confidence'],evidence=evidence,student_representable=c['action']['type'] in ('plant','wait')))
    value=dict(status='complete',root_count=len(groups),labels=sum(g['labels'] for g in groups),tests=tests,
               corrections=corrections,significance='independent futures; paired one-sided sign test; Holm across all roots and four metrics',
               confidence_note='paired bootstrap probability(mean oriented gain > 0); mowers are oriented as fewer is better',
               tick_definition='extra survival on paired losses only; winning completion times are descriptive and never rewarded for being longer',
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
                    if metric=='tick' and (outcomes[i]['won'] or outcomes[j]['won']):continue
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
    # A supervised root needs one label. Diagnostic confirmation may retain several
    # challengers; use its strongest independently confirmed evidence deterministically.
    corrections={}
    for correction in sorted(confirmation['corrections'],key=lambda c:(c['evidence']['p_holm'],-c['confidence'])):
        if correction['student_representable']:
            corrections.setdefault(correction['root'],correction)
    for correction in corrections.values():
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
    p.add_argument('stage',choices=['screen','diagnose','plan-confirm','report-confirm','export']);p.add_argument('--jobs',type=Path)
    a=p.parse_args();plan=f.check_plan(a.plan);jobs=f.load(a.jobs)['jobs'] if a.jobs else plan['jobs']
    if a.stage=='screen':_,value=screen_summary(plan,jobs)
    elif a.stage=='diagnose':value=diagnostic_report(plan,jobs)
    elif a.stage=='plan-confirm':value=confirmation_plan(plan,jobs)
    elif a.stage=='report-confirm':value=confirm_summary(plan)
    else:value=export(plan,jobs)
    print(json.dumps({k:v for k,v in value.items() if k not in ('tests','corrections','value_files')},ensure_ascii=False,indent=1))


if __name__=='__main__':main()
