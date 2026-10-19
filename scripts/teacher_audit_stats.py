"""Paired-future audit statistics with within-state multiplicity correction."""
import argparse
from collections import Counter
from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


def t_cdf(value, df):
    """Integrate the Student t density (df>=31 here); Simpson error is negligible at audit cutoffs."""
    if value == 0:
        return .5
    sign = 1 if value > 0 else -1
    x = abs(value)
    if x > 100:
        return 1.0 if sign > 0 else 0.0
    n = 2048
    h = x/n
    constant = math.exp(math.lgamma((df+1)/2)-math.lgamma(df/2))/math.sqrt(df*math.pi)
    def density(v):
        return constant*(1+v*v/df)**(-(df+1)/2)
    area = (density(0)+density(x)+4*sum(density(h*i) for i in range(1,n,2))
            +2*sum(density(h*i) for i in range(2,n,2)))*h/3
    return min(1.,max(0.,.5+sign*area))


@lru_cache(None)
def t_critical(df, comparisons=1):
    target = 1-.025/comparisons
    lo,hi = 0.,8.
    for _ in range(45):
        mid=(lo+hi)/2
        if t_cdf(mid,df) < target:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2


def paired(delta, comparisons=1):
    delta=np.asarray(delta,dtype=float)
    mean=float(delta.mean()); se=float(delta.std(ddof=1)/math.sqrt(len(delta)))
    c=t_critical(len(delta)-1,comparisons)
    p=min(1.,2*(1-t_cdf(abs(mean)/se,len(delta)-1))*comparisons) if se else (0. if mean else 1.)
    return {'mean':mean,'se':se,'ci95':[mean-c*se,mean+c*se], 'p_adjusted':p}


def point_stats(row):
    scores=np.asarray(row['scores'],dtype=float)
    assert scores.shape[1]==64 and np.isfinite(scores).all()
    assert scores.min()>=0 and scores.max()<=1
    m=len(scores)-1
    candidates=[paired(score-scores[0],m) for score in scores[1:]]
    best=1+int(np.argmax(scores[1:].mean(axis=1)))
    held=1+int(np.argmax(scores[1:,:32].mean(axis=1)))
    holdout=paired(scores[held,32:]-scores[0,32:])
    recorded=row.get('archived_targets') or []
    action=row['actions'][best]
    triple=(action.get('packet',-1),action.get('row',-1),action.get('col',-1))
    present=any(tuple(t[:3])==triple for t in recorded)
    significant=any(c['ci95'][0]>0 for c in candidates)
    witness=1+max(range(len(candidates)),key=lambda i:candidates[i]['ci95'][0])
    attributed=row['actions'][witness if significant else best]
    triple=(attributed.get('packet',-1),attributed.get('row',-1),attributed.get('col',-1))
    present=any(tuple(t[:3])==triple for t in recorded)
    return {k:row[k] for k in ('category','archive','decision','seed','gap','wave','tick','label')} | {
        'actions':len(scores),'original_mean':float(scores[0].mean()),
        'action_statistics':[{'action':action,'mean':float(score.mean()),
                              'se':float(score.std(ddof=1)/8)}
                             for action,score in zip(row['actions'],scores)],
        'best_action':action,'best':candidates[best-1],
        'witness_action':row['actions'][witness],'witness':candidates[witness-1],
        'significant_inferiority':significant,
        'minimum_adjusted_p':min(c['p_adjusted'] if c['mean']>0 else 1 for c in candidates),
        'holdout_selected_action':row['actions'][held],'holdout':holdout,
        'attribution':('no_recorded_search' if not recorded else
                       'in_archived_candidates' if present else 'absent_from_archived_candidates'),
        'archived_verification':row.get('archived_verification')}


def bootstrap_mean(values):
    a=np.asarray(values);rng=np.random.default_rng(20261019)
    means=a[rng.integers(len(a),size=(10000,len(a)))].mean(axis=1)
    return {'mean':float(a.mean()),'cluster_ci95':np.quantile(means,[.025,.975]).tolist()}


def group(rows):
    n=len(rows);k=sum(r['significant_inferiority'] for r in rows)
    z=1.95996398454;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'states':n,'games':len({r['seed'] for r in rows}), 'significantly_inferior':k,
            'fraction':p,'wilson_ci95':[center-radius,center+radius],
            'bh_fdr05_count':sum(r.get('bh_fdr05',False) for r in rows),
            'by_fdr05_count':sum(r.get('by_fdr05',False) for r in rows),
            'lower_ci_above_formal_margin02':sum(r['witness']['ci95'][0]>.02 for r in rows),
            'holdout_signed_value_loss':bootstrap_mean([r['holdout']['mean'] for r in rows]),
            'all64_best_uplift_optimistic':float(np.mean([r['best']['mean'] for r in rows])),
            'attribution_of_significant':dict(Counter(r['attribution'] for r in rows if r['significant_inferiority'])),
            'mean_actions':float(np.mean([r['actions'] for r in rows]))}


def summarize(directory,out):
    paths=sorted(directory.glob('*.json'))
    rows=[point_stats(json.loads(p.read_text())) for p in paths]
    assert len(rows)==600
    ordered=sorted(rows,key=lambda r:r['minimum_adjusted_p'])
    cutoff=max([i for i,r in enumerate(ordered,1) if r['minimum_adjusted_p']<=.05*i/len(rows)]+[0])
    accepted={id(r) for r in ordered[:cutoff]}
    for r in rows:r['bh_fdr05']=id(r) in accepted
    harmonic=sum(1/i for i in range(1,len(rows)+1))
    by_cutoff=max([i for i,r in enumerate(ordered,1)
                   if r['minimum_adjusted_p']<=.05*i/(len(rows)*harmonic)]+[0])
    by_accepted={id(r) for r in ordered[:by_cutoff]}
    for r in rows:r['by_fdr05']=id(r) in by_accepted
    summary={'states':len(rows),'futures_per_action':64,'scores':sum(r['actions']*64 for r in rows),
             'statistical_scope':'paired Student-t intervals, Bonferroni within state; BH and dependence-robust BY across600 states additionally reported; one independent game per state/class',
             'A':group([r for r in rows if r['category']=='A']),
             'B':group([r for r in rows if r['category']=='B']),
             'groups':{},'examples':{}}
    for c,field in [('A','gap'),('B','wave')]:
        cr=[r for r in rows if r['category']==c]
        assert len(cr)==300 and len({r['seed'] for r in cr})==300
        summary['groups'][c]={str(v):group([r for r in cr if r[field]==v]) for v in sorted({r[field] for r in cr})}
        summary['examples'][c]=[{k:v for k,v in r.items() if k!='action_statistics'}
                                for r in sorted(cr,key=lambda r:r['witness']['ci95'][0],reverse=True)[:4]]
    (directory.parent/'point_statistics.json').write_text(json.dumps(rows,indent=2)+'\n')
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({c:summary[c] for c in ('A','B')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();summarize(args.directory,args.out)
