"""Join counterfactual results, exact historical traces and fresh v29 controls."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from teacher_audit_stats import paired


def identity(row):
    return row['category'],row['seed'],row['decision']


def action_key(action):
    if action['type']=='wait':
        return 'wait'
    return action['type'],action.get('packet'),action['row'],action['col']


def mechanism(row,trace):
    if not trace['label_reproduced'] or not trace['targets_reproduced']:
        return 'historical_trace_not_exact'
    witness=action_key(row['witness_action'])
    index=next((i for i,a in enumerate(trace['candidates']) if action_key(a)==witness),None)
    if index is None:
        return 'candidate_generation_omission'
    if trace['counts'][index]<3:
        return 'halving_pruned_before_three_futures'
    if index==trace['best_index']:
        return 'verification_plus_margin_rejected' if trace.get('verification') else 'initial_margin_no_verification'
    if index==trace['policy_index']:
        return 'policy_overridden_by_other_candidate'
    return 'initial_three_future_ranking_not_final_challenger'


def summarize(directory,out):
    rows=json.loads((directory/'point_statistics.json').read_text())
    traces={identity(r):r for p in (directory/'native31_traces').glob('*.json') for r in [json.loads(p.read_text())]}
    controls={identity(r):r for p in (directory/'v29_policy_controls64').glob('*.json') for r in [json.loads(p.read_text())]}
    assert len(rows)==len(traces)==600
    bad=[r for r in rows if r['significant_inferiority']]
    assert set(controls)=={identity(r) for r in bad}
    joined=[]
    for row in bad:
        trace=traces[identity(row)];control=controls[identity(row)]
        delta=np.asarray(control['scores'][1])-np.asarray(control['scores'][0])
        result=paired(delta)
        joined.append({**row,'historical_mechanism':mechanism(row,trace),
                       'v29_fresh_control':result,'historical_trace_exact':trace['label_reproduced'] and trace['targets_reproduced'],
                       'v29_still_significantly_inferior':result['ci95'][0]>0,
                       'v29_significantly_favors_original':result['ci95'][1]<0})
    ordered=sorted(joined,key=lambda r:r['v29_fresh_control']['p_adjusted'] if r['v29_fresh_control']['mean']>0 else 1.)
    harmonic=sum(1/i for i in range(1,len(joined)+1))
    for method,factor in [('bh',1.),('by',harmonic)]:
        cutoff=max([i for i,r in enumerate(ordered,1) if r['v29_fresh_control']['mean']>0
                    and r['v29_fresh_control']['p_adjusted']<=.05*i/(len(joined)*factor)]+[0])
        accepted={id(r) for r in ordered[:cutoff]}
        for r in joined:r[f'v29_{method}_fdr05']=id(r) in accepted
    summary={'historical_traces':600,
             'labels_reproduced':sum(r['label_reproduced'] for r in traces.values()),
             'targets_reproduced':sum(r['targets_reproduced'] for r in traces.values()),
             'both_reproduced':sum(r['label_reproduced'] and r['targets_reproduced'] for r in traces.values()),
             'controls':len(joined),'groups':{},'examples':{},
             'interpretation':'Mechanisms describe where the historical choice diverged; fresh v29 effects distinguish original-objective errors from continuation-policy shifts. Post-verification rejection includes intentional margin anchoring, not automatically sample noise.'}
    for c in ('A','B'):
        rr=[r for r in joined if r['category']==c]
        summary['groups'][c]={'primary_significant':len(rr),
                              'historical_mechanisms':dict(Counter(r['historical_mechanism'] for r in rr)),
                              'v29_still_significant':sum(r['v29_still_significantly_inferior'] for r in rr),
                              'v29_favors_original':sum(r['v29_significantly_favors_original'] for r in rr),
                              'v29_inconclusive':sum(not r['v29_still_significantly_inferior'] and not r['v29_significantly_favors_original'] for r in rr),
                              'v29_bh_fdr05':sum(r['v29_bh_fdr05'] for r in rr),
                              'v29_by_fdr05':sum(r['v29_by_fdr05'] for r in rr),
                              'mechanisms_confirmed_v29':dict(Counter(r['historical_mechanism'] for r in rr if r['v29_still_significantly_inferior']))}
        summary['examples'][c]=[{k:v for k,v in r.items() if k!='action_statistics'}
                                for r in sorted(rr,key=lambda r:r['v29_fresh_control']['ci95'][0],reverse=True)[:4]]
    (directory/'attribution_points.json').write_text(json.dumps(joined,indent=2)+'\n')
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='examples'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();summarize(args.directory,args.out)
