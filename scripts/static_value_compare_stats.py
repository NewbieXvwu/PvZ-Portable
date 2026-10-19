"""Paired ordering/teacher agreement, with whole-game cluster confidence intervals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'python')]
import numpy as np
from static_value_compare import key
from teacher_audit_stats import t_critical


def ordering_credit(prediction_delta,truth_delta):
    if truth_delta==0:return None
    if prediction_delta==0:return .5
    return float(prediction_delta*truth_delta>0)


def point_counts(row):
    actions=row['actions'];scores=np.asarray(row['truth_scores'],dtype=np.float64)
    assert scores.shape==(len(actions),64) and np.isfinite(scores).all()
    assert np.logical_and(scores>=0,scores<=1).all()
    assert len({key(a) for a in actions})==len(actions)
    values=scores.mean(axis=1);out=defaultdict(float)
    for i in range(len(actions)):
        for j in range(i+1,len(actions)):
            delta=scores[i]-scores[j];truth=float(delta.mean())
            same_card=(actions[i]['type']==actions[j]['type']=='plant'
                       and actions[i]['packet']==actions[j]['packet'])
            if truth==0:
                out['truth_ties_all']+=1
                out['truth_ties_within']+=int(same_card)
                continue
            groups=['all']
            if same_card:
                groups.append('within')
                if abs(truth)>t_critical(63)*float(delta.std(ddof=1))/8:
                    groups.append('robust_within')
            for group in groups:
                out[group+'_pairs']+=1
                for name in ('mac','desktop'):
                    pred=row['static_scores'][name][i]-row['static_scores'][name][j]
                    out[group+'_'+name]+=ordering_credit(pred,truth)
    out['points']=1
    arms=row['normal_arms'];heavy=key(row['heavy_label'])
    for name in ('mac','desktop'):
        for action in arms[name]['candidates']:
            assert key(action) in {key(a) for a in actions}
        assert arms[name]['candidates']==arms[name+'_noise']['candidates']
        out[name+'_heavy']=float(key(arms[name]['chosen'])==heavy)
        out[name+'_noise']=float(key(arms[name]['chosen'])==key(arms[name+'_noise']['chosen']))
        out[name+'_noise_heavy']=float(key(arms[name+'_noise']['chosen'])==heavy)
    out['inter_model_agreement']=float(key(arms['mac']['chosen'])==key(arms['desktop']['chosen']))
    out['student_heavy']=float(key(row['policy'])==heavy)
    out['truth_continuations']=row['truth_continuations']
    out['heavy_extra_continuations']=row['heavy_extra_continuations']
    out['normal_continuations']=sum(a['continuations'] for a in arms.values())
    # Before wait: exactly the original one-best-cell-per-card static candidates.
    top={}
    for name in ('mac','desktop'):
        top[name]={}
        for action in arms[name]['candidates']:
            if action['type']=='wait':break
            assert action['type']=='plant' and action['packet'] not in top[name]
            top[name][action['packet']]=action
    assert top['mac'].keys()==top['desktop'].keys()
    index={key(a):i for i,a in enumerate(actions)}
    for packet in top['mac']:
        m,d=key(top['mac'][packet]),key(top['desktop'][packet])
        gap=float(values[index[d]]-values[index[m]])
        out['top_cells']+=1;out['top_desktop_minus_mac_sum']+=gap
        if m!=d:
            out['different_top_cells']+=1
            out['different_top_desktop_minus_mac_sum']+=gap
            out['different_top_desktop_wins']+=int(gap>0)
            out['different_top_mac_wins']+=int(gap<0)
            out['different_top_truth_ties']+=int(gap==0)
    return dict(out)


def clustered(games,seed=20261020):
    """Each row is one game's sums; all ratios share the same game resamples."""
    names=sorted({name for game in games for name in game})
    arrays={name:np.asarray([game.get(name,0) for game in games],dtype=float) for name in names}
    rng=np.random.default_rng(seed);picks=rng.integers(0,len(games),size=(10000,len(games)))
    def ratio(numerator,denominator):
        num=np.asarray(numerator,dtype=float);den=arrays.get(denominator,np.zeros(len(games)))
        if den.sum()==0:return {'n':0,'estimate':None,'game_cluster_ci95':None}
        sampled_den=den[picks].sum(axis=1);good=sampled_den>0
        sampled=num[picks].sum(axis=1)[good]/sampled_den[good]
        return {'n':int(den.sum()),'estimate':float(num.sum()/den.sum()),
                'game_cluster_ci95':np.quantile(sampled,[.025,.975]).tolist(),
                'bootstrap_resamples':int(good.sum())}
    result={}
    get=lambda name:arrays.get(name,np.zeros(len(games)))
    for group in ('within','robust_within','all'):
        for model in ('mac','desktop'):
            result[group+'_'+model]=ratio(get(group+'_'+model),group+'_pairs')
        result[group+'_desktop_minus_mac']=ratio(get(group+'_desktop')-get(group+'_mac'),group+'_pairs')
    for name in ('mac_heavy','desktop_heavy','mac_noise','desktop_noise',
                 'mac_noise_heavy','desktop_noise_heavy','inter_model_agreement','student_heavy'):
        result[name]=ratio(get(name),'points')
    result['heavy_agreement_desktop_minus_mac']=ratio(get('desktop_heavy')-get('mac_heavy'),'points')
    result['reseed_agreement_desktop_minus_mac']=ratio(get('desktop_noise')-get('mac_noise'),'points')
    result['top_cell_mean_leaf_desktop_minus_mac']=ratio(get('top_desktop_minus_mac_sum'),'top_cells')
    result['different_top_mean_leaf_desktop_minus_mac']=ratio(get('different_top_desktop_minus_mac_sum'),'different_top_cells')
    return {'games':len(games),'totals':{k:float(v.sum()) for k,v in arrays.items()},'metrics':result}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    trajectory=json.loads((args.data/'generate_summary.json').read_text())
    expected={(m['seed'],p['decision']) for m in trajectory for p in m['points']}
    rows=[json.loads(p.read_text()) for p in sorted((args.data/'points').glob('*.json'))]
    assert {(r['seed'],r['decision']) for r in rows}==expected
    assert len(rows)==len(expected) and len({r['seed'] for r in rows})==32
    games=defaultdict(lambda:defaultdict(float))
    for row in rows:
        for k,v in point_counts(row).items():games[row['seed']][k]+=v
    report=clustered([games[s] for s in sorted(games)])
    report.update(states=len(rows),reference='common full pool64 futures/action + V18 verification; v30/v3/2500/mask61',
                  uncertainty='whole-game bootstrap, not independent action pairs; robust_within is a secondary paired95%-separated subset, not multiplicity-corrected discoveries',
                  per_game={str(s):dict(games[s]) for s in sorted(games)})
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['metrics'],indent=2))


if __name__=='__main__':main()
