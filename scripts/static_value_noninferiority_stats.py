"""Prespecified new-game noninferiority and paired search CPU/cost intervals."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
from static_value_compare import key
from static_value_noninferiority import correctness_difference
from teacher_audit_stats import t_critical


def counts(row):
    arms = row['arms']
    same = key(arms['A']['chosen']) == key(arms['B3']['chosen'])
    assert row['judge_required'] == (not same)
    assert (row['heavy'] is None) == same
    label = row['heavy']['label'] if row['heavy'] else None
    delta = correctness_difference(arms['A']['chosen'], arms['B3']['chosen'], label)
    assert row['delta'] == delta
    out = {'points':1, 'delta':delta, 'identical':int(same), 'judged':int(not same),
           'only_A_correct':int(delta == -1), 'only_B3_correct':int(delta == 1),
           'disagreed_both_wrong':int(not same and delta == 0),
           'truth_continuations':row['heavy']['truth_continuations'] if row['heavy'] else 0,
           'heavy_verification_continuations':row['heavy']['verification_continuations'] if row['heavy'] else 0}
    for name in ('A','B3'):
        a = arms[name]
        assert a['continuations'] == a['halving_continuations'] + a['verification_continuations']
        assert abs(a['cpu']['total_s'] - a['cpu']['worker']['total_s'] - a['cpu']['simulator']['total_s']) < 1e-9
        out[name+'_cost'] = a['continuations']
        out[name+'_halving'] = a['halving_continuations']
        out[name+'_verify'] = a['verification_continuations']
        out[name+'_cpu'] = a['cpu']['total_s']
        out[name+'_worker_cpu'] = a['cpu']['worker']['total_s']
        out[name+'_sim_cpu'] = a['cpu']['simulator']['total_s']
        out[name+'_candidates'] = len(a['candidates'])
    return out


def summarize(games, formal=True):
    assert len(games) >= 2 and all(g['points'] == 8 for g in games)
    names = sorted(set().union(*(set(g) for g in games)))
    arrays = {n:np.asarray([g.get(n,0) for g in games],dtype=float) for n in names}
    means = arrays['delta']/arrays['points']
    estimate = float(means.mean())
    se = float(means.std(ddof=1)/math.sqrt(len(means)))
    critical = t_critical(len(means)-1)
    ci = [estimate-critical*se, estimate+critical*se]
    picks = np.random.default_rng(20261020).integers(0,len(games),size=(10000,len(games)))
    def ratio(num,den):
        assert den.sum() > 0
        boot = num[picks].sum(axis=1)/den[picks].sum(axis=1)
        return {'estimate':float(num.sum()/den.sum()),'game_cluster_ci95':np.quantile(boot,[.025,.975]).tolist()}
    metrics = {n:ratio(v,arrays['points']) for n,v in arrays.items() if n != 'points'}
    for cost in ('cost','cpu','worker_cpu','sim_cpu'):
        metrics['B3_to_A_'+cost+'_ratio'] = ratio(arrays['B3_'+cost],arrays['A_'+cost])
        v = metrics['B3_to_A_'+cost+'_ratio']
        metrics['B3_'+cost+'_saving'] = {'estimate':1-v['estimate'],
            'game_cluster_ci95':[1-v['game_cluster_ci95'][1],1-v['game_cluster_ci95'][0]]}
    passed = ci[0] > -.03
    bootstrap_passed = metrics['delta']['game_cluster_ci95'][0] > -.03
    return {'games':len(games),'points':int(arrays['points'].sum()),'formal_final_analysis':formal,
        'primary':{'difference_B3_minus_A':estimate,'se_of_game_means':se,'df':len(games)-1,
                   't_critical':critical,'ci95':ci,'NI_margin':.03,
                   'noninferiority_passed':bool(passed) if formal else None,
                   'bootstrap_agrees_with_NI':bool(bootstrap_passed) if formal else None,
                   'robust_NI':bool(passed and bootstrap_passed) if formal else None},
        'metrics':metrics,'totals':{n:float(v.sum()) for n,v in arrays.items()},
        'bootstrap_resamples':10000,'bootstrap_seed':20261020,
        'absolute_accuracy':'Not estimated: identical-choice roots cancel exactly but are deliberately unlabelled. Primary paired difference is exact over ALL sampled roots.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    plan=json.loads((Path(__file__).resolve().parents[1]/'MEMORY/docs/static-value-ablation-ni-plan.json').read_text())
    summaries=[json.loads(f.read_text()) for f in sorted(args.data.glob('*_summary.json'))]
    records=[r for s in summaries for r in s]
    metas=[r['trajectory'] for r in records]
    expected={(m['seed'],p['decision']) for m in metas for p in m['points']}
    rows=[json.loads(p.read_text()) for p in sorted((args.data/'points').glob('*.json'))]
    assert len(rows)==len(expected) and {(r['seed'],r['decision']) for r in rows}==expected
    seeds=sorted(m['seed'] for m in metas)
    assert seeds==plan['sampling']['seeds'][:len(seeds)] and len(seeds) in (128,256)
    assert not set(seeds).intersection(plan['sampling']['excluded_old_seeds'])
    games=defaultdict(lambda:defaultdict(float))
    for row in rows:
        for k,v in counts(row).items():games[row['seed']][k]+=v
    report=summarize([games[s] for s in sorted(games)],formal=len(seeds)==256)
    report['per_game']={str(s):dict(games[s]) for s in sorted(games)}
    report['strict_replay_frames']=sum(r['comparison']['strict_replay_frames'] for r in records)
    report['terminal_trajectories']=sum(m['terminal'] for m in metas)
    report['total_actual_continuations']=sum(report['totals'][n] for n in
        ['A_cost','B3_cost','truth_continuations','heavy_verification_continuations'])
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'games':report['games'],'points':report['points'],'primary':report['primary']}))


if __name__=='__main__':main()
