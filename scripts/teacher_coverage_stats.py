"""Summarize coverage and actual continuation cost, clustering by source game."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
import numpy as np
from pvz_candidate_coverage import action_key


def wilson(count,n):
    p=count/n;z=1.959963984540054;den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'count':int(count),'n':n,'rate':p,'wilson95':[mid-half,mid+half]}


def cluster_metrics(rows):
    games=sorted({r['seed'] for r in rows})
    index={s:i for i,s in enumerate(games)}
    arrays={name:np.zeros(len(games)) for name in ['n','old_noise','new2_agreement','new4_agreement',
        'new2_noise','new4_noise','old_count','new2_count','new4_count']}
    for r in rows:
        i=index[r['seed']];a=r['arms'];arrays['n'][i]+=1
        same=lambda x,y:action_key(a[x]['chosen'])==action_key(a[y]['chosen'])
        arrays['old_noise'][i]+=same('old','old_noise')
        arrays['old_count'][i]+=a['old']['actual_continuations']
        for budget in (2,4):
            name=f'new{budget}'
            arrays[name+'_agreement'][i]+=same('old',name)
            arrays[name+'_noise'][i]+=same(name,name+'_noise')
            arrays[name+'_count'][i]+=a[name]['actual_continuations']
    rng=np.random.default_rng(20261020)
    picks=rng.integers(0,len(games),size=(10000,len(games)))
    def totals(name):return arrays[name][picks].sum(axis=1)
    def metric(name,denominator='n'):
        return {'estimate':float(arrays[name].sum()/arrays[denominator].sum()),
                'game_cluster_ci95':np.quantile(totals(name)/totals(denominator),[.025,.975]).tolist()}
    report={name:metric(name) for name in arrays if name!='n'}
    for budget in (2,4):
        name=f'new{budget}'
        report[name+'_count_increase']={'estimate':float(arrays[name+'_count'].sum()/arrays['old_count'].sum()-1),
            'game_cluster_ci95':np.quantile(totals(name+'_count')/totals('old_count')-1,[.025,.975]).tolist()}
        report[name+'_agreement_minus_old_noise']={'estimate':report[name+'_agreement']['estimate']-report['old_noise']['estimate'],
            'game_cluster_ci95':np.quantile((totals(name+'_agreement')-totals('old_noise'))/totals('n'),[.025,.975]).tolist()}
    return {'points':len(rows),'games':len(games),'metrics':report}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--witnesses',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    rows=[json.loads(p.read_text()) for p in sorted(args.audit.glob('[AB]_*.json'))]
    assert len(rows)==600 and all(r['old_label_and_targets_exact'] for r in rows)
    assert Counter(r['category'] for r in rows)=={'A':300,'B':300}
    witness={(r['category'],r['seed'],r['decision']):r for r in json.loads(args.witnesses.read_text())
             if r['significant_inferiority']}
    report={'all600_default_off_labels_and_rounded_targets_exact':True,'pooled':cluster_metrics(rows),
            'classes':{},'witness_action':'original task9 multiplicity-significant witness_action, not reselected',
            'no_win_claim':True}
    for category in ('A','B'):
        group=[r for r in rows if r['category']==category]
        significant=[r for r in group if (category,r['seed'],r['decision']) in witness]
        assert len(significant)==(83 if category=='A' else 236)
        cls=cluster_metrics(group);cls['significant_points']=len(significant);cls['arms']={}
        for arm in group[0]['arms']:
            included=chosen=0;best_included=best_chosen=0
            for row in significant:
                w=witness[(category,row['seed'],row['decision'])]
                a=row['arms'][arm]
                keys={action_key(v) for v in a['candidates']}
                included+=action_key(w['witness_action']) in keys
                chosen+=action_key(w['witness_action'])==action_key(a['chosen'])
                best_included+=action_key(w['best_action']) in keys
                best_chosen+=action_key(w['best_action'])==action_key(a['chosen'])
            added=[a for row in group for a in row['arms'][arm]['added']]
            new_cells=new_rows=new_cols=0
            for row in group:
                old=[a for a in row['arms']['old']['candidates'] if a['type']=='plant']
                additions=row['arms'][arm]['added']
                new_cells+=len({(a['row'],a['col']) for a in additions}-{(a['row'],a['col']) for a in old})
                new_rows+=len({a['row'] for a in additions}-{a['row'] for a in old})
                new_cols+=len({a['col'] for a in additions}-{a['col'] for a in old})
            cls['arms'][arm]={'witness_included':wilson(included,len(significant)),
                'witness_chosen':wilson(chosen,len(significant)),
                'best_included_secondary':wilson(best_included,len(significant)),
                'best_chosen_secondary':wilson(best_chosen,len(significant)),
                'added_candidates':len(added),'added_by_packet':dict(Counter(a['packet'] for a in added)),
                'new_cells_total':new_cells,'new_rows_total':new_rows,'new_columns_total':new_cols,
                'actual_continuations':sum(r['arms'][arm]['actual_continuations'] for r in group)}
        report['classes'][category]=cls
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['pooled'],indent=2))


if __name__=='__main__':main()
