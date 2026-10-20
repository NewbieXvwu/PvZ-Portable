"""Freeze controlled training-row multipliers without rewriting any source frame."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def normalize_selection(base,selected):
    if not selected.any():selected=np.ones(len(base),bool)
    factor=float(np.sum(base,dtype=np.float64)/np.sum(base[selected],dtype=np.float64))
    weights=selected.astype(np.float32)*np.float32(factor)
    actual=np.sum(base.astype(np.float64)*weights,dtype=np.float64)
    assert np.isclose(actual,np.sum(base,dtype=np.float64),rtol=2e-6)
    return weights,selected,factor


def matched_random(keep,compared,cell_soft,rng):
    code=compared.astype(np.uint8)*2+cell_soft.astype(np.uint8)
    result=np.zeros(len(keep),bool)
    for group in range(4):
        where=np.flatnonzero(code==group);n=int(keep[where].sum())
        if n:result[rng.choice(where,n,replace=False)]=True
        assert result[where].sum()==n
    return result


def build(cohort_dir,old_labels,out):
    cohort=json.loads((cohort_dir/'cohort.json').read_text());out.mkdir(parents=True,exist_ok=True)
    keys=['episode','packet','spread','compared','cell_soft','override','is_val','weight']
    a={k:np.load(cohort_dir/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in keys}
    n=cohort['rows'];train=~a['is_val'];weight=a['weight'];ep=a['episode'];packet=a['packet']
    assert all(len(v)==n for v in a.values())
    labels=np.asarray([r['label'] for r in cohort['episodes']]);z=np.asarray([r['multiplier'] for r in cohort['episodes']])
    unique_z=sorted(set(z));zcode=np.searchsorted(unique_z,z)
    M=np.ones(n,np.float32);I=np.ones(n,np.float32);R=np.ones(n,np.float32)
    balancing=[];rules=[];rng=np.random.default_rng(20261020)
    for p in sorted(set(packet)):
        for soft_type in (False,True):
            ix=np.flatnonzero(train&(packet==p)&(a['cell_soft']==soft_type))
            if not len(ix):continue
            groups=zcode[ep[ix]]
            mass=np.bincount(groups,weights=weight[ix],minlength=len(unique_z));present=mass>0
            factors=np.ones(len(unique_z),np.float64)
            factors[present]=mass.sum()/present.sum()/mass[present]
            M[ix]=factors[groups].astype(np.float32)
            after=np.bincount(groups,weights=weight[ix]*M[ix],minlength=len(unique_z))
            assert np.isclose(after.sum(),mass.sum(),rtol=2e-6)
            assert np.allclose(after[present],mass.sum()/present.sum(),rtol=2e-6)
            balancing.append({'packet':int(p),'cell_soft':soft_type,'multipliers':unique_z,
                              'original_mass':mass.tolist(),'balanced_mass':after.tolist(),'factors':factors.tolist()})
    for label in sorted(set(labels)-set(old_labels)):
        episode_ids=np.flatnonzero(labels==label)
        in_label=train&np.isin(ep,episode_ids)
        for p in sorted(set(packet[in_label])):
            for soft_type in (False,True):
                ix=np.flatnonzero(in_label&(packet==p)&(a['cell_soft']==soft_type))
                if not len(ix):continue
                compared=np.asarray(a['compared'][ix]);soft=np.asarray(a['cell_soft'][ix])
                spread=np.asarray(a['spread'][ix]);override=np.asarray(a['override'][ix])
                nonzero=compared&(spread>0)
                threshold=float(np.quantile(spread[nonzero],.75)) if nonzero.any() else None
                keep=override.copy()
                if threshold is not None:keep|=nonzero&(spread>=threshold)
                fallback=not keep.any()
                iw,keep,ifactor=normalize_selection(np.asarray(weight[ix]),keep)
                control=matched_random(keep,compared,soft,rng)
                rw,control,rfactor=normalize_selection(np.asarray(weight[ix]),control)
                assert keep.sum()==control.sum()
                I[ix]=iw;R[ix]=rw
                rules.append({'label':label,'packet':int(p),'cell_soft':soft_type,'train_rows':len(ix),
                              'selected_each':int(keep.sum()),'overrides':int(override.sum()),
                              'no_informative_rows_preserve_stratum':fallback,
                              'spread_q75_nonzero_compared':threshold,
                              'informative_mass_factor':ifactor,'random_mass_factor':rfactor})
    specs={}
    for name,multipliers in [('M',M),('I',I),('R',R)]:
        assert np.equal(multipliers[a['is_val']],1).all()
        path=out/(name+'.npy');np.save(path,multipliers)
        spec={'name':name,'rows':n,'multipliers':path.name,
              'row_identity_sha256':cohort['row_identity_sha256'],'split_sha256':cohort['split_sha256'],
              'multipliers_sha256':hashlib.sha256(multipliers.tobytes()).hexdigest(),
              'positive_train_rows':int((multipliers[train]>0).sum()),
              'train_weight_before':float(np.sum(weight[train],dtype=np.float64)),
              'train_weight_after':float(np.sum(weight[train].astype(np.float64)*multipliers[train],dtype=np.float64)),
              'max_multiplier':float(multipliers.max()),'min_multiplier':float(multipliers.min())}
        assert np.isclose(spec['train_weight_before'],spec['train_weight_after'],rtol=2e-6)
        dest=out/(name+'.json');dest.write_text(json.dumps(spec,indent=2)+'\n');specs[name]=spec
    report={'cohort':str(cohort_dir),'rows':n,'old_labels_unchanged_for_I_R':sorted(old_labels),
            'balanced_multiplier_within_packet':balancing,'informative_vs_random_strata':rules,
            'random_mask_seed':20261020,'validation_multipliers_all_one':True,'specs':specs}
    (out/'interventions.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohorts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    old=json.loads((args.cohorts/'D30/cohort.json').read_text())
    report=build(args.cohorts/'D35',old['labels'],args.out)
    print(json.dumps(report['specs'],indent=2))


if __name__=='__main__':main()
