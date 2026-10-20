"""Build small row-identity/weight metadata for fixed-cohort DAgger experiments."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from pvz_training_rows import update_identity


def kept_rows(hints,rows,rng,wait_keep=.1):
    # Match the trainer's global Random(0) stream, including frame0 and invalid rows.
    mask=np.asarray([bool(v) or rng.random()<wait_keep for v in hints])
    return np.flatnonzero(mask[rows])


def assemble(inventory,models,out,cache,name,model_name,expected_rows):
    import torch
    torch.set_num_threads(1)
    labels=set(models[model_name]['labels'].split(','))
    episodes=[r for r in inventory if r['label'] in labels]
    assert episodes==sorted(episodes,key=lambda r:r['source'])
    rng=random.Random(0);selection=[];n=0
    for i,r in enumerate(episodes):
        with np.load(r['audit_arrays'],allow_pickle=False) as z:
            chosen=kept_rows(z['hints'],z['rows'],rng)
        selection.append(chosen.astype(np.int32));n+=len(chosen)
    assert n==expected_rows,(name,n,expected_rows)
    dest=out/name;dest.mkdir(parents=True,exist_ok=True)
    dtypes={'episode':np.int32,'row':np.int32,'packet':np.int8,'spread':np.float32,
            'compared':bool,'cell_soft':bool,'override':bool,'known':bool}
    arrays={k:np.lib.format.open_memmap(dest/(k+'.npy'),mode='w+',dtype=t,shape=(n,)) for k,t in dtypes.items()}
    digest=hashlib.sha256();offset=0
    for i,(r,selected) in enumerate(zip(episodes,selection)):
        count=len(selected);sl=slice(offset,offset+count)
        with np.load(r['audit_arrays'],allow_pickle=False) as z:
            rows=z['rows'][selected]
            arrays['episode'][sl]=i;arrays['row'][sl]=rows
            for k in ('packet','spread','compared','cell_soft','override'):arrays[k][sl]=z[k][selected]
        ident=hashlib.sha256(str(Path(r['source']).resolve()).encode()).hexdigest()
        with np.load(cache/'v2'/(ident+'.npz'),allow_pickle=False) as z:
            assert str(z['digest'])==r['cache_digest'],('cache changed',r['source'])
            values=z['values'][selected]
            assert np.logical_or(np.isnan(values),np.isfinite(values)).all()
            arrays['known'][sl]=(~np.isnan(values)).any(1)
        update_identity(digest,r['relative_source'],rows)
        offset+=count
        if i%5000==0:print(json.dumps({'cohort':name,'assembled_episodes':i,'of':len(episodes)}),flush=True)
    assert offset==n
    all_seeds=sorted({r['seed'] for r,s in zip(episodes,selection) if len(s)})
    random.Random(0).shuffle(all_seeds)
    val_seeds=set(all_seeds[:len(all_seeds)//10])
    val_by_episode=np.asarray([r['seed'] in val_seeds for r in episodes])
    is_val=np.lib.format.open_memmap(dest/'is_val.npy',mode='w+',dtype=bool,shape=(n,))
    is_val[:]=val_by_episode[arrays['episode']]
    packet=np.asarray(arrays['packet']);planted=packet>=0
    weights=torch.from_numpy(np.where(planted|np.asarray(arrays['known']),1.,10.).astype(np.float32))
    pk=torch.from_numpy(packet.astype(np.int64))
    freq=torch.bincount(pk[planted],minlength=10).float()/int(planted.sum())
    scale=freq.clamp(min=1e-6)**-.5
    scale=(scale/scale[pk[planted]].mean()).clamp(max=20.)
    weights[planted]*=scale[pk[planted]]
    compared=np.asarray(arrays['compared']);spread=torch.tensor(np.asarray(arrays['spread'])[compared])
    mean_spread=spread.mean()
    weights[compared]*=((1+8*spread/mean_spread)/9).clamp(max=10.)
    np.save(dest/'weight.npy',weights.numpy())
    summary={}
    for label in sorted(labels):
        ids=[i for i,r in enumerate(episodes) if r['label']==label]
        raw=[episodes[i] for i in ids]
        mask=np.isin(arrays['episode'],ids);tr=mask&~is_val
        w=weights.numpy()[tr].astype(np.float64);ov=np.asarray(arrays['override'])[tr]
        summary[label]={'episodes':len(ids),'multiplier':raw[0]['multiplier'],
            'raw_rows':sum(r['valid_rows'] for r in raw),'raw_overrides':sum(r['override_rows'] for r in raw),
            'kept_rows':int(mask.sum()),'train_rows':int(tr.sum()),'train_weight':float(w.sum()),
            'train_override_rows':int(ov.sum()),'train_override_weight':float(w[ov].sum()),
            'train_compared_weight':float(w[np.asarray(arrays['compared'])[tr]].sum()),
            'train_cell_soft_weight':float(w[np.asarray(arrays['cell_soft'])[tr]].sum()),
            'train_plant_rows':int(planted[tr].sum()),'train_wait_weight':float(w[~planted[tr]].sum())}
    for v in arrays.values():v.flush()
    is_val.flush()
    info={'name':name,'labels':sorted(labels),'rows':n,'train_rows':int((~is_val).sum()),
          'val_rows':int(is_val.sum()),'seeds':len(all_seeds),'row_identity_sha256':digest.hexdigest(),
          'split_sha256':hashlib.sha256(np.packbits(is_val).tobytes()).hexdigest(),
          'optimizer_steps_12epochs':12*math.ceil(int((~is_val).sum())/2048),
          'mean_spread':float(mean_spread),'packet_scale':scale.tolist(),
          'by_label':summary,'episodes':episodes}
    (dest/'cohort.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path.home()/'PvZ-search/artifacts/student/v2_cache')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    inventory=json.loads((args.audit/'inventory.json').read_text())
    models=json.loads(args.historical.read_text())
    for name,model,expected in [('D30','d30cs3w8s1',8038389),('D35','d35cs3w8s1',9914278)]:
        info=assemble(inventory,models,args.out,args.cache,name,model,expected)
        print(json.dumps({k:info[k] for k in ['name','rows','train_rows','val_rows','optimizer_steps_12epochs']}),flush=True)


if __name__=='__main__':main()
