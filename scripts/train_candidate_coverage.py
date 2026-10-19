"""Fit the preregistered set-conditioned marginal coverage predictor on disjoint games."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'),str(ROOT/'scripts')]
import numpy as np
import torch
from pvz_candidate_coverage import CoverageNet, SCHEMA


def marginal(scores, selected):
    return np.maximum(scores-scores[selected].max(axis=0),0).mean(axis=1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261020)
    rng=random.Random(20261020)
    selection=json.loads(args.selection.read_text())['points']
    data={}; records=[]
    for point in selection:
        tag=f"{point['category']}_{point['seed']}_{point['decision']}"
        path=args.data/(tag+'.npz')
        with np.load(path) as z:
            d={k:z[k] for k in z.files}
        assert np.isfinite(d['scores']).all()
        data[tag]=d
        base=d['base'].tolist()
        available=[i for i in range(len(d['scores'])) if i not in base]
        oracle=max(available,key=lambda i:float(d['scores'][i].mean())) if available else None
        random_action=rng.choice(available) if available else None
        for extra in (None,oracle,random_action):
            chosen=base+([extra] if extra is not None else [])
            target=marginal(d['scores'],chosen)
            records.append((point['split'],tag,chosen,torch.from_numpy(target)))
    train=[r for r in records if r[0]=='train']
    valid=[r for r in records if r[0]=='validation']
    assert train and valid
    games={split:{p['seed'] for p in selection if p['split']==split} for split in ('train','validation')}
    assert not games['train'] & games['validation']
    example=next(iter(data.values()))
    model=CoverageNet(len(example['state']),example['features'].shape[1])
    train_tags=sorted({r[1] for r in train})
    states=np.stack([data[t]['state'] for t in train_tags])
    actions=np.concatenate([data[t]['features'] for t in train_tags])
    with torch.no_grad():
        for prefix,x in [('state',states),('action',actions)]:
            getattr(model,prefix+'_mean').copy_(torch.from_numpy(x.mean(axis=0)))
            getattr(model,prefix+'_scale').copy_(torch.from_numpy(np.maximum(x.std(axis=0),1e-3)))
    tensors={t:(torch.from_numpy(d['state']),torch.from_numpy(d['features'])) for t,d in data.items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    best=float('inf'); curve=[]
    args.out.mkdir(parents=True,exist_ok=True)
    for epoch in range(80):
        model.train(); rng.shuffle(train); losses=[]
        for start in range(0,len(train),8):
            optimizer.zero_grad(); group=[]
            for _,tag,chosen,target in train[start:start+8]:
                s,x=tensors[tag]
                group.append(torch.nn.functional.mse_loss(model(s,x,chosen),target))
            loss=torch.stack(group).mean()
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            mse=float(np.mean([float(torch.nn.functional.mse_loss(model(*tensors[tag],chosen),target))
                               for _,tag,chosen,target in valid]))
        row={'epoch':epoch+1,'train_mse':float(np.mean(losses)),'validation_mse':mse}
        curve.append(row)
        if mse<best:
            best=mse
            torch.save({'schema_version':SCHEMA,'state_dim':len(example['state']),
                        'action_dim':example['features'].shape[1], 'state_dict':model.state_dict(),
                        'epoch':epoch+1,'validation_mse':mse},args.out/'model.pt')
        print(json.dumps(row),flush=True)
        (args.out/'learning_curve.json').write_text(json.dumps(curve,indent=1)+'\n')
    sources={str(args.data/(t+'.npz')):hashlib.sha256((args.data/(t+'.npz')).read_bytes()).hexdigest() for t in data}
    model_file=args.out/'model.pt'
    report={'seed':20261020,'epochs':80,'train_games':sorted(games['train']),
            'validation_games':sorted(games['validation']),'states':len(data),
            'train_contexts':len(train),'validation_contexts':len(valid),
            'best_validation_mse':best,
            'zero_validation_mse':float(np.mean([float((r[3]**2).mean()) for r in valid])),
            'model_sha256':hashlib.sha256(model_file.read_bytes()).hexdigest(),
            'model_bytes':model_file.stat().st_size, 'source_sha256':sources,
            'selection_sha256':hashlib.sha256(args.selection.read_bytes()).hexdigest()}
    (args.out/'training.json').write_text(json.dumps(report,indent=1)+'\n')


if __name__=='__main__':
    main()
