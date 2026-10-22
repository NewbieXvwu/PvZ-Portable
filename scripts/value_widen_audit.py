"""Frozen, seed-disjoint A6 audit of existing value nets; never launches games.

A sample is the first nonterminal frame of a wave. Cohorts are hash sampled by
seed (all archives of that seed stay together); old training seeds are excluded.
The manifest is also the authoritative held-out set for value v4 training.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'python'), str(ROOT / 'scripts')]
from pvz_frame_store import frame_path, read_frames
from pvz_value_net import context_vector, featurize, load_value_net
import pvz_value_net as value_encoding
from train_value_net import auc, wave_starts


def seed_key(seed):
    return hashlib.sha256(f'value-widen:{seed}'.encode()).hexdigest()


def cohort(label):
    if label.startswith('n20o'): return 'night_original_conveyor'
    if label.startswith('n') and ('dag' in label or 't' in label): return 'night_n9_n10'
    if label.startswith('p'): return 'pool_p9k'
    if label.startswith('d10orig'): return 'day_original_conveyor'
    if label.startswith('d') and 'p' in label: return 'day_d7p'
    if label.startswith('d'): return 'day_d7'
    if label.startswith(('z13', 'z14', 'z15')): return 'day_l7_control'
    return None


def calibration(y, p):
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        if mask.any():
            bins.append(dict(bin=i, n=int(mask.sum()), predicted=float(p[mask].mean()), actual=float(y[mask].mean())))
    return dict(n=len(y), actual=float(y.mean()), predicted=float(p.mean()),
                brier=float(np.mean((p-y)**2)), ece=sum(b['n']*abs(b['predicted']-b['actual']) for b in bins)/len(y), bins=bins)


def summarize(records, predictions):
    out = {}
    for group in sorted({r['cohort'] for r in records}):
        idx = [i for i,r in enumerate(records) if r['cohort']==group]
        y = np.array([records[i]['won'] for i in idx], dtype=float)
        games = {(records[i]['seed'],records[i]['archive']): records[i]['won'] for i in idx}
        models = {}
        for name,pred in predictions.items():
            p=np.asarray(pred)[idx]
            waves=[]
            for wave in sorted({records[i]['wave'] for i in idx}):
                mask=np.array([records[i]['wave']==wave for i in idx]); yy=y[mask]; pp=p[mask]
                waves.append(dict(wave=wave,n=int(mask.sum()),wins=int(yy.sum()),auc=auc(pp,yy),calibration=calibration(yy,pp)))
            models[name]=dict(calibration=calibration(y,p),per_wave=waves)
        out[group]=dict(games=len(games),wins=sum(games.values()),seeds=len({s for s,_ in games}),models=models)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',action='append',required=True,help='name=/absolute/model.pt')
    p.add_argument('--seeds-per-cohort',type=int,default=256)
    p.add_argument('--manifest',type=Path,help='reuse existing frozen manifest')
    args=p.parse_args();torch.set_num_threads(2);args.out.mkdir(parents=True,exist_ok=True)
    nets={};excluded=set()
    for spec in args.model:
        name,path=spec.split('=',1);ck=torch.load(path,map_location='cpu',weights_only=False)
        nets[name]=load_value_net(path);excluded.update(ck.get('trained_on',[]))
    if args.manifest:
        manifest=json.loads(args.manifest.read_text())
    else:
        groups=defaultdict(lambda:defaultdict(list))
        for sd in args.data.iterdir():
            if not sd.is_dir() or not sd.name.startswith('seed_'):continue
            seed=int(sd.name[5:])
            if seed in excluded:continue
            for archive in sd.iterdir():
                group=cohort(archive.name)
                if group and (archive/'meta.json').is_file():groups[group][seed].append(str(archive))
        # One archive per seed per cohort prevents prolific policies dominating the audit.
        manifest=[]
        for group,seeds in sorted(groups.items()):
            for seed in sorted(seeds,key=seed_key)[:args.seeds_per_cohort]:
                paths=sorted(seeds[seed]);archive=paths[int(seed_key(seed),16)%len(paths)]
                manifest.append(dict(cohort=group,seed=seed,archive=archive))
        (args.out/'manifest.json').write_text(json.dumps(manifest,indent=1))
    records=[];predictions={name:[] for name in nets}
    for gi,game in enumerate(manifest):
        archive=Path(game['archive']);meta=json.loads((archive/'meta.json').read_text())
        frames=read_frames(frame_path(archive));starts=wave_starts(frames)
        selected=[frames[i] for i in starts.values()]
        if not selected:continue
        # Each checkpoint chooses its own version; legacy calls remain bitwise frozen.
        for name,net in nets.items():
            pairs=[value_encoding.encode_frame(f, net.encoder_version) if getattr(net, "encoder_version", 1) != 1 else featurize(f) for f in selected]
            b=torch.stack([x[0] for x in pairs]);s=torch.stack([x[1] for x in pairs])
            if net.context:
                c=meta['context'];ctx=torch.from_numpy(context_vector(c['multiplier'],c['roster']))
                s=torch.cat([s,ctx.expand(len(s),-1)],1)
            with torch.inference_mode():predictions[name].extend(torch.sigmoid(net(b,s)).tolist())
        records.extend(dict(**game,wave=w,won=int(meta['outcome']['won'])) for w in starts)
        if (gi+1)%100==0:print(f'{gi+1}/{len(manifest)} games',flush=True)
    report=dict(protocol='A6: first nonterminal frame per game/wave; seed hash sampling; one archive per seed/cohort; all old training seeds excluded',
                excluded_training_seeds=len(excluded),cohorts=summarize(records,predictions))
    (args.out/'metrics.json').write_text(json.dumps(report,indent=1))
    np.savez_compressed(args.out/'predictions.npz',records=np.array(json.dumps(records)),**{k:np.array(v) for k,v in predictions.items()})
    for group,data in report['cohorts'].items():
        print(group,data['games'],data['wins'],{k:round(v['calibration']['brier'],3) for k,v in data['models'].items()},flush=True)
if __name__=='__main__':main()
