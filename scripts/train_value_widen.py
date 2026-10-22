"""Train value v4 on legacy L7 branches and cross-zone result-labelled archives.

Memory is bounded by one episode and one minibatch: archive features are cached in
on-disk numpy memmaps; branch tensors use torch mmap, never concatenated. Audit
seeds are excluded globally, and early stopping is split by environment seed.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from pvz_frame_store import frame_path,read_frames
from pvz_value_net import SCALARS, CONTEXT, ENCODER_V2, ValueNet, encode_frame, context_vector, load_value_net
from value_widen_audit import cohort, seed_key


def held(seed):
    return int(seed_key(int(seed))[:8],16)%10==0


def prepare(args):
    excluded={g['seed'] for g in json.loads(args.audit_manifest.read_text())}
    groups=defaultdict(lambda:defaultdict(list))
    for sd in args.data.iterdir():
        if not sd.is_dir() or not sd.name.startswith('seed_'):continue
        seed=int(sd.name[5:])
        if seed in excluded:continue
        for a in sd.iterdir():
            group=cohort(a.name)
            if group and (a/'meta.json').is_file():groups[group][seed].append(str(a))
    selected=[]
    for group,seeds in sorted(groups.items()):
        for seed in sorted(seeds,key=seed_key)[:args.games_per_cohort]:
            paths=sorted(seeds[seed]);selected.append(dict(cohort=group,seed=seed,archive=paths[int(seed_key(seed),16)%len(paths)]))
    # Supplemental teacher games are few: retain all of them, leaving deterministic
    # held-out seeds for validation. They never reuse any external A6 audit seed.
    if args.supplement:
        for meta_path in sorted((args.supplement/'frames').glob('*/*/meta.json')):
            meta=json.loads(meta_path.read_text());seed=int(meta['task']['seed'])
            if seed in excluded:continue
            group='pool_p9k' if 'pool' in meta_path.parent.name else 'night_original_conveyor'
            selected.append(dict(cohort=group,seed=seed,archive=str(meta_path.parent)))
    if not selected:raise ValueError('no training archives')
    args.out.mkdir(parents=True,exist_ok=True)
    capacity=(len(selected)+256)*args.frames_per_game
    b=np.lib.format.open_memmap(args.out/'boards.npy',mode='w+',dtype=np.float16,shape=(capacity,92,6,10))
    s=np.lib.format.open_memmap(args.out/'scalars.npy',mode='w+',dtype=np.float32,shape=(capacity,SCALARS+6+CONTEXT))
    labels=[];seeds=[];cohorts=[];n=0;summary=defaultdict(lambda:dict(games=0,wins=0,rows=0))
    for gi,g in enumerate(selected):
        a=Path(g['archive']);meta=json.loads((a/'meta.json').read_text());frames=read_frames(frame_path(a))
        live=[i for i,f in enumerate(frames) if not f.get('terminal')]
        # Uniformly spread a fixed quota through each game; long night games cannot dominate.
        picked=np.linspace(0,len(live)-1,min(len(live),args.frames_per_game),dtype=int)
        ctx=meta['context'];cv=context_vector(ctx['multiplier'],ctx['roster'])
        for index in picked:
            board,scalars=encode_frame(frames[live[index]],2);b[n]=board;s[n]=np.concatenate([scalars.numpy(),cv]);n+=1
            labels.append(float(meta['outcome']['won']));seeds.append(g['seed']);cohorts.append(g['cohort'])
        item=summary[g['cohort']];item['games']+=1;item['wins']+=int(meta['outcome']['won']);item['rows']+=len(picked)
        if (gi+1)%100==0:print(f'prepared {gi+1}/{len(selected)} games',flush=True)
    b.flush();s.flush()
    np.savez(args.out/'rows.npz',labels=np.array(labels,np.float32),seeds=np.array(seeds,np.int64),cohorts=np.array(cohorts))
    (args.out/'episodes.json').write_text(json.dumps(dict(games=selected,rows=n,excluded_audit_seeds=sorted(excluded),summary=summary),indent=1))
    print(json.dumps(summary,indent=1),flush=True)


def append_supplement(args):
    """Append completed teacher archives without re-encoding the frozen base corpus."""
    meta=json.loads((args.out/'episodes.json').read_text())
    old=np.load(args.out/'rows.npz');labels=old['labels'].tolist();seeds=old['seeds'].tolist();cohorts=old['cohorts'].tolist()
    b=np.load(args.out/'boards.npy',mmap_mode='r+');s=np.load(args.out/'scalars.npy',mmap_mode='r+')
    seen={g['archive'] for g in meta['games']}
    excluded={g['seed'] for g in json.loads(args.audit_manifest.read_text())}
    if set(seeds)&excluded:raise ValueError('updated external audit overlaps prepared training rows')
    meta['excluded_audit_seeds']=sorted(excluded);n=meta['rows']
    for mp in sorted((args.supplement/'frames').glob('*/*/meta.json')):
        if str(mp.parent) in seen:continue
        m=json.loads(mp.read_text());seed=int(m['task']['seed'])
        if seed in excluded:continue
        label=mp.parent.name
        group=('pool_p9k' if 'pool' in label else 'day_original_conveyor' if 'day10' in label else 'night_original_conveyor')
        frames=read_frames(frame_path(mp.parent));live=[i for i,f in enumerate(frames) if not f.get('terminal')]
        picked=np.linspace(0,len(live)-1,min(len(live),args.frames_per_game),dtype=int)
        if n+len(picked)>len(b):raise ValueError('supplement exceeds reserved memmap capacity')
        ctx=m['context'];cv=context_vector(ctx['multiplier'],ctx['roster'])
        for index in picked:
            board,scalar=encode_frame(frames[live[index]],2);b[n]=board;s[n]=np.concatenate([scalar.numpy(),cv]);n+=1
            labels.append(float(m['outcome']['won']));seeds.append(seed);cohorts.append(group)
        meta['games'].append(dict(cohort=group,seed=seed,archive=str(mp.parent)))
        item=meta['summary'].setdefault(group,dict(games=0,wins=0,rows=0));item['games']+=1;item['wins']+=int(m['outcome']['won']);item['rows']+=len(picked)
    b.flush();s.flush();meta['rows']=n
    np.savez(args.out/'rows.npz',labels=np.array(labels,np.float32),seeds=np.array(seeds,np.int64),cohorts=np.array(cohorts))
    (args.out/'episodes.json').write_text(json.dumps(meta,indent=1));print(json.dumps(meta['summary'],indent=1),flush=True)


class Branches:
    def __init__(self,paths,excluded):
        self.parts=[];self.ends=[];self.fit=[];self.val=[];self.seeds=set();offset=0
        for path in paths:
            d=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            if d['scalars'].shape[1]!=SCALARS+CONTEXT:raise ValueError(f'{path}: not a legacy context dump')
            cfg=d.get('config',{});task=cfg.get('task_id','')
            if not task.startswith('day_l7_'):raise ValueError(f'{path}: v1 conversion only supports known L7 day branches')
            seed=d['seed'].numpy();mask=~np.isin(seed,list(excluded));stop=np.array([held(v) for v in seed])
            self.fit.extend((np.flatnonzero(mask&~stop)+offset).tolist());self.val.extend((np.flatnonzero(mask&stop)+offset).tolist())
            self.seeds.update(int(v) for v in seed[mask]);self.parts.append(d);offset+=len(seed);self.ends.append(offset)
        self.fit=np.array(self.fit);self.val=np.array(self.val)
        if not len(self.fit) or not len(self.val):raise ValueError('branch seed split is empty')

    def batch(self,indices):
        boards=torch.empty((len(indices),92,6,10),dtype=torch.float32);scalars=torch.zeros((len(indices),SCALARS+6+CONTEXT));won=torch.empty(len(indices))
        part_ids=np.searchsorted(self.ends,indices,side='right')
        for part_id in np.unique(part_ids):
            dst=np.flatnonzero(part_ids==part_id);src=indices[dst]-(self.ends[part_id-1] if part_id else 0)
            d=self.parts[part_id];old=d['scalars'][src]
            boards[dst]=d['boards'][src].float();scalars[dst,:SCALARS]=old[:,:SCALARS]
            scalars[dst,SCALARS+6:]=old[:,SCALARS:];won[dst]=d['won'][src].float()
        return boards,scalars,won


def initialize(path):
    old=load_value_net(path);new=ValueNet(width=old.conv[0].out_channels,context=True,encoder_version=2)
    state=old.state_dict();weight=state['scalar.0.weight'];expanded=torch.zeros((len(weight),SCALARS+6+CONTEXT))
    expanded[:,:SCALARS]=weight[:,:SCALARS];expanded[:,SCALARS+6:]=weight[:,SCALARS:]
    state['scalar.0.weight']=expanded;new.load_state_dict(state);return new


def train(args):
    if args.device=='cuda':
        # The task explicitly reserves GPU training for after dw5i finishes.
        result=__import__('subprocess').run(['pgrep','-f','train_rollout_policy.py'],capture_output=True,text=True)
        if result.returncode==0:raise RuntimeError('another student trainer is active; wait before GPU value training')
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    meta=json.loads((args.out/'episodes.json').read_text());rows=np.load(args.out/'rows.npz');n=meta['rows']
    boards=np.load(args.out/'boards.npy',mmap_mode='r');scalars=np.load(args.out/'scalars.npy',mmap_mode='r')
    seeds=rows['seeds'];labels=rows['labels'];stop=np.array([held(s) for s in seeds])
    fit_idx=np.flatnonzero(~stop);val_idx=np.flatnonzero(stop)
    if not len(fit_idx) or not len(val_idx):raise ValueError('episode seed split is empty')
    paths=[f for p in map(Path,args.branches.split(',')) for f in (sorted(p.glob('*.pt')) if p.is_dir() else [p])]
    branches=Branches(paths,set(meta['excluded_audit_seeds']))
    net=initialize(args.init).to(args.device);opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    def batch(branch_indices,episode_indices):
        b,s,y=branches.batch(branch_indices)
        if len(episode_indices):
            b=torch.cat([b,torch.from_numpy(np.array(boards[episode_indices],dtype=np.float32))]);s=torch.cat([s,torch.from_numpy(np.array(scalars[episode_indices]))]);y=torch.cat([y,torch.from_numpy(labels[episode_indices])])
        return b.to(args.device),s.to(args.device),y.to(args.device)
    val_b=rng.choice(branches.val,min(len(branches.val),8192),replace=False)
    val_e=rng.choice(val_idx,min(len(val_idx),8192),replace=False)
    best=float('inf');patience=0;history=[]
    for epoch in range(args.epochs):
        net.train();losses=[]
        nb=int(args.batch_size*args.branch_fraction);ne=args.batch_size-nb
        for _ in range(args.steps):
            b,s,y=batch(rng.choice(branches.fit,nb),rng.choice(fit_idx,ne))
            logits=net(b,s);loss=F.binary_cross_entropy_with_logits(logits,y)
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        net.eval();validation={}
        with torch.inference_mode():
            for name,indices in [('branch',val_b),('episode',val_e)]:
                total=count=0
                for ii in range(0,len(indices),args.batch_size):
                    pick=indices[ii:ii+args.batch_size];empty=np.array([],dtype=np.int64)
                    b,s,y=batch(pick,empty) if name=='branch' else batch(empty,pick)
                    total+=F.binary_cross_entropy_with_logits(net(b,s),y,reduction='sum').item();count+=len(y)
                validation[name]=total/count
        val_loss=args.branch_fraction*validation['branch']+(1-args.branch_fraction)*validation['episode']
        entry=dict(epoch=epoch,fit=float(np.mean(losses)),validation=val_loss,validation_components=validation)
        history.append(entry);print(entry,flush=True)
        if val_loss<best-1e-4:
            best=val_loss;patience=0
            used=sorted(branches.seeds|set(map(int,seeds)))
            # Includes stopping seeds: no later audit may treat them as untouched.
            checkpoint=dict(state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                            encoder=ENCODER_V2,width=net.conv[0].out_channels,context=True,trained_on=used,
                            source='L7 branches + multi-zone episode outcomes',init=str(args.init),
                            audit_manifest=str(args.audit_manifest),branches=[str(p) for p in paths],epoch=epoch,
                            code_commit=__import__('subprocess').check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                            branch_fit_rows=len(branches.fit),branch_validation_rows=len(branches.val),
                            episode_fit_rows=len(fit_idx),episode_validation_rows=len(val_idx))
            torch.save(checkpoint,args.out/'model.pt')
        else:
            patience+=1
        (args.out/'training.json').write_text(json.dumps(dict(history=history,config=vars(args)|{'out':str(args.out),'data':str(args.data),'audit_manifest':str(args.audit_manifest),'supplement':str(args.supplement),'init':str(args.init)},
            branch_fit_rows=len(branches.fit),branch_validation_rows=len(branches.val),episode_fit_rows=len(fit_idx),episode_validation_rows=len(val_idx),episode_summary=meta['summary']),indent=1))
        if patience>=4:break


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','append','train'])
    p.add_argument('--data',type=Path,required=True);p.add_argument('--audit-manifest',type=Path,required=True)
    p.add_argument('--supplement',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--games-per-cohort',type=int,default=512);p.add_argument('--frames-per-game',type=int,default=32)
    p.add_argument('--branches',default='/home/newbiexvwu/PvZ-search/artifacts/hard/branch/z13_384.pt,/home/newbiexvwu/PvZ-search/artifacts/hard/branch3/z15d_shards')
    p.add_argument('--init',type=Path,default=Path('/home/newbiexvwu/PvZ-search/artifacts/value_net/v3/model.pt'))
    p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=24);p.add_argument('--steps',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=512);p.add_argument('--branch-fraction',type=float,default=.75)
    p.add_argument('--lr',type=float,default=1e-4)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args)
    elif args.stage=='append':append_supplement(args)
    else:train(args)
if __name__=='__main__':main()
