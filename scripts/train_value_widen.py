"""Train value v4 on legacy L7 branches and cross-zone result-labelled archives.

Memory is bounded by one episode and one minibatch: archive features are cached in
on-disk numpy memmaps; branch tensors use torch mmap, never concatenated. Audit
seeds are excluded globally, and early stopping is split by environment seed.
"""
from __future__ import annotations
import argparse
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'scripts')]
from pvz_frame_store import archive_seed,frame_path,read_frames
from pvz_value_net import SCALARS, CONTEXT, ENCODER_V2, ValueNet, encode_frame, context_vector, load_value_net
from value_widen_audit import cohort, seed_key


def held(seed):
    return int(seed_key(int(seed))[:8],16)%10==0


def require_training_memory():
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    if available < 10*1024**3:
        raise RuntimeError(f'value training requires MemAvailable >= 10 GiB; got {available/1024**3:.2f} GiB')
    print(f'MemAvailable: {available/1024**3:.2f} GiB (gate 10 GiB)',flush=True)


def gather(parts,ends,key,indices,dtype):
    shape=parts[0][key].shape[1:] if key in parts[0] else ()
    out=torch.empty((len(indices),*shape),dtype=dtype)
    part_ids=np.searchsorted(ends,indices,side='right')
    for part_id in np.unique(part_ids):
        dst=np.flatnonzero(part_ids==part_id);src=indices[dst]-(ends[part_id-1] if part_id else 0)
        part=parts[part_id]
        if key in part:out[dst]=part[key][torch.from_numpy(src)].to(dtype)
        elif key=='kind':out[dst]=0 # Legacy L7 dumps contain leaves only.
        elif key=='packet':out[dst]=-2
        else:raise KeyError(key)
    return out


def score_rows(model,data,indices,batch_size,device):
    prediction=torch.empty(len(indices))
    model=model.to(device).eval()
    with torch.inference_mode():
        for first in range(0,len(indices),batch_size):
            pick=indices[first:first+batch_size];b,s,_=data.batch(pick)
            prediction[first:first+len(pick)]=torch.sigmoid(model(b.to(device),s.to(device))).cpu()
    return prediction


def prepare(args):
    excluded={g['seed'] for g in json.loads(args.audit_manifest.read_text())}
    groups=defaultdict(lambda:defaultdict(list))
    for sd in args.data.iterdir():
        if not sd.is_dir() or not sd.name.startswith('seed_'):continue
        seed=archive_seed(sd.name)
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

    def rows(self,indices):
        return {k:gather(self.parts,self.ends,k,indices,torch.float32 if k=='won' else torch.long)
                for k in ('won','seed','position','candidate','packet','kind')}


class Decisions:
    """Decision-level dumps (value_ranking_check.py --dump-shards) as encoder-v2 training rows.

    One dump per seed; each row is (board, encoder-v2 scalars, result of that run, seed, position,
    candidate, packet, kind). Neighbours differ by the move and by the random future, so the label
    answers the question the search asks, not "which game is this". Only dumps whose task_id starts
    with ``prefix`` are accepted, so a day/pool mix cannot slip in by accident.
    """
    def __init__(self,paths,excluded,prefix):
        self.parts=[];self.ends=[];self.fit=[];self.val=[];self.seeds=set();offset=0
        for path in paths:
            d=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            task=d.get('config',{}).get('task_id','')
            if not task.startswith(prefix):raise ValueError(f'{path}: task {task!r} does not start with {prefix!r}')
            if 'scalars_v2' not in d:raise ValueError(f'{path}: dump has no scalars_v2 (recollect it)')
            if d['scalars_v2'].shape[1]!=SCALARS+6+CONTEXT:raise ValueError(f'{path}: scalars_v2 is not SCALARS+6+CONTEXT wide')
            seed=d['seed'].numpy();mask=~np.isin(seed,list(excluded));stop=np.array([held(v) for v in seed])
            self.fit.extend((np.flatnonzero(mask&~stop)+offset).tolist());self.val.extend((np.flatnonzero(mask&stop)+offset).tolist())
            self.seeds.update(int(v) for v in seed[mask]);self.parts.append(d);offset+=len(seed);self.ends.append(offset)
        self.fit=np.array(self.fit,dtype=np.int64);self.val=np.array(self.val,dtype=np.int64)
        if not len(self.fit) or not len(self.val):raise ValueError('decision seed split is empty')

    def _gather(self,key,indices,dtype):
        return gather(self.parts,self.ends,key,indices,dtype)

    def batch(self,indices):
        return (self._gather('boards',indices,torch.float32),self._gather('scalars_v2',indices,torch.float32),
                self._gather('won',indices,torch.float32))

    def rows(self,indices):
        return {k:self._gather(k,indices,torch.float32 if k=='won' else torch.long)
                for k in ('won','seed','position','candidate','packet','kind')}


def initialize(path):
    old=load_value_net(path)
    if old.encoder_version==2:
        # v4 already is the v2 architecture (scalar layer is SCALARS+6+CONTEXT wide): warm start is a copy.
        if not old.context:raise ValueError('warm start requires a context value net')
        return old
    if old.encoder_version!=1 or not old.context:raise ValueError('warm start requires a legacy context value net')
    new=ValueNet(width=old.conv[0].out_channels,context=True,encoder_version=2)
    state=old.state_dict();weight=state['scalar.0.weight'];expanded=torch.zeros((len(weight),SCALARS+6+CONTEXT))
    expanded[:,:SCALARS]=weight[:,:SCALARS];expanded[:,SCALARS+6:]=weight[:,SCALARS:]
    state['scalar.0.weight']=expanded;new.load_state_dict(state);return new


def train(args):
    require_training_memory()
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
    decisions=None
    if args.decision_data:
        dpaths=[f for p in map(Path,args.decision_data.split(',')) for f in (sorted(p.glob('*.pt')) if p.is_dir() else [p])]
        decisions=Decisions(dpaths,set(meta['excluded_audit_seeds']),args.decision_prefix)
        print(f'decision rows: fit {len(decisions.fit)}, held out {len(decisions.val)} from {len(decisions.seeds)} seeds',flush=True)
    elif args.decision_fraction>0:raise ValueError('--decision-fraction needs --decision-data')
    if decisions is not None and args.decision_fraction<=0:raise ValueError('--decision-data needs --decision-fraction > 0')
    init_seeds=set(torch.load(args.init,map_location='cpu',weights_only=False).get('trained_on',[]))
    if init_seeds&set(meta['excluded_audit_seeds']):raise ValueError('initial checkpoint saw external audit seeds')
    net=initialize(args.init).to(args.device);opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=1e-4)
    def batch(branch_indices,episode_indices,decision_indices=None):
        b,s,y=branches.batch(branch_indices)
        if decision_indices is not None and len(decision_indices):
            db,ds,dy=decisions.batch(decision_indices);b=torch.cat([b,db]);s=torch.cat([s,ds]);y=torch.cat([y,dy])
        if len(episode_indices):
            b=torch.cat([b,torch.from_numpy(np.array(boards[episode_indices],dtype=np.float32))]);s=torch.cat([s,torch.from_numpy(np.array(scalars[episode_indices]))]);y=torch.cat([y,torch.from_numpy(labels[episode_indices])])
        return b.to(args.device),s.to(args.device),y.to(args.device)
    val_b=rng.choice(branches.val,min(len(branches.val),8192),replace=False)
    val_e=rng.choice(val_idx,min(len(val_idx),8192),replace=False)
    val_d=rng.choice(decisions.val,min(len(decisions.val),8192),replace=False) if decisions is not None else None
    best=float('inf');patience=0;history=[]
    for epoch in range(args.epochs):
        net.train();losses=[]
        nd=int(args.batch_size*args.decision_fraction) if decisions is not None else 0
        nb=int((args.batch_size-nd)*args.branch_fraction);ne=args.batch_size-nd-nb
        margin_losses=[]
        for _ in range(args.steps):
            decision_indices=(rng.choice(decisions.fit,nd) if decisions is not None else None)
            b,s,y=batch(rng.choice(branches.fit,nb),rng.choice(fit_idx,ne),decision_indices)
            logits=net(b,s);loss=F.binary_cross_entropy_with_logits(logits,y)
            if decision_indices is not None and args.margin_weight>0:
                # Pairwise hinge on same-root candidates with differing outcomes. The decision
                # rows occupy the [nb:nb+nd] slice of the concatenated batch (branch, decision,
                # episode order in batch()); gate (a) measures top1-top2 gaps, so train the gap.
                rows=decisions.rows(decision_indices)
                groups={}
                for i,(sd,pos,w) in enumerate(zip(rows['seed'].tolist(),rows['position'].tolist(),rows['won'].tolist())):
                    groups.setdefault((sd,pos),[[],[]])[0 if w>0.5 else 1].append(i)
                win=[];lose=[]
                for wins,loses_ in groups.values():
                    if wins and loses_:
                        win.extend(wins*len(loses_));lose.extend([l for _ in wins for l in loses_])
                if win:
                    z=logits[nb:nb+len(decision_indices)]
                    diff=z[torch.tensor(win,device=z.device)]-z[torch.tensor(lose,device=z.device)]
                    mloss=F.relu(args.margin-diff).mean()
                    loss=loss+args.margin_weight*mloss;margin_losses.append(mloss.item())
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        net.eval();validation={}
        with torch.inference_mode():
            for name,indices in [('branch',val_b),('episode',val_e),('decision',val_d)]:
                if indices is None:continue
                total=count=0
                for ii in range(0,len(indices),args.batch_size):
                    pick=indices[ii:ii+args.batch_size];empty=np.array([],dtype=np.int64)
                    b,s,y=(batch(pick,empty) if name=='branch' else batch(empty,pick) if name=='episode'
                           else batch(empty,empty,pick))
                    total+=F.binary_cross_entropy_with_logits(net(b,s),y,reduction='sum').item();count+=len(y)
                validation[name]=total/count
        core=args.branch_fraction*validation['branch']+(1-args.branch_fraction)*validation['episode']
        val_loss=(args.decision_fraction*validation['decision']+(1-args.decision_fraction)*core
                  if decisions is not None else core)
        entry=dict(epoch=epoch,fit=float(np.mean(losses)),validation=val_loss,validation_components=validation,
                   margin=(float(np.mean(margin_losses)) if margin_losses else None))
        history.append(entry);print(entry,flush=True)
        if val_loss<best-1e-4:
            best=val_loss;patience=0
            used=sorted(init_seeds|branches.seeds|set(map(int,seeds))|(decisions.seeds if decisions is not None else set()))
            # Includes stopping seeds: no later audit may treat them as untouched.
            checkpoint=dict(state_dict={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                            encoder=ENCODER_V2,width=net.conv[0].out_channels,context=True,trained_on=used,
                            source='L7 branches + multi-zone episode outcomes',init=str(args.init),
                            audit_manifest=str(args.audit_manifest),branches=[str(p) for p in paths],epoch=epoch,
                            code_commit=__import__('subprocess').check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
                            branch_fit_rows=len(branches.fit),branch_validation_rows=len(branches.val),
                            episode_fit_rows=len(fit_idx),episode_validation_rows=len(val_idx),
                            decision_data=str(args.decision_data),decision_fraction=args.decision_fraction,
                            margin_weight=args.margin_weight,margin=args.margin,
                            decision_fit_rows=(len(decisions.fit) if decisions is not None else 0),
                            decision_validation_rows=(len(decisions.val) if decisions is not None else 0),
                            decision_seeds=(len(decisions.seeds) if decisions is not None else 0))
            torch.save(checkpoint,args.out/'model.pt')
        else:
            patience+=1
        (args.out/'training.json').write_text(json.dumps(dict(history=history,config=vars(args)|{'out':str(args.out),'data':str(args.data),'audit_manifest':str(args.audit_manifest),'supplement':str(args.supplement),'init':str(args.init)},
            branch_fit_rows=len(branches.fit),branch_validation_rows=len(branches.val),episode_fit_rows=len(fit_idx),episode_validation_rows=len(val_idx),episode_summary=meta['summary']),indent=1))
        if patience>=4:break
    if decisions is not None:
        # The number the branch corpus is judged by (train_value_branches.pair_accuracy), on the
        # held-out decision seeds, split into leaves and the board right after planting.
        from train_value_branches import pair_accuracy
        report={}
        # Report the saved best epoch, not the last early-stopping epoch. Use all
        # held-out rows in batches: row subsampling can split a candidate's futures.
        for name,path in (('v5',args.out/'model.pt'),('init',args.init)):
            model=load_value_net(path)
            cohorts={}
            for group,data in (('pool_decision',decisions),('day_l7_branch',branches)):
                indices=data.val;rows=data.rows(indices)
                pred=score_rows(model,data,indices,args.batch_size,args.device)
                cohorts[group]=dict(rows=len(indices),seeds=len(set(rows['seed'].tolist())),
                    leaves=pair_accuracy(pred,rows,(rows['kind']==0).nonzero().flatten(),args.pair_min_gap),
                    cells=pair_accuracy(pred,rows,(rows['kind']==1).nonzero().flatten(),args.pair_min_gap,by_packet=True))
            report[name]=dict(**cohorts['pool_decision'],cohorts=cohorts,checkpoint=str(path),
                              checkpoint_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
        (args.out/'pair_accuracy.json').write_text(json.dumps(report,indent=1));print(json.dumps(report,indent=1),flush=True)


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
    p.add_argument('--decision-data',help='value_ranking_check dumps or shard dirs, comma separated')
    p.add_argument('--decision-prefix',default='pool_',help='only dumps whose task_id starts with this')
    p.add_argument('--decision-fraction',type=float,default=0.0,
                   help='share of every minibatch taken from the decision dumps (0 = old behaviour)')
    p.add_argument('--pair-min-gap',type=float,default=0.25)
    p.add_argument('--margin-weight',type=float,default=0.0,
                   help='weight of the same-root win/lose pairwise hinge on decision rows (0 = old behaviour)')
    p.add_argument('--margin',type=float,default=1.0,help='required logit gap between winner and loser logits')
    args=p.parse_args()
    if args.stage=='prepare':prepare(args)
    elif args.stage=='append':append_supplement(args)
    else:train(args)
if __name__=='__main__':main()
