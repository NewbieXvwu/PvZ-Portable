"""Train value v6 on the complete v5 corpus plus audited fog/roof decision labels.

The original five-source recipe is required. Missing branches or episode caches cannot
be replaced by a different mixture. Original tensors stay memory mapped; only batches
are converted to float32. Restored paths are bound to v5's declared inputs by corpus.json.
"""
from __future__ import annotations

import argparse
import json
import signal
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
import torch
import torch.nn.functional as F
from pvz_value_net import load_value_net
from value_widen_audit import seed_key
from task78_data import sha

MIX = dict(old_episode=64, old_branch=192, old_decision=128, fog=64, roof=64)
# Diagnostic fallback only: the v5 branch/decision shards and episode cache are absent on
# this machine, so their 320/512 share is carried by the same-source rebuilt episode cache.
# 75/12.5/12.5 old/new exposure is preserved; a model trained this way is never the
# preregistered candidate and says so in its checkpoint.
MIX_DIAGNOSTIC = dict(old_episode=384, fog=64, roof=64)
BASELINE_SHA = '8a636d337c03dcf3037f42ce41f79abc5e0b898eb5a2b39d9a00ad3b9d88c3b4'


def held(seed):
    return int(seed_key(int(seed))[:8], 16) % 10 == 0


def require_memory():
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:')))*1024
    if available < 10*1024**3:
        raise RuntimeError(f'value training requires MemAvailable >= 10 GiB; got {available/1024**3:.2f} GiB')
    return available/1024**3


class Rows:
    """One training source: tensors plus fit/validation index arrays."""

    def __init__(self, boards, scalars, won, seeds, stop, allow_empty=False):
        self.boards = boards
        self.scalars = scalars
        self.won = won
        stop = np.asarray(stop, dtype=bool)
        self.fit = np.flatnonzero(~stop)
        self.val = np.flatnonzero(stop)
        self.seeds = {int(v) for v in np.unique(np.asarray(seeds))}
        if not allow_empty and (not len(self.fit) or not len(self.val)):
            raise ValueError(f'source split is empty: fit {len(self.fit)} val {len(self.val)}')

    def batch(self, indices, device):
        index = torch.from_numpy(np.asarray(indices))
        return tuple(t[index].to(device,dtype=torch.float32,non_blocking=True)
                     for t in (self.boards,self.scalars,self.won))

    def sample(self, rng, count):
        return rng.choice(self.fit,count)

    def loss(self, model, indices, batch_size, device):
        total = 0.0
        model.eval()
        with torch.inference_mode():
            for first in range(0, len(indices), batch_size):
                pick = indices[first:first+batch_size]
                b, s, y = self.batch(pick, device)
                total += F.binary_cross_entropy_with_logits(model(b, s), y, reduction='sum').item()
        return total/len(indices)


def load_decisions(directory, stop_rule=None):
    boards, scalars, won, seeds, stop, groups = [], [], [], [], [], []
    paths = sorted(Path(directory).glob('*.pt'))
    if not paths:
        raise ValueError(f'no decision shards in {directory}')
    for path in paths:
        shard = torch.load(path, map_location='cpu', weights_only=False)
        n = len(shard['won'])
        boards.append(shard['boards'].float())
        scalars.append(shard['scalars_v2'].float())
        won.append(shard['won'].float())
        seeds.append(shard['seed'].numpy())
        stop.append(shard['stop'].numpy() if stop_rule is None else
                    np.array([stop_rule(int(v)) for v in shard['seed'].numpy()],dtype=bool))
        groups.extend([shard['config']['group']]*n)
    return (torch.cat(boards), torch.cat(scalars), torch.cat(won),
            np.concatenate(seeds), np.concatenate(stop), groups)


def load_episodes(directory):
    meta = json.loads((Path(directory)/'episodes.json').read_text())
    rows = np.load(Path(directory)/'rows.npz')
    boards = torch.from_numpy(np.load(Path(directory)/'boards.npy',mmap_mode='c')[:meta['rows']])
    scalars = torch.from_numpy(np.load(Path(directory)/'scalars.npy',mmap_mode='c')[:meta['rows']])
    stop=rows['stop'] if 'stop' in rows else np.array([held(s) for s in rows['seeds']],dtype=bool)
    return (boards, scalars, torch.from_numpy(rows['labels']), rows['seeds'],
            stop,list(rows['cohorts']))


class MappedRows:
    """Adapt the original v5 mmap loaders without materialising their full boards."""
    loss=Rows.loss

    def __init__(self,data):
        self.data=data
        self.fit,self.val,self.seeds=data.fit,data.val,data.seeds

    def sample(self,rng,count):
        return rng.choice(self.fit,count)

    def batch(self,indices,device):
        return tuple(t.to(device,dtype=torch.float32) for t in self.data.batch(indices))


def balanced_weights(keys,indices):
    """Equal group/world/root/horizon mass, without consulting outcome labels."""
    from collections import Counter,defaultdict
    children=defaultdict(set)
    counts=Counter(tuple(keys[i]) for i in indices)
    for key in counts:
        for depth,value in enumerate(key):
            children[key[:depth]].add(value)
    weights=[]
    for index in indices:
        key=tuple(keys[index]);weight=1/counts[key]
        for depth in range(len(key)):
            weight/=len(children[key[:depth]])
        weights.append(weight)
    weights=np.asarray(weights,dtype=np.float64)
    return weights/weights.sum()


class BalancedRows(Rows):
    def __init__(self,*args,keys):
        super().__init__(*args)
        self.probabilities=balanced_weights(keys,self.fit)

    def sample(self,rng,count):
        return rng.choice(self.fit,count,p=self.probabilities)


def new_sources(directory):
    """Only the complete, independently audited export is eligible for training."""
    from task78_export import stop_set,collection_horizons
    directory=Path(directory)
    manifest_path=directory.parent/'export.json'
    export=json.loads(manifest_path.read_text())
    if export.get('complete') is not True:
        raise ValueError('a partial/smoke export cannot be used for v6 training')
    source_path=Path(export['source_manifest'])
    if sha(source_path)!=export['source_manifest_sha256']:
        raise ValueError('new source manifest changed after export')
    archives=json.loads(source_path.read_text())['archives']
    from collections import Counter
    if Counter(row['group'] for row in archives)!=dict(fog_demo_l31=8,fog_demo_l32=8,
            fog_demo_l33_cactus=8,fog_demo_l33_cattail=8,fog_l31_human=8,roof_l41=24):
        raise ValueError('new corpus does not cover the frozen 64-archive cohort design')
    expected={(i,h) for i,row in enumerate(archives) for h in collection_horizons(row)}
    shards=export['shards']
    if (len(shards)!=len(expected) or
            {(s['archive_index'],s['horizon']) for s in shards}!=expected):
        raise ValueError('new export does not cover every planned archive/horizon')
    paths={Path(s['path']) for s in shards if s['rows']}
    if paths!=set(directory.glob('*.pt')):
        raise ValueError('new shard directory contains missing or unexpected files')
    stops,_=stop_set(archives,False)
    keys=[]
    worlds={}
    for path in sorted(paths):
        record=next(s for s in shards if s.get('path')==str(path))
        if sha(path)!=record['sha256']:
            raise ValueError(f'new shard changed: {path}')
        part=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        index=record['archive_index'];row=archives[index]
        if (not bool((part['stop']==stops[index]).all()) or
                part['config']['world']!=row['world'] or part['config']['group']!=row['group']
                or len(part['won'])!=record['rows']):
            raise ValueError(f'new shard world/split/rows differs: {path}')
        worlds[json.dumps(row['world'],sort_keys=True)]=row['world']
        keys.extend((row['group'],json.dumps(row['world'],sort_keys=True),int(position),record['horizon'])
                    for position in part['position'])
    boards,scalars,won,seeds,stop,groups=load_decisions(directory)
    groups=np.asarray(groups)
    sources={}
    for name in ('fog','roof'):
        pick=np.flatnonzero([g.startswith('roof')==(name=='roof') for g in groups])
        sources[name]=BalancedRows(boards[pick],scalars[pick],won[pick],seeds[pick],stop[pick],
                                   keys=[keys[i] for i in pick])
    return sources,dict(export=str(manifest_path),sha256=sha(manifest_path),worlds=list(worlds.values()))


def restored_file(entry):
    path=Path(entry['path'])
    if not path.is_absolute() or not path.is_file() or sha(path)!=entry['sha256']:
        raise ValueError(f'missing, changed, or relative original corpus input: {path}')
    return path


def original_sources(init,decision_directory,episode_directory):
    """Require all original inputs, identified by source paths and content hashes.

    corpus.json beside the decision directory maps copied files to the exact original
    paths in the v5 checkpoint. All row counts must also match that checkpoint.
    """
    from train_value_widen import Branches,Decisions
    if sha(init)!=BASELINE_SHA:
        raise ValueError('v6 must start from the frozen adopted v5 checkpoint')
    initial=torch.load(init,map_location='cpu',weights_only=False)
    descriptor_path=Path(decision_directory).parent/'corpus.json'
    if not descriptor_path.is_file():
        raise FileNotFoundError(f'complete original v5 corpus descriptor required: {descriptor_path}')
    descriptor=json.loads(descriptor_path.read_text())
    if descriptor['baseline_sha256']!=BASELINE_SHA:
        raise ValueError('original corpus descriptor belongs to a different baseline')
    branch_entries=descriptor['branches']
    if (len(branch_entries)!=len(initial['branches']) or
            {x['source_path'] for x in branch_entries}!=set(initial['branches'])):
        raise ValueError('original L7 branch coverage is incomplete')
    decisions=descriptor['decision_shards']
    if descriptor['decision_source_path']!=initial['decision_data'] or len(decisions)!=initial['decision_seeds']:
        raise ValueError('original pool decision coverage is incomplete')
    decision_paths=[restored_file(x) for x in decisions]
    if set(decision_paths)!=set(Path(decision_directory).glob('*.pt')):
        raise ValueError('unexpected or missing original decision shards')
    branch_paths=[restored_file(x) for x in branch_entries]
    audit_entry=descriptor['audit_manifest']
    if audit_entry['source_path']!=initial['audit_manifest']:
        raise ValueError('original external audit manifest differs')
    audit=json.loads(restored_file(audit_entry).read_text())
    excluded={int(r['seed']) for r in audit}
    episode_files=descriptor['episode_files']
    if {Path(e['path']).name for e in episode_files}!={'episodes.json','rows.npz','boards.npy','scalars.npy'}:
        raise ValueError('original episode cache is incomplete')
    if {restored_file(e).parent for e in episode_files}!={Path(episode_directory)}:
        raise ValueError('episode cache paths differ from the frozen original descriptor')
    meta=json.loads((Path(episode_directory)/'episodes.json').read_text())
    if 'substitute_for' in meta or set(meta['excluded_audit_seeds'])!=excluded:
        raise ValueError('replacement episode caches cannot stand in for the original v5 corpus')
    sources=dict(old_branch=MappedRows(Branches(branch_paths,excluded)),
                 old_decision=MappedRows(Decisions(decision_paths,excluded,'pool_')),
                 old_episode=Rows(*load_episodes(episode_directory)[:5]))
    for name,prefix in [('old_branch','branch'),('old_decision','decision'),('old_episode','episode')]:
        data=sources[name]
        if (len(data.fit),len(data.val))!=(initial[prefix+'_fit_rows'],initial[prefix+'_validation_rows']):
            raise ValueError(f'{name}: row split differs from original v5 checkpoint')
        if not data.seeds<=set(initial['trained_on']):
            raise ValueError(f'{name}: unexpected seeds absent from original v5 training')
    return sources,initial,dict(descriptor=str(descriptor_path),sha256=sha(descriptor_path))


def reconstructed_sources(init,episode_directory):
    """Same-source rebuilt old-region episodes, explicitly not the v5 corpus."""
    if sha(init)!=BASELINE_SHA:
        raise ValueError('v6 must start from the frozen adopted v5 checkpoint')
    initial=torch.load(init,map_location='cpu',weights_only=False)
    meta=json.loads((Path(episode_directory)/'episodes.json').read_text())
    if 'substitute_for' not in meta:
        raise ValueError('reconstructed mode requires a cache marked as a substitute')
    rows=Rows(*load_episodes(episode_directory)[:5])
    provenance=dict(mode='reconstructed',episodes=str(Path(episode_directory)),
                    sha256=sha(Path(episode_directory)/'episodes.json'),
                    preregistered_corpus=False,
                    deviation=('original v5 branch, pool-decision and episode sources are '
                               'unrecoverable on this machine; their 320/512 share is carried '
                               'by the rebuilt episode cache (task78_episode.py)'))
    return dict(old_episode=rows),initial,provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', choices=['original','reconstructed'], default='original',
                        help='reconstructed = documented diagnostic fallback, not the preregistered corpus')
    parser.add_argument('--new-data', default=str(ROOT/'artifacts/task78/data'))
    parser.add_argument('--old-data', default=str(ROOT/'artifacts/task78/original_v5/decision'))
    parser.add_argument('--episodes', default=str(ROOT/'artifacts/task78/original_v5/episodes'))
    parser.add_argument('--init', default='/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt')
    parser.add_argument('--out', default=str(ROOT/'artifacts/task78/v6'))
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--min-improvement', type=float, default=1e-4)
    args = parser.parse_args()
    available = require_memory()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.corpus=='original':
        sources,initial,original_provenance=original_sources(args.init,args.old_data,args.episodes)
        mix=dict(MIX)
    else:
        sources,initial,original_provenance=reconstructed_sources(args.init,args.episodes)
        mix=dict(MIX_DIAGNOSTIC)
    added,new_provenance=new_sources(args.new_data)
    sources.update(added)
    if set(sources)!=set(mix):
        raise ValueError(f'sources {sorted(sources)} do not match the frozen mixture {sorted(mix)}')
    world_seeds={int(v) for w in new_provenance['worlds'] for v in w.values() if v is not None}
    used=sorted(set(initial.get('trained_on',[])) | world_seeds |
                set().union(*(src.seeds for src in sources.values())))
    if any(78000000<=s<78002000 or 78100000<=s<78102000 for s in used):
        raise ValueError('training or initialization overlaps a reserved evaluation seed/future')
    for name in sources:
        if len(sources[name].fit) == 0 or len(sources[name].val) == 0:
            raise ValueError(f'{name}: fit and validation rows are both required; refusing to drop a source')
    dropped = []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'model.pt').exists() or (out/'run.json').exists():
        raise ValueError('training output already contains a run; inspect its live handle before another attempt')

    # The process, including CUDA allocation and validation, cannot outlive the
    # user's one-hour GPU limit. SIGALRM terminates it even if an epoch stalls.
    if args.device.startswith('cuda'):
        available = require_memory()
        signal.alarm(3600)
    model = load_value_net(args.init).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    validation = {name: rng.choice(src.val, min(len(src.val), 8192), replace=False)
                  for name, src in sources.items()}
    started = time.time()
    run_identity=dict(pid=os.getpid(),started=time.time(),init_sha256=sha(args.init),
                      inputs=dict(original=original_provenance,new=new_provenance),mix=mix)
    (out/'run.json').write_text(json.dumps(run_identity,indent=1)+'\n')
    best, best_epoch, stale, history = float('inf'), None, 0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps):
            parts = [src.batch(src.sample(rng,max(1,round(mix[name]*args.batch_size/512))),
                               args.device) for name, src in sources.items()]
            b = torch.cat([p[0] for p in parts])
            s = torch.cat([p[1] for p in parts])
            y = torch.cat([p[2] for p in parts])
            loss = F.binary_cross_entropy_with_logits(model(b, s), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        components = {name: src.loss(model, validation[name], args.batch_size, args.device)
                      for name, src in sources.items()}
        total = sum(mix[name]*components[name] for name in sources)/sum(mix.values())
        entry = dict(epoch=epoch, fit=float(np.mean(losses)), validation=total,
                     components=components, seconds=time.time()-started)
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if total < best - args.min_improvement:
            best, best_epoch, stale = total, epoch, 0
            torch.save(dict(
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                encoder=2, width=model.conv[0].out_channels, context=True,
                trained_on=used, init=args.init, epoch=epoch,
                validation=total, components=components,
                mix=mix, dropped_sources=dropped,
                preregistered_corpus=original_provenance.get('preregistered_corpus',True),
                sources={name: dict(fit=len(src.fit), val=len(src.val))
                         for name, src in sources.items()},
                hyper=dict(lr=args.lr, weight_decay=1e-4, epochs=args.epochs, steps=args.steps,
                           batch_size=args.batch_size, seed=args.seed, device=args.device,
                           patience=args.patience, min_improvement=args.min_improvement),
                mem_available_gib=available, wall_seconds=time.time()-started,
                code_commit=__import__('subprocess').check_output(
                    ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                input_provenance=dict(original=original_provenance,new=new_provenance),
                trained_worlds=new_provenance['worlds'],
                source=('task78 fog/roof decision labels + complete original v5 corpus'
                        if args.corpus=='original' else
                        'DIAGNOSTIC: task78 fog/roof labels + rebuilt old-region episodes')),
                out/'model.pt.tmp')
            (out/'model.pt.tmp').replace(out/'model.pt')
        else:
            stale += 1
            if stale >= args.patience:
                print(f'early stop at epoch {epoch}', flush=True)
                break
        (out/'train_log.json').write_text(json.dumps(dict(complete=False,history=history,
            best_epoch=best_epoch,best_validation=best,run=run_identity),indent=1)+'\n')
    if args.device.startswith('cuda'):
        signal.alarm(0)
    (out/'train_log.json').write_text(json.dumps(dict(complete=True, history=history, best_epoch=best_epoch,
        best_validation=best, mix=mix, dropped_sources=dropped, wall_seconds=time.time()-started,
        preregistered_corpus=original_provenance.get('preregistered_corpus',True),
        sources={name: dict(rows=len(src.fit)+len(src.val), fit=len(src.fit), val=len(src.val))
                 for name, src in sources.items()}), indent=1)+'\n')
    print(json.dumps(dict(best_epoch=best_epoch, best_validation=best), indent=1))


if __name__ == '__main__':
    main()
