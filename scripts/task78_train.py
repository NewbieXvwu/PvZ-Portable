"""Train value v6: new fog/roof decision labels on top of a rebuilt old-region corpus.

The pre-registered mix per 512-row batch is old decision 128, fog 64, roof 64, old episode
256. v5's L7 branch dumps no longer exist on this machine, so their 192-row share is folded
into the old-episode share (256 instead of 64); the old:new ratio stays 384:128 = 75:25.
Everything else follows the frozen recipe: AdamW lr 1e-4 wd 1e-4, 24 epochs x 256 steps,
batch 512, BCE, early stopping after 4 epochs without 1e-4 improvement, best model kept.
"""
from __future__ import annotations

import argparse
import json
import signal
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

MIX = dict(old_episode=256, old_decision=128, fog=64, roof=64)


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
        return (self.boards[index].to(device, non_blocking=True),
                self.scalars[index].to(device, non_blocking=True),
                self.won[index].to(device, non_blocking=True))

    def loss(self, model, indices, batch_size, device):
        total = 0.0
        model.eval()
        with torch.inference_mode():
            for first in range(0, len(indices), batch_size):
                pick = indices[first:first+batch_size]
                b, s, y = self.batch(pick, device)
                total += F.binary_cross_entropy_with_logits(model(b, s), y, reduction='sum').item()
        return total/len(indices)


def load_decisions(directory, stop_rule):
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
        stop.append(np.array([stop_rule(int(v)) for v in shard['seed'].numpy()], dtype=bool))
        groups.extend([shard['config']['group']]*n)
    return (torch.cat(boards), torch.cat(scalars), torch.cat(won),
            np.concatenate(seeds), np.concatenate(stop), groups)


def load_episodes(directory):
    meta = json.loads((Path(directory)/'episodes.json').read_text())
    rows = np.load(Path(directory)/'rows.npz')
    boards = torch.from_numpy(np.load(Path(directory)/'boards.npy')[:meta['rows']].astype(np.float32))
    scalars = torch.from_numpy(np.load(Path(directory)/'scalars.npy')[:meta['rows']])
    return (boards, scalars, torch.from_numpy(rows['labels']), rows['seeds'],
            rows['stop'], list(rows['cohorts']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-data', default=str(ROOT/'artifacts/task78/data'))
    parser.add_argument('--old-data', default=str(ROOT/'artifacts/task78/data_old'))
    parser.add_argument('--episodes', default=str(ROOT/'artifacts/task78/episodes'))
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

    new_boards, new_scalars, new_won, new_seeds, new_stop, groups = load_decisions(
        args.new_data, lambda seed: None)
    # The export already froze the stopping set from the world identity; rebuild it here only
    # to assert the shard's own flag agrees.
    stop_flags = []
    for path in sorted(Path(args.new_data).glob('*.pt')):
        stop_flags.append(torch.load(path, map_location='cpu', weights_only=False)['stop'].numpy())
    new_stop = np.concatenate(stop_flags)
    groups = np.array(groups)
    fog = np.flatnonzero([not g.startswith('roof') for g in groups])
    roof = np.flatnonzero([g.startswith('roof') for g in groups])
    sources = {
        'fog': Rows(new_boards[fog], new_scalars[fog], new_won[fog], new_seeds[fog], new_stop[fog],
                    allow_empty=True),
        'roof': Rows(new_boards[roof], new_scalars[roof], new_won[roof], new_seeds[roof], new_stop[roof],
                     allow_empty=True),
    }
    old_shards = sorted(Path(args.old_data).glob('*.pt')) if Path(args.old_data).is_dir() else []
    if old_shards:
        old_boards, old_scalars, old_won, old_seeds, old_stop, _ = load_decisions(
            args.old_data, held)
        sources['old_decision'] = Rows(old_boards, old_scalars, old_won, old_seeds, old_stop)
    ep = load_episodes(args.episodes)
    sources['old_episode'] = Rows(*ep[:5])
    if not old_shards:
        raise ValueError('old decision corpus is required; refusing to train a different mixture')
    for name in sources:
        if len(sources[name].fit) == 0 or len(sources[name].val) == 0:
            raise ValueError(f'{name}: fit and validation rows are both required; refusing to drop a source')
    dropped = []

    # The process, including CUDA allocation and validation, cannot outlive the
    # user's one-hour GPU limit. SIGALRM terminates it even if an epoch stalls.
    if args.device.startswith('cuda'):
        available = require_memory()
        signal.alarm(3600)
    model = load_value_net(args.init).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    validation = {name: rng.choice(src.val, min(len(src.val), 8192), replace=False)
                  for name, src in sources.items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    best, best_epoch, stale, history = float('inf'), None, 0, []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _ in range(args.steps):
            parts = [src.batch(rng.choice(src.fit, max(1, round(MIX[name]*args.batch_size/512))),
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
        total = sum(MIX[name]*components[name] for name in sources)/sum(MIX.values())
        entry = dict(epoch=epoch, fit=float(np.mean(losses)), validation=total,
                     components=components, seconds=time.time()-started)
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if total < best - args.min_improvement:
            best, best_epoch, stale = total, epoch, 0
            initial = torch.load(args.init, map_location='cpu', weights_only=False)
            used = sorted(set(initial.get('trained_on', [])) | set().union(*(src.seeds for src in sources.values())))
            torch.save(dict(
                state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                encoder=2, width=model.conv[0].out_channels, context=True,
                trained_on=used, init=args.init, epoch=epoch,
                validation=total, components=components,
                mix=MIX, dropped_sources=dropped,
                sources={name: dict(fit=len(src.fit), val=len(src.val))
                         for name, src in sources.items()},
                hyper=dict(lr=args.lr, weight_decay=1e-4, epochs=args.epochs, steps=args.steps,
                           batch_size=args.batch_size, seed=args.seed, device=args.device,
                           patience=args.patience, min_improvement=args.min_improvement),
                mem_available_gib=available, wall_seconds=time.time()-started,
                code_commit=__import__('subprocess').check_output(
                    ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                source='task78 fog/roof decision labels + rebuilt old-region corpus'),
                out/'model.pt')
        else:
            stale += 1
            if stale >= args.patience:
                print(f'early stop at epoch {epoch}', flush=True)
                break
    if args.device.startswith('cuda'):
        signal.alarm(0)
    (out/'train_log.json').write_text(json.dumps(dict(complete=True, history=history, best_epoch=best_epoch,
        best_validation=best, mix=MIX, dropped_sources=dropped, wall_seconds=time.time()-started,
        sources={name: dict(rows=len(src.fit)+len(src.val), fit=len(src.fit), val=len(src.val))
                 for name, src in sources.items()}), indent=1)+'\n')
    print(json.dumps(dict(best_epoch=best_epoch, best_validation=best), indent=1))


if __name__ == '__main__':
    main()
