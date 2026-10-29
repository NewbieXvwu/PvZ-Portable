"""Compare existing c53 token loading with the block-shuffled shared-memory reader.

Reuses a private frozen pack and merge; no repacking or persistent loader metadata.
Prints initial-epoch startup separately from subsequent complete-epoch throughput.
"""
import argparse
from contextlib import closing
from functools import partial
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
import torch
from pvz_token_loader import TokenBlockLoader
import train_rollout_policy as trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--recipe', type=Path, help='optionally run the real trainer after measuring loading')
    p.add_argument('--train-out', type=Path)
    args = p.parse_args()
    torch.set_num_threads(1)
    loaded = trainer.read_pack(args.pack, args.data)
    store = loaded[-1]
    rows = len(loaded[2])
    indices = np.arange(rows)
    result = dict(rows=rows, batch=args.batch, workers=args.workers, prefetch=args.prefetch,
                  token_bytes=store.nbytes, disk_added_bytes=0,
                  cache='OS page cache warm; no cold-disk or GPU throughput claim')
    for name, order, budget in (
            ('previous_random', np.random.default_rng(163).permutation(indices), 0),
            ('previous_sequential', indices, 16*1024**2)):
        store.cache.clear()
        store.cached_bytes = 0
        store.cache_bytes = budget
        started = time.perf_counter()
        for at in range(0, rows, args.batch):
            store.batch(order[at:at+args.batch])
        seconds = time.perf_counter()-started
        result[name] = dict(seconds=seconds, rows_s=rows/seconds)

    with TokenBlockLoader(store, args.workers, args.prefetch) as loader:
        def epoch(seed):
            started = time.perf_counter()
            with closing(loader.batches(indices, args.batch, seed)) as batches:
                for _idx, _tokens in batches:
                    pass
            return time.perf_counter()-started
        first = epoch(163)
        timings = [epoch(164+i) for i in range(args.epochs)]
    total = sum(timings)
    result['block_prefetch_first_epoch'] = dict(seconds=first, rows_s=rows/first)
    result['block_prefetch_steady'] = dict(epochs=args.epochs, seconds=total,
        rows=rows*args.epochs, rows_s=rows*args.epochs/total,
        epoch_seconds=timings, projected_23m_seconds=23_000_000*total/(rows*args.epochs))
    print(json.dumps(result, indent=2), flush=True)
    if args.recipe:
        directory = args.train_out or args.data.parent/'train-prefetch'
        trainer.Metrics = partial(trainer.Metrics, directory=directory/'telemetry')
        sys.argv = ['train_rollout_policy.py', '--recipe', str(args.recipe), '--out', str(directory),
                    '--pack', str(args.pack), '--skip-validation']
        trainer.main()


if __name__ == '__main__':
    main()
