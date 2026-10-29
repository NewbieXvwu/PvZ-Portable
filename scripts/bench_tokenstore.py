"""Pack a few completed deploy18 attachments in a private directory and time CPU use.

Example (run at nice 10 with a memory limit and one BLAS/OpenMP thread):
  python scripts/bench_tokenstore.py --out /private/tokenstore-sample \
    --archives /production/seed_*/label/frames.npz
Sources are read-only hard links. No replay, production cache writes or GPU work.
The output directory must be new. Results go to stdout, not a storage sidecar.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
import torch
from functools import partial
from pack_training_rows import pack
from pvz_recipe import write_recipe
from pvz_training_pack import pack_options, archive_fingerprint
import train_rollout_policy as trainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archives', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--lookups', type=int, default=4096)
    p.add_argument('--steps', type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    data = args.out/'data'
    labels, source_bytes = set(), 0
    for path in args.archives:
        folder = data/path.parent.parent.name/path.parent.name
        folder.mkdir(parents=True, exist_ok=True)
        labels.add(path.parent.name)
        for name in ('frames.npz', 'tokens_v3.npz', 'meta.json'):
            source = path.parent/name
            if source.is_file():
                os.link(source, folder/name)
                if name == 'tokens_v3.npz':
                    source_bytes += source.stat().st_size
    options = pack_options(labels, obs='v3', token_input='obs_v3', packet_head='identity',
                           sources=archive_fingerprint(data, labels, 'obs_v3'))
    started = time.perf_counter()
    index = pack(data, options, precompute_workers=1)
    pack_seconds = time.perf_counter()-started
    started = time.perf_counter()
    loaded = trainer.read_pack(index, data)
    merge_seconds = time.perf_counter()-started
    store = loaded[-1]
    rows = len(loaded[2])
    started = time.perf_counter()
    reused = trainer.read_pack(index, data)
    reopen_seconds = time.perf_counter()-started
    del reused

    def measure(indices, cache_bytes):
        store.cache.clear()
        store.cached_bytes = 0
        store.cache_bytes = cache_bytes
        started = time.perf_counter()
        for at in range(0, len(indices), 16):
            store.batch(indices[at:at+16])
        seconds = time.perf_counter()-started
        return dict(rows=len(indices), seconds=seconds, rows_s=len(indices)/seconds)

    rng = random.Random(163)
    random_rows = [rng.randrange(rows) for _ in range(args.lookups)]
    random_load = measure(random_rows, 0)
    sequential_load = measure(list(range(rows)), 16*1024**2)
    token_bytes = store.nbytes
    with np.load(index) as stored:
        segments = [index.parent/name for name in stored['segments'].tolist()]
    pack_bytes = index.stat().st_size + sum(path.stat().st_size for path in segments)
    merge_bytes = sum(path.stat().st_size for path in (data/'.merge/shared-v1').glob('*.npy'))
    del store, loaded
    recipe = args.out/'recipe.json'
    write_recipe(recipe, dict(schema_version=1, kind='training', data=str(data.resolve()),
        out=str((args.out/'train').resolve()), labels=sorted(labels), obs='v3', token_input='obs_v3',
        packet_head='identity', width=128, network=dict(architecture='token', layers=2, heads=4, ff_width=512),
        epochs=1, max_steps=args.steps, batch_size=8, validation_batch_size=8, seed=163,
        validation=dict(seed=0, fraction=.25)))
    losses = []
    # Keep telemetry and its retention policy inside this experiment as well.
    trainer.Metrics = partial(trainer.Metrics, directory=args.out/'metrics')
    sys.argv = [str(ROOT/'scripts/train_rollout_policy.py'), '--recipe', str(recipe),
                '--pack', str(index), '--skip-validation']
    started = time.perf_counter()
    trainer.main(step_observer=lambda _step, loss: losses.append(loss))
    training_wall = time.perf_counter()-started
    saved = torch.load(args.out/'train/model.pt', map_location='cpu', weights_only=False)
    seconds = saved['history'][-1]['seconds']
    report = dict(archives=[str(path) for path in args.archives], rows=rows,
        original_attachment_bytes=source_bytes, token_bytes=token_bytes,
        token_bytes_per_row=token_bytes/rows, pack_bytes=pack_bytes, merge_bytes=merge_bytes,
        token_projection_23m_bytes=23_000_000*token_bytes/rows,
        pack_seconds=pack_seconds, pack_rows_s=rows/pack_seconds,
        initial_merge_seconds=merge_seconds, reopen_seconds=reopen_seconds,
        random_no_decoded_cache=random_load, sequential_16mib_cache=sequential_load,
        cache_note='OS page cache warm; random lookup decoded cache disabled; includes CPU padding/tensors; no cold-disk claim',
        training=dict(device='cpu', threads=1, width=128, layers=2, heads=4, batch=8,
                      steps=len(losses), finite_losses=bool(np.isfinite(losses).all()),
                      first_loss=losses[0], last_loss=losses[-1], seconds=seconds,
                      rows_s=len(losses)*8/seconds, total_wall_seconds=training_wall))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
