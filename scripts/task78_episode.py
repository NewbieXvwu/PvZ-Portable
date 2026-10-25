"""Rebuild the old-region episode corpus from read-only qc archives.

v5's own decision shards and L7 branch dumps are gone from this machine (see the task
document's 问题 section), so the "old corpus" half of the v6 mix is reconstructed here from
the same qc frame store with the same encoder and the same whole-game outcome label. This
is a substitute, not v5's original corpus, and the conclusion must say so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
import numpy as np
from pvz_frame_store import frame_path, read_frames
from pvz_value_net import CONTEXT, SCALARS, context_vector, encode_frame
from value_widen_audit import cohort, seed_key

MAIN = Path('/home/newbiexvwu/PvZ-search')
RESERVED = set(range(78000000, 78002000)) | set(range(78100000, 78102000))
COHORTS = ('day_d7', 'night_n9_n10', 'pool_p9k')


def held(seed):
    return int(seed_key(int(seed))[:8], 16) % 10 == 0


def order(path):
    return hashlib.sha256(str(path).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games-per-cohort', type=int, default=96)
    parser.add_argument('--frames-per-game', type=int, default=32)
    parser.add_argument('--out', default=str(ROOT/'artifacts/task78/episodes'))
    args = parser.parse_args()
    source = MAIN/'artifacts/student/frames_v2'
    groups = defaultdict(list)
    for seed_dir in sorted(source.iterdir()):
        if not seed_dir.is_dir() or not seed_dir.name.startswith('seed_'):
            continue
        for archive in seed_dir.iterdir():
            if not archive.is_dir() or not (archive/'meta.json').is_file():
                continue
            name = cohort(archive.name)
            if name in COHORTS:
                groups[name].append(archive)
    selected = []
    for name in COHORTS:
        candidates = sorted(groups[name], key=order)
        if len(candidates) < args.games_per_cohort:
            raise ValueError(f'{name}: need {args.games_per_cohort} archives, found {len(candidates)}')
        selected.extend((name, p) for p in candidates[:args.games_per_cohort])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    capacity = len(selected)*args.frames_per_game
    boards = np.lib.format.open_memmap(out/'boards.npy', mode='w+', dtype=np.float16,
                                       shape=(capacity, 92, 6, 10))
    scalars = np.lib.format.open_memmap(out/'scalars.npy', mode='w+', dtype=np.float32,
                                        shape=(capacity, SCALARS+6+CONTEXT))
    labels, seeds, cohorts, stop, index = [], [], [], [], 0
    summary = defaultdict(lambda: dict(games=0, wins=0, rows=0))
    for name, archive in selected:
        meta = json.loads((archive/'meta.json').read_text())
        world_seed = int(meta['task']['seed'])
        if world_seed in RESERVED:
            raise ValueError(f'{archive}: reserved evaluation seed')
        frames = read_frames(frame_path(archive))
        live = [i for i, f in enumerate(frames) if not f.get('terminal')]
        picked = np.linspace(0, len(live)-1, min(len(live), args.frames_per_game), dtype=int)
        ctx = meta['context']
        cv = context_vector(ctx['multiplier'], ctx['roster'])
        for i in picked:
            board, scalar = encode_frame(frames[live[i]], 2)
            boards[index] = board
            scalars[index] = np.concatenate([scalar.numpy(), cv])
            labels.append(float(meta['outcome']['won']))
            seeds.append(world_seed)
            cohorts.append(name)
            stop.append(held(world_seed))
            index += 1
        item = summary[name]
        item['games'] += 1; item['wins'] += int(meta['outcome']['won']); item['rows'] += len(picked)
    boards.flush(); scalars.flush()
    np.savez(out/'rows.npz', labels=np.array(labels, np.float32),
             seeds=np.array(seeds, np.int64), cohorts=np.array(cohorts),
             stop=np.array(stop, bool))
    meta = dict(games=[dict(cohort=c, archive=str(p)) for c, p in selected], rows=index,
                capacity=capacity, frames_per_game=args.frames_per_game,
                summary=summary, source='qc frames_v2, whole-game outcome label',
                substitute_for='v5 original corpus (unrecoverable on this machine)')
    (out/'episodes.json').write_text(json.dumps(meta, indent=1)+'\n')
    print(json.dumps(dict(rows=index, summary=summary), indent=1))


if __name__ == '__main__':
    main()
