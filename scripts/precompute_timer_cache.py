#!/usr/bin/env python3
"""Build only v2sleep/v2timer episode caches, leaving packs and controller state alone.

Run this separately from deployment at nice 19. Existing v2 rows become zero-padding
views where possible, then v2timer computes just the four timer channels. No corpus
replay, training, or verification pass is performed.
"""
import argparse
from datetime import datetime, timedelta
from functools import partial
import json
import multiprocessing as mp
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

from pvz_frame_store import cache_episode, frame_paths
from pvz_recipe import load_recipe


def build_one(path, cache_dir, read_cache_dir):
    # Production is still on v2: materialize the cheap v2sleep view first so the
    # timer encoder can reuse it. New archives with observed sleep fields use the
    # ordinary sleep encoder, not zero padding.
    cache_episode(path, cache_dir, 'v2sleep', read_cache_dir)
    cache_episode(path, cache_dir, 'v2timer', read_cache_dir)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--memory-kib', type=int, default=2500000,
                        help='address-space cap per process, inherited by workers')
    parser.add_argument('--progress', type=Path, required=True)
    args = parser.parse_args()
    limit = args.memory_kib * 1024
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    recipe = load_recipe(args.recipe, 'training')
    if recipe['obs'] != 'v2timer':
        parser.error('--recipe must select v2timer')
    data = Path(recipe['data'])
    cache = Path(recipe['cache_dir'])
    read_cache = Path(recipe['read_cache_dir']) if recipe['read_cache_dir'] else None
    started = time.monotonic()
    paths = frame_paths(data)
    total, done = len(paths), 0

    def progress(status, error=None):
        elapsed = time.monotonic() - started
        remaining = elapsed * (total - done) / done if done else None
        record = dict(status=status, done=done, total=total, workers=args.workers,
                      elapsed_seconds=round(elapsed, 1),
                      eta=(datetime.now().astimezone() + timedelta(seconds=remaining)).isoformat()
                      if remaining is not None else None,
                      data=str(data), cache_dir=str(cache), recipe=str(args.recipe))
        if error is not None:
            record['error'] = str(error)
        text = json.dumps(record, ensure_ascii=False)
        args.progress.write_text(text + '\n')
        print(text, flush=True)

    progress('running')
    worker = partial(build_one, cache_dir=cache, read_cache_dir=read_cache)
    last = time.monotonic()
    try:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            for count in pool.imap_unordered(worker, paths, chunksize=4):
                done += count
                if time.monotonic() - last >= 30:
                    progress('running')
                    last = time.monotonic()
    except BaseException as error:
        progress('failed', error)
        raise
    progress('complete')


if __name__ == '__main__':
    main()
