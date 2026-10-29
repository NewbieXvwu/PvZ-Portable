"""Freeze two CNN jobs from a production round without rebuilding/replaying any data.

Only small index headers and file metadata are read. Immutable pack segments and an
already-completed merged cache are hardlinked into research-owned paths, so production
pruning/replacing its next generation cannot invalidate these jobs or copy 14 GB.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
from pvz_recipe import load_recipe, write_recipe, training_row_options
from pvz_training_pack import pack_options
from gpu_window_runner import write_json


def signature(paths):
    return hashlib.sha256(json.dumps([(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
                                     for path in paths]).encode()).hexdigest()


def freeze_pack(recipe, destination):
    data = Path(recipe['data'])
    wanted = pack_options(**training_row_options(recipe))
    wanted.pop('sources', None)
    indexes = []
    for path in sorted((data / 'packs').glob('*.npz')):
        with np.load(path, allow_pickle=False) as packed:
            meta = json.loads(str(packed['meta']))
            if {k: v for k, v in meta.items() if k != 'sources'} == wanted:
                indexes.append(path)
    if len(indexes) != 1:
        raise ValueError(f'expected one frozen round pack, found {indexes}')
    source = indexes[0]
    destination.mkdir(parents=True, exist_ok=False)
    target = destination / source.name
    shutil.copyfile(source, target)
    with np.load(source, allow_pickle=False) as packed:
        names = packed['segments'].tolist()
        rows = int(packed['rows'])
    original_paths, frozen_paths = [], []
    for name in names:
        original, frozen = source.parent / name, destination / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        os.link(original, frozen)
        original_paths.append(original); frozen_paths.append(frozen)
    merge = destination.parent / 'merge'
    old_merge = data / '.merge' / 'shared-v1'
    old_marker = old_merge / 'complete'
    reused = old_marker.exists() and old_marker.read_text() == signature(original_paths)
    if reused:
        # Completed arrays are immutable; trainer weights are copied privately on load.
        merge.mkdir()
        for member in old_merge.glob('*.npy'):
            os.link(member, merge / member.name)
        (merge / 'complete').write_text(signature(frozen_paths))
    write_json(destination.parent / 'corpus.json', dict(source=str(source), pack=str(target), rows=rows,
        segments=len(names), merge_reused=reused, merge_dir=str(merge)))
    return target, merge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True, help='the frozen production round recipe')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int, help='separate short demonstration cohort, not a full training result')
    args = parser.parse_args()
    recipe = load_recipe(args.recipe, 'training')
    if recipe['obs'] != 'v2timer':
        raise ValueError('examples require the v2timer pack')
    args.state = args.state.resolve()
    args.state.mkdir(parents=True, exist_ok=True)
    pack, merge = freeze_pack(recipe, args.state / 'corpus' / 'packs')
    for width in (64, 128):
        name = f'r42_cnn{width}' + ('_smoke' if args.smoke_steps else '')
        out = args.state / 'runs' / name
        variant = dict(recipe, width=width, arch='cnn', out=str(out), init=None, seed=1,
                       source_row_report=False, validation_batch_size=256)
        if args.smoke_steps:
            variant['max_steps'] = args.smoke_steps
        path = args.state / 'recipes' / (name + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        write_recipe(path, variant)
        job = dict(id=name, recipe=str(path), out=str(out), host_gib=5, gpu_gib=4, slice_seconds=900,
            command=[sys.executable, str(ROOT / 'scripts/train_rollout_policy.py'), '--recipe', str(path),
                     '--pack', str(pack), '--merge-dir', str(merge), '--checkpoint-seconds', '30', '--skip-validation'],
            evaluate=[sys.executable, str(ROOT / 'scripts/eval_window_policy.py'), '--recipe', str(path),
                      '--pack', str(pack), '--merge-dir', str(merge), '--out', str(out)])
        write_json(args.state / 'queue' / (name + '.json'), job)
    print(json.dumps(dict(state=str(args.state), pack=str(pack), merge=str(merge))))


if __name__ == '__main__':
    main()
