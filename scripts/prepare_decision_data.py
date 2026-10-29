"""Interruptible task162 preparation: frozen row plan/deck split, then sparse public tokens.

Runs as a resource-limited queue job, never on the agent/startup critical path.
Only source metadata is read for the split. Token construction decodes each selected
archive once and keeps the exact r42 source_row inputs; no engine replay or new labels.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import signal
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
import numpy as np
from gpu_window_runner import write_json


def split(args, stopped):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    progress = out / 'progress.json'
    state = json.loads(progress.read_text()) if progress.exists() else dict(segment=0, mask_row=0)
    connection = sqlite3.connect(out / 'sources.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS sources (source TEXT PRIMARY KEY, rows BLOB, n INTEGER)')
    with np.load(args.pack, allow_pickle=False) as index:
        names = index['segments'].tolist()
        total = int(index['rows'])
    for si in range(state['segment'], len(names)):
        with np.load(args.pack.parent / names[si], allow_pickle=False) as segment:
            sources, archive, rows = segment['source'].tolist(), segment['source_archive'], segment['source_row']
            counts = np.bincount(archive, minlength=len(sources))
        at = 0
        for source, count in zip(sources, counts):
            selected = np.asarray(rows[at:at+count], np.int32); at += count
            if not len(selected):
                continue
            raw = args.data / source
            frozen = out / 'raw' / source
            frozen.parent.mkdir(parents=True, exist_ok=True)
            if not frozen.exists():
                os.link(raw, frozen)
            meta = raw.parent / 'meta.json'
            if meta.exists() and not (frozen.parent / 'meta.json').exists():
                shutil.copyfile(meta, frozen.parent / 'meta.json')
            connection.execute('INSERT OR IGNORE INTO sources VALUES (?, ?, ?)', (source, selected.tobytes(), len(selected)))
        connection.commit()
        state['segment'] = si + 1; write_json(progress, state)
        if stopped():
            raise SystemExit(75)
    kinds = np.load(args.merge_dir / 'packet_col0.npy', mmap_mode='r')
    from eval_window_policy import deck_keys
    groups = defaultdict(dict)
    position = 0
    for source, count in connection.execute('SELECT source,n FROM sources ORDER BY source'):
        meta = out / 'raw' / Path(source).parent / 'meta.json'
        record = json.loads(meta.read_text()) if meta.exists() else {}
        level = record.get('context', {}).get('level')
        # Conveyor hands are not static decks; leave them in the seed-only split.
        if level is not None and level not in (10, 20, 30, 40, 50) and record.get('task', {}).get('deck'):
            from eval_window_policy import terrain_for_level
            key = deck_keys(kinds[position:position+1])[0].tobytes().hex()
            terrain = terrain_for_level(level)
            groups[terrain][key] = groups[terrain].get(key, 0) + count
        position += count
        if stopped():
            raise SystemExit(75)
    if position != total:
        raise ValueError('source plan row count differs from frozen r42 pack')
    rng = random.Random(162)
    selected = []
    for terrain, decks in sorted(groups.items()):
        candidates = sorted(decks); rng.shuffle(candidates)
        budget = sum(decks.values()) * .1
        mass = 0
        for key in candidates:
            if len(decks) > 1 and mass + decks[key] <= budget and key not in selected:
                selected.append(key); mass += decks[key]
    if not selected:
        raise ValueError('no complete deck holdout available in this corpus')
    mask_path = out / 'unseen.npy'
    mask = (np.load(mask_path, mmap_mode='r+') if mask_path.exists() else
            np.lib.format.open_memmap(mask_path, mode='w+', dtype=bool, shape=(total,)))
    heldout = np.array([bytes.fromhex(key) for key in selected], dtype='V10')
    for start in range(state['mask_row'], total, 65536):
        stop = min(total, start + 65536)
        mask[start:stop] = np.isin(deck_keys(kinds[start:stop]), heldout)
        mask.flush()
        state['mask_row'] = stop; write_json(progress, state)
        if stopped():
            raise SystemExit(75)
    write_json(out / 'ready.json', dict(rows=total, deck_seed=162, heldout_decks=selected,
        unseen_rows=int(mask.sum()), mask=str(mask_path), source_plan='sources.sqlite',
        rule='complete deck multisets excluded from training, plus original seed holdout'))


def tokens(args, stopped):
    from pvz_frame_store import read_frames
    from pvz_student_tokens import frame_tokens
    from pvz_research_data import FIELDS
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    progress = out / 'progress.json'
    state = json.loads(progress.read_text()) if progress.exists() else dict(archives=0, rows=0, tokens=0, features=0,
        sizes={name: 8 if name in ('offsets', 'feature_offsets') else 0 for name in FIELDS})
    handles = {}
    for name in FIELDS:
        path = out / (name + '.bin')
        if not path.exists():
            path.touch()
            if name in ('offsets', 'feature_offsets'):
                path.write_bytes(np.zeros(1, np.int64).tobytes())
        handle = path.open('r+b'); handle.truncate(state['sizes'][name]); handle.seek(0, 2)
        handles[name] = handle
    connection = sqlite3.connect(args.sources / 'sources.sqlite')
    def save():
        for name, handle in handles.items():
            handle.flush(); state['sizes'][name] = handle.tell()
        write_json(progress, state)
    for source, raw_rows in connection.execute('SELECT source,rows FROM sources ORDER BY source LIMIT -1 OFFSET ?', (state['archives'],)):
        frames = read_frames(args.sources / 'raw' / source)
        for row in np.frombuffer(raw_rows, np.int32):
            sample = frame_tokens(frames[int(row) - 1])
            if np.any(sample['target_indices'] != -1):
                raise ValueError('compact r42 sidecar cannot silently discard rich target links')
            features = sample['features']
            rr, cc = np.nonzero(features)
            arrays = dict(ids=sample['ids'], counts=np.bincount(rr, minlength=len(features)).astype(np.uint8),
                columns=cc.astype(np.uint8), values=features[rr, cc], packet_index=sample['packet_index'],
                cell_index=sample['cell_index'], terrain=sample['terrain'])
            state['tokens'] += len(features); state['features'] += len(rr); state['rows'] += 1
            arrays.update(offsets=np.array([state['tokens']], np.int64), feature_offsets=np.array([state['features']], np.int64))
            for name, array in arrays.items():
                np.asarray(array, dtype=FIELDS[name][0]).tofile(handles[name])
        state['archives'] += 1
        save()
        if stopped():
            raise SystemExit(75)
    for handle in handles.values():
        handle.close()
    write_json(out / 'ready.json', dict(rows=state['rows'], archives=state['archives'], tokens=state['tokens'],
        upstream='2a866783', input='original compact public frames for exact r42 source_row-1', sparse_float32=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['split', 'tokens'])
    parser.add_argument('--pack', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--merge-dir', type=Path)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    interrupted = False
    def stop(*_):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGUSR1, stop); signal.signal(signal.SIGTERM, stop)
    (split if args.stage == 'split' else tokens)(args, lambda: interrupted)


if __name__ == '__main__':
    main()
