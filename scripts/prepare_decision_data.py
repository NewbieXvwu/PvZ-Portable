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
            if raw.exists():
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
    mask_path = out / 'unseen.npy'
    mask = (np.load(mask_path, mmap_mode='r+') if mask_path.exists() else
            np.lib.format.open_memmap(mask_path, mode='w+', dtype=bool, shape=(total,)))
    groups = defaultdict(dict)
    position = 0
    for source, count in connection.execute('SELECT source,n FROM sources ORDER BY source'):
        raw = out / 'raw' / source
        # read_segments sorts source parents before merging; insertion order is only segment order.
        # Archives whose frames are retired keep their row numbers but are never trained or scored.
        if not raw.exists():
            mask[position:position+count] = True
            mask.flush()
            position += count
            continue
        meta = raw.parent / 'meta.json'
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
    heldout = np.array([bytes.fromhex(key) for key in selected], dtype='V10')
    for start in range(state['mask_row'], total, 65536):
        stop = min(total, start + 65536)
        mask[start:stop] = mask[start:stop] | np.isin(deck_keys(kinds[start:stop]), heldout)
        mask.flush()
        state['mask_row'] = stop; write_json(progress, state)
        if stopped():
            raise SystemExit(75)
    write_json(out / 'ready.json', dict(rows=total, deck_seed=162, heldout_decks=selected,
        unseen_rows=int(mask.sum()), mask=str(mask_path), source_plan='sources.sqlite',
        rule='complete deck multisets excluded from training, plus original seed holdout; '
             'retired-frame archives excluded from both sides'))


def tokens(args, stopped):
    from pvz_frame_store import read_frames
    from pvz_student_tokens import frame_tokens
    from pvz_research_data import FIELDS, keep_training_rows
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    progress = out / 'progress.json'
    state = json.loads(progress.read_text()) if progress.exists() else dict(archives=0, rows=0, tokens=0, features=0, stored=0,
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
    budget_path = getattr(args, 'budget', None)
    budget = json.loads(budget_path.read_text()) if budget_path else {}
    if budget_path:
        from pvz_frame_store import archive_seed
        import random
        seeds = sorted({archive_seed(Path(source).parent.parent.name) for source, in connection.execute('SELECT source FROM sources')})
        random.Random(0).shuffle(seeds)
        validation_seeds = set(seeds[:int(len(seeds)*.1)])
        unseen = np.load(args.sources/'unseen.npy', mmap_mode='r')
    preview_rows = budget.get('preview_rows', 0)
    heldout_rows, excluded_sources = 0, []
    if preview_rows:
        position = 0
        for source, n in connection.execute('SELECT source,n FROM sources ORDER BY source'):
            if not (args.sources/'raw'/source).exists():
                excluded_sources.append(source)
            elif archive_seed(Path(source).parent.parent.name) in validation_seeds:
                heldout_rows += n
            else:
                heldout_rows += int(np.count_nonzero(unseen[position:position+n]))
            position += n
    def save():
        for name, handle in handles.items():
            handle.flush(); state['sizes'][name] = handle.tell()
        write_json(progress, state)
    for source, raw_rows in connection.execute('SELECT source,rows FROM sources ORDER BY source LIMIT -1 OFFSET ?', (state['archives'],)):
        rows = np.frombuffer(raw_rows, np.int32)
        selected = np.ones(len(rows), dtype=bool)
        if budget_path:
            indices = np.arange(state['rows'], state['rows']+len(rows))
            validation = np.array(unseen[state['rows']:state['rows']+len(rows)])
            if archive_seed(Path(source).parent.parent.name) in validation_seeds:
                validation[:] = True
            selected = keep_training_rows(indices, budget['train_row_fraction']) & ~validation
            if preview_rows and heldout_rows:
                selected |= keep_training_rows(indices, min(1, preview_rows/heldout_rows)) & validation
        if not (args.sources/'raw'/source).exists():
            # Retired archive: no tokens here, but its rows keep the pack numbering intact.
            state['rows'] += len(rows); state['archives'] += 1
            if stopped():
                raise SystemExit(75)
            continue
        frames = read_frames(args.sources/'raw'/source) if selected.any() else None
        for row, included in zip(rows, selected):
            global_row = state['rows']
            if not included:
                state['rows'] += 1
                continue
            sample = frame_tokens(frames[int(row) - 1])
            if np.any(sample['target_indices'] != -1):
                raise ValueError('compact r42 sidecar cannot silently discard rich target links')
            if not np.array_equal(sample['cell_index'], np.arange(2,56)):
                raise ValueError('compact sidecar requires the fixed cell-token order')
            kept = np.r_[0:2,56:len(sample['ids'])]
            ids = sample['ids'][kept]
            features = sample['features'][kept]
            packet_index = sample['packet_index'].copy()
            packet_index[packet_index>=2] -= 54
            rr, cc = np.nonzero(features)
            arrays = dict(ids=ids, counts=np.bincount(rr, minlength=len(features)).astype(np.uint8),
                columns=cc.astype(np.uint8), values=features[rr, cc], packet_index=packet_index,
                terrain=sample['terrain'], stored_row=np.array([global_row], np.int32))
            state['stored'] += 1
            state['tokens'] += len(features); state['features'] += len(rr); state['rows'] += 1
            arrays.update(offsets=np.array([state['tokens']], np.int64), feature_offsets=np.array([state['features']], np.int64))
            for name, array in arrays.items():
                np.asarray(array, dtype=FIELDS[name][0]).tofile(handles[name])
        state['archives'] += 1
        save()
        size = sum(state['sizes'].values())
        if budget_path and size > budget['sidecar_max_bytes']:
            raise RuntimeError(f'token cache reached its disk budget: {size} bytes')
        if budget_path:
            host_free = min(shutil.disk_usage(path).free for path in ('/mnt/c','/mnt/d') if Path(path).exists())
            if host_free < 12*(1<<30):
                raise RuntimeError(f'host disk reserve reached: {host_free} bytes free; token construction stopped')
        if stopped():
            raise SystemExit(75)
    for handle in handles.values():
        handle.close()
    write_json(out / 'ready.json', dict(rows=state['rows'], archives=state['archives'], tokens=state['tokens'],
        upstream='2a866783', input='original compact public frames for exact r42 source_row-1', sparse_float32=True,
        bytes=sum(state['sizes'].values()), stored_rows=state['stored'], schema=3,
        preview_rows=preview_rows, heldout_rows=heldout_rows, excluded_sources=excluded_sources,
        evaluation='uncached rows stream from source archives; release cache after all decision jobs complete'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['split', 'tokens'])
    parser.add_argument('--pack', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--merge-dir', type=Path)
    parser.add_argument('--sources', type=Path)
    parser.add_argument('--budget', type=Path)
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
