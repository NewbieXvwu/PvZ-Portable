"""Turn task78's frozen root labels into encoder-v2 decision shards.

One shard per (archive, horizon). Rows are the non-terminal leaves the value net actually
scores; terminal leaves stay in the label record for audit but are dropped here, exactly as
the v5 sparse export did. The stopping flag is computed from the world identity (new
regions) or v5's seed hash (old-region substitute) before any label is read, so training and
stopping identities cannot cross.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from task78_data import sha
from task78_collect import paths
from value_teacher_feasibility import write_json
from value_widen_audit import seed_key


def identity_stop(world):
    import hashlib
    return int(hashlib.sha256(json.dumps(world, sort_keys=True).encode()).hexdigest()[:8], 16) % 10 == 0


def v5_stop(seed):
    return int(seed_key(int(seed))[:8], 16) % 10 == 0


def region(group):
    return 'roof' if group.startswith('roof') else 'fog'


def stop_set(archives, old):
    """Stopping identities, fixed before any label is read.

    New regions use SHA256(world identity)[:8] % 10 == 0. The pre-registration adds a floor:
    if a region ends with fewer than two stopping archives, the region's hash-smallest
    archives join the stopping set, so validation always covers at least two worlds.
    """
    import hashlib
    flags = {i: (v5_stop(a['world']['seed']) if old else identity_stop(a['world']))
             for i, a in enumerate(archives)}
    added = {}
    if not old:
        for name in ('fog', 'roof'):
            index = [i for i, a in enumerate(archives) if region(a['group']) == name]
            if sum(flags[i] for i in index) >= 2:
                continue
            order = sorted(index, key=lambda i: hashlib.sha256(
                json.dumps(archives[i]['world'], sort_keys=True).encode()).hexdigest())
            added[name] = []
            for i in order:
                if sum(flags[j] for j in index) >= 2:
                    break
                flags[i] = True
                added[name].append(i)
    return flags, added


def shard(index, horizon, row, stop, output):
    import torch
    from pvz_value_net import encode_frame
    _, labels = paths(ROOT/'artifacts/task78/sources.json')
    directory = labels/f'archive_{index:02d}/h{horizon}'
    if not (directory/'complete.json').exists():
        raise ValueError(f'incomplete labels: {directory}')
    boards, scalars, legacy, won, positions, candidates, packets = [], [], [], [], [], [], []
    terminal = 0
    for root in sorted(directory.glob('root_*')):
        root_record = json.loads((root/'root.json').read_text())
        context = torch.tensor(root_record['context'], dtype=torch.float32)
        position = int(root.name.split('_')[1])
        for path in sorted(root.glob('candidate_[0-9]*.json')):
            record = json.loads(path.read_text())
            frame = record['leaf_frame']
            if frame.get('terminal'):
                terminal += 1
                continue
            outcome = record['outcome']
            if outcome.get('terminal') is not True or type(outcome.get('won')) is not bool:
                raise ValueError(f'{path}: no terminal label')
            board, scalars_v2 = encode_frame(frame, 2)
            _, scalars_v1 = encode_frame(frame, 1)
            boards.append(board.half())
            scalars.append(torch.cat([scalars_v2, context]))
            legacy.append(torch.cat([scalars_v1, context]))
            won.append(float(outcome['won']))
            positions.append(position)
            candidates.append(int(record['candidate']))
            packets.append(int(record['action'].get('packet', -1)))
    if not boards:
        return dict(archive_index=index, group=row['group'], horizon=horizon, rows=0,
                    wins=0, terminal_dropped=terminal, skipped='all leaves terminal')
    n = len(boards)
    target = output/f'archive_{index:02d}_h{horizon}.pt'
    tmp = target.with_suffix('.pt.tmp')
    torch.save(dict(
        boards=torch.stack(boards), scalars=torch.stack(legacy),
        scalars_v2=torch.stack(scalars), won=torch.tensor(won),
        seed=torch.full((n,), int(row['world']['seed']), dtype=torch.long),
        position=torch.tensor(positions), candidate=torch.tensor(candidates),
        packet=torch.tensor(packets), kind=torch.zeros(n, dtype=torch.long),
        stop=torch.full((n,), bool(stop)),
        config=dict(task_id=f"task78_{row['group']}", group=row['group'], horizon=horizon,
                    archive=row['archive'], world=row['world'],
                    roots=[r['frame'] for r in row['roots']],
                    label_channel='sample_root full-leaf, common future, teacher terminal continuation',
                    terminal_rows_dropped=terminal)), tmp)
    tmp.replace(target)
    return dict(archive_index=index, group=row['group'], horizon=horizon, world=row['world'],
                stop=bool(stop), rows=n, wins=int(sum(won)), terminal_dropped=terminal,
                path=str(target), sha256=sha(target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest', nargs='?', default=str(ROOT/'artifacts/task78/sources.json'))
    parser.add_argument('--only', type=int, help='single archive index (smoke test)')
    args = parser.parse_args()
    manifest = Path(args.manifest)
    source = json.loads(manifest.read_text())
    replay_path, _ = paths(manifest)
    replay = json.loads(replay_path.read_text())
    if not replay['passed'] or replay['source_manifest_sha256'] != sha(manifest):
        raise ValueError('frozen archives must pass exact root replay first')
    old = manifest.name.startswith('old')
    output = ROOT/('artifacts/task78/data_old' if old else 'artifacts/task78/data')
    output.mkdir(parents=True, exist_ok=True)
    stops, added = stop_set(source['archives'], old)
    shards = []
    for index, row in enumerate(source['archives']):
        if args.only is not None and index != args.only:
            continue
        stop = stops[index]
        for horizon in ((2500,) if old else (2500, 4000, 6000)):
            result = shard(index, horizon, row, stop, output)
            shards.append(result)
            print(json.dumps(result), flush=True)
    total = dict(shards=len(shards), rows=sum(s['rows'] for s in shards),
                 wins=sum(s['wins'] for s in shards),
                 terminal_dropped=sum(s['terminal_dropped'] for s in shards),
                 skipped=sum(1 for s in shards if s.get('skipped')))
    by_group = {}
    for s in shards:
        item = by_group.setdefault(s['group'], dict(rows=0, wins=0, archives=set(), stop_archives=set()))
        item['rows'] += s['rows']; item['wins'] += s['wins']
        item['archives'].add(s['archive_index'])
        if s.get('stop'):
            item['stop_archives'].add(s['archive_index'])
    summary = {g: dict(rows=v['rows'], wins=v['wins'], archives=len(v['archives']),
                       stop_archives=len(v['stop_archives'])) for g, v in by_group.items()}
    name = 'export_old.json' if old else 'export.json'
    write_json(ROOT/f'artifacts/task78/{name}',
               dict(source_manifest=str(manifest), source_manifest_sha256=sha(manifest),
                    stopping_archives=sorted(i for i, v in stops.items() if v),
                    stopping_floor_added=added,
                    total=total, by_group=summary, shards=shards))
    print(json.dumps(dict(total=total, by_group=summary), indent=1))


if __name__ == '__main__':
    main()
