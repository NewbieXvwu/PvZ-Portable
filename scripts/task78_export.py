"""Turn task78's frozen root labels into encoder-v2 decision shards.

One shard per (archive, horizon). Rows are the non-terminal leaves the value net actually
scores; terminal leaves are kept in the label record for audit but dropped here, exactly as
the v5 sparse export did. The stopping flag is computed from the world identity before any
label is read, so training and stopping identities cannot cross.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from task78_data import sha
from value_teacher_feasibility import write_json


def stop_flag(world):
    identity = json.dumps(world, sort_keys=True)
    return int(sha_bytes(identity)[:8], 16) % 10 == 0


def sha_bytes(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    manifest = ROOT/'artifacts/task78/sources.json'
    source = json.loads(manifest.read_text())
    replay = json.loads((ROOT/'artifacts/task78/replay.json').read_text())
    if not replay['passed'] or replay['source_manifest_sha256'] != sha(manifest):
        raise ValueError('frozen archives must pass exact root replay first')
    import torch
    from pvz_value_net import encode_frame
    output = ROOT/'artifacts/task78/data'
    output.mkdir(parents=True, exist_ok=True)
    shards = []
    for index, row in enumerate(source['archives']):
        stop = stop_flag(row['world'])
        for horizon in (2500, 4000, 6000):
            directory = ROOT/f'artifacts/task78/labels/archive_{index:02d}/h{horizon}'
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
                raise ValueError(f'{directory}: no non-terminal value rows')
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
                config=dict(task_id=f"task78_{row['group']}", group=row['group'],
                            horizon=horizon, archive=row['archive'], world=row['world'],
                            roots=[r['frame'] for r in row['roots']],
                            label_channel='sample_root full-leaf, common future, teacher terminal continuation',
                            terminal_rows_dropped=terminal)), tmp)
            tmp.replace(target)
            shards.append(dict(archive_index=index, group=row['group'], horizon=horizon,
                               world=row['world'], stop=stop, rows=n, wins=int(sum(won)),
                               terminal_dropped=terminal, path=str(target), sha256=sha(target)))
            print(json.dumps(shards[-1]), flush=True)
    total = dict(shards=len(shards), rows=sum(s['rows'] for s in shards),
                 wins=sum(s['wins'] for s in shards),
                 terminal_dropped=sum(s['terminal_dropped'] for s in shards))
    by_group = {}
    for s in shards:
        item = by_group.setdefault(s['group'], dict(rows=0, wins=0, archives=set(), stop_archives=set()))
        item['rows'] += s['rows']; item['wins'] += s['wins']
        item['archives'].add(s['archive_index'])
        item['stop_archives'].add(s['archive_index']) if s['stop'] else None
    summary = {g: dict(rows=v['rows'], wins=v['wins'], archives=len(v['archives']),
                       stop_archives=len(v['stop_archives'])) for g, v in by_group.items()}
    write_json(ROOT/'artifacts/task78/export.json',
               dict(source_manifest_sha256=sha(manifest), total=total, by_group=summary, shards=shards))
    print(json.dumps(dict(total=total, by_group=summary), indent=1))


if __name__ == '__main__':
    main()
