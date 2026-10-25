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


def collection_horizons(row, old=False):
    """Fog retains the original three horizons; the master capped roof at h2500."""
    return (2500,4000,6000) if not old and row['group'].startswith('fog') else (2500,)


def audit_labels(index, horizon, row, manifest, replay):
    """Recompute coverage/identity from raw labels before trusting complete.json."""
    from value_teacher_feasibility import group_summary
    from value_teacher_feasibility_report import validate_income
    _, labels = paths(manifest)
    directory = labels/f'archive_{index:02d}/h{horizon}'
    complete_path = directory/'complete.json'
    if not complete_path.is_file():
        raise ValueError(f'incomplete labels: {directory}')
    complete = json.loads(complete_path.read_text())
    identity = complete['identity']
    expected = dict(source_manifest_sha256=sha(manifest), source=row, horizon=horizon,
                    executable_sha256=replay['executable_sha256'])
    for name, value in expected.items():
        if identity.get(name) != value:
            raise ValueError(f'{directory}: wrong {name}')
    for name in ('teacher_source', 'label_channel'):
        filename = 'search_play.py' if name == 'teacher_source' else 'value_teacher_feasibility.py'
        if identity[name+'_sha256'] != sha(ROOT/'scripts'/filename):
            raise ValueError(f'{directory}: changed {name}')
    if set(identity['models']) != {'value', 'rollout_policy', 'static_value'}:
        raise ValueError(f'{directory}: missing frozen model identities')
    for name, value in identity['models'].items():
        if sha(value['path']) != value['sha256']:
            raise ValueError(f'{directory}: changed model {name}')
    if sha(row['frames_path']) != row['frames_sha256'] or sha(Path(row['archive'])/'meta.json') != row['meta_sha256']:
        raise ValueError(f'{directory}: source archive changed')
    expected_roots = {r['frame']: r for r in row['roots']}
    root_dirs = {int(p.name[5:]): p for p in directory.glob('root_*') if p.is_dir()}
    summaries = complete['roots']
    if (len(summaries) != len(expected_roots) or {r['frame'] for r in summaries} != set(expected_roots)
            or set(root_dirs) != set(expected_roots)):
        raise ValueError(f'{directory}: root coverage differs from source manifest')
    verified = next(r for r in replay['results'] if r['archive'] == row['archive'])
    verified_roots = {r['frame']: r for r in verified['roots']}
    artifacts = {str(complete_path): sha(complete_path)}
    output = []
    for frame, root_dir in sorted(root_dirs.items()):
        root_path = root_dir/'root.json'
        root = json.loads(root_path.read_text())
        original = expected_roots[frame]
        if (root['root_frame']['tick'], root['root_frame']['wave']) != (original['tick'], original['wave']):
            raise ValueError(f'{root_dir}: root tick/wave differs')
        if root['root_state_sha256'] != verified_roots[frame]['engine_sha256']:
            raise ValueError(f'{root_dir}: root engine state differs from replay')
        validate_income(root, root['root_frame']['tick'])
        records = [json.loads(p.read_text()) for p in sorted(root_dir.glob('candidate_[0-9]*.json'))]
        count = len(root['candidates'])
        if len(records) != count or {r['candidate'] for r in records} != set(range(count)):
            raise ValueError(f'{root_dir}: candidate coverage differs')
        for record in records:
            candidate = record['candidate']
            outcome = record['outcome']
            if (record['action'] != root['candidates'][candidate]
                    or record['common_future'] != root['common_future']
                    or record['continuation_seed_base'] != root['decision_seed']+5000):
                raise ValueError(f'{root_dir}: candidate identity differs')
            if (outcome.get('terminal') is not True or type(outcome.get('won')) is not bool
                    or outcome['result'] not in (1, 2) or outcome['won'] != (outcome['result'] == 1)):
                raise ValueError(f'{root_dir}: invalid terminal truth')
            checks = record['history_checks']
            required = ('frame', 'income', 'income_start')
            if not record['leaf_frame']['terminal']:
                required += ('engine_state',)
            if any(checks.get(k) is not True for k in required):
                raise ValueError(f'{root_dir}: incomplete history/leaf validation')
            validate_income(record, record['leaf_frame']['tick'])
            trace = root_dir/f'candidate_{candidate}_teacher.jsonl'
            actions = [json.loads(line) for line in trace.read_text().splitlines()]
            if [a['decision'] for a in actions] != list(range(outcome['decisions'])):
                raise ValueError(f'{root_dir}: incomplete teacher action trace')
            if (any(b['tick'] < a['tick'] for a,b in zip(actions, actions[1:]))
                    or actions and (actions[0]['tick'] < record['leaf_frame']['tick']
                                    or actions[-1]['tick'] > outcome['tick'])):
                raise ValueError(f'{root_dir}: invalid teacher trace timeline')
            artifacts[str(trace)] = sha(trace)
            label = root_dir/f'candidate_{candidate}.json'
            artifacts[str(label)] = sha(label)
            output.append((frame, root, record))
        recomputed = group_summary(records)
        saved = json.loads((root_dir/'summary.json').read_text())
        if any(saved.get(k) != v for k,v in recomputed.items()):
            raise ValueError(f'{root_dir}: raw labels and summary disagree')
        artifacts[str(root_path)] = sha(root_path)
    return output, artifacts


def shard(index, horizon, row, stop, output, manifest, replay):
    import torch
    from pvz_value_net import encode_frame
    records, provenance = audit_labels(index, horizon, row, manifest, replay)
    boards, scalars, legacy, won, positions, candidates, packets = [], [], [], [], [], [], []
    terminal = 0
    for position, root_record, record in records:
        context = torch.tensor(root_record['context'], dtype=torch.float32)
        frame = record['leaf_frame']
        if frame.get('terminal'):
            terminal += 1
            continue
        outcome = record['outcome']
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
                    wins=0, stop=bool(stop), world=row['world'],
                    terminal_dropped=terminal, skipped='all leaves terminal', provenance=provenance)
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
                path=str(target), sha256=sha(target), provenance=provenance)


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
        for horizon in collection_horizons(row, old):
            result = shard(index, horizon, row, stop, output, manifest, replay)
            shards.append(result)
            print(json.dumps({k:v for k,v in result.items() if k != 'provenance'}), flush=True)
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
                    complete=args.only is None, selection='all fog horizons; master roof h2500 resource amendment',
                    stopping_archives=sorted(i for i, v in stops.items() if v),
                    stopping_floor_added=added,
                    total=total, by_group=summary, shards=shards))
    print(json.dumps(dict(total=total, by_group=summary), indent=1))


if __name__ == '__main__':
    main()
