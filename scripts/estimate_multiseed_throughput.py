"""Estimate task92c queue service from historical timings; launch no jobs."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(pipeline, state_file, experiments):
    state = json.loads(state_file.read_text())
    single_path = experiments / 'task92/gpu_single/result.json'
    double_path = experiments / 'task92/gpu_parallel2/result.json'
    single = json.loads(single_path.read_text())['workers'][0]
    double = json.loads(double_path.read_text())['workers']
    speedup = 2 * single['training_seconds'] / max(row['training_seconds'] for row in double)
    load_single, load_double = single['load_seconds'], max(row['load_seconds'] for row in double)
    sources = {str(p): sha(p) for p in (state_file, single_path, double_path)}
    selection_wall, selection_mtimes = {}, {}
    selection_root = experiments / 'task92b/v2/selection'
    for folder in selection_root.iterdir():
        config, log = folder / 'config.json', folder / 'driver.out'
        settings = json.loads(config.read_text())
        model = Path(settings['rollout_policy']).stem
        # Real per-task driver duration from config publication to the final
        # driver log write; excludes interpreter import before config publication.
        selection_wall[model] = selection_wall.get(model, 0) + max(0, log.stat().st_mtime - config.stat().st_mtime)
        selection_mtimes[str(folder)] = dict(config_mtime_ns=config.stat().st_mtime_ns,
                                              driver_mtime_ns=log.stat().st_mtime_ns)
        sources[str(config)], sources[str(log)] = sha(config), sha(log)
    pairs = [sum(values) for values in itertools.combinations(selection_wall.values(), 2)]
    selection_seconds = statistics.mean(pairs)
    rows = []
    for index in range(12, 18):
        path = pipeline / f'round_{index:06d}/complete.json'
        original = json.loads(path.read_text())
        sources[str(path)] = sha(path)
        pack, train, gate = [original[name + '_finished'] - original[name + '_started']
                             for name in ('pack', 'train', 'evaluation')]
        groups = next(row['groups'] for row in state['log'] if row['round'] == index)
        collection = sum(group['collection_finished'] - group['collection_started'] for group in groups)
        train_parallel = load_double + 2 * max(0, train - load_single) / speedup
        rows.append(dict(round=index, collection_seconds=collection, pack_seconds=pack,
                         single_train_seconds=train, gate_seconds=gate,
                         parallel_train_seconds=train_parallel, selection_seconds=selection_seconds,
                         parallel_service_seconds=pack + train_parallel + selection_seconds + gate,
                         serial_fallback_service_seconds=pack + 2 * train + selection_seconds + gate))
    arrival, finish = 0., 0.
    for row in rows:
        arrival += row['collection_seconds']
        start = max(finish, arrival)
        row['estimated_wait_seconds'] = start - arrival
        finish = start + row['parallel_service_seconds']
    return dict(schema_version=1, rows=rows,
        assumptions='Apply 256-step dual/single optimizer speedup to historical train time after subtracting '
                    'the measured single load; add parallel load. Reuse historical packing/gate durations. '
                    'Selection is average of all pairs of fixed task92b four-model per-task driver durations, '
                    'two game workers and 380 games/model. Future panel has <=384 games; import overhead '
                    'and random-deck preparation are not included. Active collection durations omit downtime '
                    'between groups; simulate a FIFO starting empty before round12, not actual historical queue state.',
        dual_training_speedup=speedup, load_single_seconds=load_single, load_double_seconds=load_double,
        selection_wall_by_model=selection_wall, selection_pair_range_seconds=[min(pairs), max(pairs)],
        selection_mtime_measurements=selection_mtimes,
        average_collection_seconds=statistics.mean(row['collection_seconds'] for row in rows),
        average_parallel_service_seconds=statistics.mean(row['parallel_service_seconds'] for row in rows),
        service_to_arrival_ratio=sum(row['parallel_service_seconds'] for row in rows) / sum(row['collection_seconds'] for row in rows),
        sources_sha256=sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', type=Path, default=Path.home() / 'pvz-logs/curriculum_state.pipeline')
    parser.add_argument('--state', type=Path, default=Path.home() / 'pvz-logs/curriculum_state.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = estimate(args.pipeline, args.state, ROOT / 'artifacts')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'sources_sha256'}, indent=2))
