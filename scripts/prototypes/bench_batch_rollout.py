#!/usr/bin/env python3
"""Compare native serial rollout with isolated C++ lockstep batches of 1 and 9.

Uses one simulator. All 64 roots and nine futures are checked, including complete
leaf observations and logical statistics. This script never edits production.
"""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'python')]
import bench_engine as bench


def running_games():
    count = 0
    for path in Path('/proc').glob('[0-9]*/exe'):
        try:
            binary = path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
        if binary.is_relative_to(ROOT) and binary.name.startswith('pvz-portable'):
            count += 1
    return count


def identity(response):
    return bench._rollout_digest(response['observation'], response)


def summarize(report, expected_roots):
    roots = report['roots']
    if len(roots) != expected_roots or not roots:
        raise RuntimeError('incomplete root coverage')
    totals = {arm: sum(root['arms'][arm]['process_cpu_seconds'] for root in roots)
              for arm in ('native', '1', '9')}
    components = {arm: {key: sum(root['arms'][arm][key] for root in roots)
                       for key in ('network_cpu_seconds', 'simulation_cpu_seconds', 'snapshot_cpu_seconds')}
                  for arm in ('native', '1', '9')}
    for arm in ('1', '9'):
        components[arm]['feature_cpu_seconds'] = sum(root['arms'][arm]['feature_cpu_seconds'] for root in roots)
    return {'cpu_seconds': totals, 'components': components,
            'prototype_serial_over_batch': totals['1'] / totals['9'],
            'native_over_batch': totals['native'] / totals['9'],
            'serial_batch_identical_roots': sum(r['serial_batch_identical'] for r in roots),
            'native_batch_identical_roots': sum(r['native_batch_identical'] for r in roots),
            'network_speedup': components['1']['network_cpu_seconds'] / components['9']['network_cpu_seconds']}


def require_identical(report):
    if any(not root['serial_batch_identical'] or not root['native_batch_identical'] for root in report['roots']):
        raise RuntimeError('prototype leaf observations/logical statistics differ; timings are diagnostic only')


def main():
    if running_games() >= 2:
        raise RuntimeError('two perf game processes are live; wait for an existing job to finish')
    out = ROOT / 'artifacts/perf/batch_prototype'
    build = json.loads((out / 'build.json').read_text())
    manifest = json.loads((ROOT / 'artifacts/perf/engine_roots.json').read_text())
    expected_roots = sum(len(entry['decisions']) for entries in manifest['families'].values() for entry in entries)
    model = out / 'student.bin'
    bench._export_student(model)
    report = {'build': build, 'horizon': 2500, 'lanes': 9, 'sim_skip': 61,
              'replay_gate_version': bench.REPLAY_GATE_VERSION,
              'complete': False, 'roots': [], 'replays': []}
    (out / 'results.json').write_text(json.dumps(report, indent=1) + '\n')
    env = bench._open_env(build['binary'])
    try:
        env.load_rollout_net(model)
        env.set_sim_skip(61)
        for family, entries in manifest['families'].items():
            for entry in entries:
                archive = Path(entry['archive'])
                rows = bench._archive_rows(archive / ('frames.npz' if (archive / 'frames.npz').exists() else 'frames.jsonl'))
                recorded = [i for i in range(1, len(rows)) if rows[i].get('action')]
                whole, errors = bench.replay_episode(env, entry, rows, recorded[-1:])
                if errors:
                    raise RuntimeError(f'whole-episode replay gate failed: {archive}')
                report['replays'].append({'archive': str(archive), 'frames': len(recorded), 'mismatches': 0})
                for snapshot in whole.values():
                    env.release_snapshot(snapshot)
                snapshots, mismatches = bench.replay_episode(env, entry, rows, entry['decisions'])
                if mismatches or len(snapshots) != len(entry['decisions']):
                    raise RuntimeError(f'replay gate failed: {archive}')
                for index, root in snapshots.items():
                    history, history_start = env._sun_history_snapshots[root][:2]
                    samples = ' '.join(f'{tick} {amount}' for tick, amount in history)
                    suffix = f'{history_start} {len(history)} {samples}'.rstrip()
                    result = {'family': family, 'archive': str(archive), 'frame': index, 'arms': {}}
                    # Alternate scheduling order by root; native serial is also measured
                    # so a batch-only ratio cannot conceal snapshot-switching overhead.
                    order = ('native', '1', '9') if len(report['roots']) % 2 == 0 else ('9', '1', 'native')
                    for arm in order:
                        root_tick = int(env.restore(root)['tick'])
                        before_cpu = bench._process_cpu(env._process.pid)
                        before_wall = time.perf_counter()
                        if arm == 'native':
                            leaves = []
                            metrics = {key: 0 for key in ('network_cpu_seconds', 'simulation_cpu_seconds',
                                                         'snapshot_cpu_seconds')}
                            for seed in range(9):
                                response = env._command(f'ROLLOUT {root} {seed} {root_tick + 2500} '
                                                        f'61 1000 0 A {suffix}')
                                if not response.get('ok'):
                                    raise RuntimeError('native rollout failed')
                                leaves.append(identity(response))
                                for key in metrics:
                                    metrics[key] += response[key]
                        else:
                            response = env._command(f'PROTOTYPE_BATCH {root} 2500 61 9 {arm} 0 {suffix}')
                            if not response.get('ok') or len(response['lanes']) != 9:
                                raise RuntimeError('prototype rollout failed')
                            leaves = [identity(lane) for lane in response['lanes']]
                            metrics = {key: value for key, value in response.items() if key.endswith('cpu_seconds')}
                        metrics.update(process_cpu_seconds=bench._process_cpu(env._process.pid) - before_cpu,
                                       wall_seconds=time.perf_counter() - before_wall, digests=leaves)
                        result['arms'][arm] = metrics
                    result['serial_batch_identical'] = result['arms']['1']['digests'] == result['arms']['9']['digests']
                    result['native_batch_identical'] = result['arms']['native']['digests'] == result['arms']['9']['digests']
                    report['roots'].append(result)
                    (out / 'results.json').write_text(json.dumps(report, indent=1) + '\n')
                    print(f'{family} frame={index} roots={len(report["roots"])}/{expected_roots} '
                          f'serial_batch={result["serial_batch_identical"]} '
                          f'native_batch={result["native_batch_identical"]}', flush=True)
                    env.release_snapshot(root)
    finally:
        env.close()
    report['summary'] = summarize(report, expected_roots)
    report['complete'] = True
    report['correctness_passed'] = all(r['serial_batch_identical'] and r['native_batch_identical']
                                       for r in report['roots'])
    (out / 'results.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report['summary'], indent=1))
    require_identical(report)


if __name__ == '__main__':
    main()
