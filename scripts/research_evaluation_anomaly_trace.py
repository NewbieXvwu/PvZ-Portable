"""Trace a preregistered greedy evaluation anomaly using existing instrumentation."""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import research_baseline_trace as trace


def check_inventory(protocol):
    if protocol['scope'] != 'readonly_greedy_roof_anomaly_no_gate_override':
        raise ValueError('wrong diagnostic scope')
    if protocol['mode'] != 'greedy' or protocol['action_seed'] != protocol['environment_seed'] + 170_000:
        raise ValueError('preserve original greedy evaluation and action seed')
    if (protocol['workers'] != [1, 8] or protocol['relation_bias_fusion'] != [True, False]
            or protocol['repeats'] != 8 or len(protocol['executables']) != 2
            or protocol['worker_threads'] != 1 or protocol['max_actions'] != 4000):
        raise ValueError('diagnostic configuration inventory changed')
    if protocol['expected_episodes'] != 64:
        raise ValueError('all64 native/fusion/worker repeats required')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    check_inventory(protocol)
    for name, expected in protocol['required_fingerprints'].items():
        if trace.hashlib.sha256((trace.ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen diagnostic source changed: {name}')
    if args.output_dir.exists() or args.summary.exists():
        raise ValueError('fresh evidence paths required')
    trace.agent.configure_torch_threads(1)
    checkpoint = trace.torch.load(trace.ROOT / protocol['checkpoint'], map_location='cpu', weights_only=False)
    manifest = json.loads((trace.ROOT / checkpoint['experiment_config']['evaluation']['manifest']).read_text())
    task = next(t for t in manifest['tasks'] if t['task_id'] == protocol['task_id'])
    if protocol['environment_seed'] not in task['seeds']:
        raise ValueError('original environment seed missing')
    jobs = {i: {'task': task, 'seed': protocol['environment_seed'], 'action_seed': protocol['action_seed'],
                'deterministic': True, 'max_actions': 4000, 'allow_truncation': True}
            for i in range(protocol['repeats'])}
    args.output_dir.mkdir(parents=True)
    records = {}
    started = time.monotonic()
    deadline = started + protocol['timeout_seconds']
    report = {'scope': protocol['scope'], 'status': 'running', 'protocol_sha256': trace.hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'checkpoint_sha256': trace.hashlib.sha256((trace.ROOT / protocol['checkpoint']).read_bytes()).hexdigest(),
              'model_state_sha256': trace.profile._state_sha256(checkpoint['state_dict']),
              'coexecution': protocol['coexecution'], 'interpretation': 'Readonly localization; no training or capability acceptance, no replacement of original rows.'}
    with (args.output_dir / '.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for native, executable in protocol['executables'].items():
                for fusion in protocol['relation_bias_fusion']:
                    for workers in protocol['workers']:
                        key = f'{native}_fusion{int(fusion)}_workers{workers}'
                        directory = (args.output_dir / key).resolve()
                        directory.mkdir()
                        pool = multiprocessing.get_context('spawn').Pool(workers, initializer=trace.initialize,
                            initargs=(protocol['resource_dir'], checkpoint['state_dict'], jobs, 1, checkpoint['config'],
                                      str(directory), fusion, str((trace.ROOT / executable).resolve())))
                        try:
                            iterator = pool.imap_unordered(trace.run_job, sorted(jobs), chunksize=1)
                            collected = {}
                            for _ in jobs:
                                remaining = deadline - time.monotonic()
                                if remaining <= 0:
                                    raise RuntimeError('preregistered trace deadline exceeded')
                                job, result = iterator.next(timeout=min(remaining, 180))
                                collected[job] = result
                            pool.close()
                        except BaseException:
                            pool.terminate()
                            raise
                        finally:
                            pool.join()
                        records[key] = [collected[i] for i in sorted(jobs)]
                        trace.atomic_json(args.output_dir / 'partial.json', records)
                        print(key, [(r['outcome']['won'], r['outcome']['actions'], r['outcome']['terminal_tick']) for r in records[key]], flush=True)
            reference = json.loads(gzip.decompress(Path(next(iter(records.values()))[0]['trace_path']).read_bytes()))['steps']
            comparisons = []
            for key, group in records.items():
                for job, record in enumerate(group):
                    steps = json.loads(gzip.decompress(Path(record['trace_path']).read_bytes()))['steps']
                    comparisons.append({'configuration': key, 'job': job, 'first_difference': trace.first_difference(reference, steps)})
            if sum(len(v) for v in records.values()) != 64:
                raise ValueError('incomplete trace inventory')
            report.update(status='complete', records=records, comparisons=comparisons)
        except BaseException:
            report.update(status='failed', error=traceback.format_exc(), records=records)
            raise
        finally:
            report['seconds'] = time.monotonic() - started
            trace.atomic_json(args.output_dir / 'report.json', report)
            trace.atomic_json(args.summary, report)


if __name__ == '__main__':
    main()
