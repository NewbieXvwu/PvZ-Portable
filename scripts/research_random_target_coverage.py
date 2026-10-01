"""Measure actual target coverage under uniform legal actions; never PPO data."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import json
import multiprocessing
from pathlib import Path
import random
import time
import traceback

import research_multicap_pool_audit as pool

ROOT = pool.ROOT


def choose(observation, rng):
    legal = observation['legal_actions']
    kinds = ['wait'] + (['plant'] if legal['plants'] else []) + (['shovel'] if legal['shovels'] else [])
    kind = rng.choice(kinds)
    if kind == 'plant':
        return {'type': kind, **rng.choice(legal['plants'])}
    if kind == 'shovel':
        col, row = rng.choice(legal['shovels'])
        return {'type': kind, 'col': col, 'row': row}
    return {'type': 'wait', 'ticks': rng.choice([60, 150, 300])}


def checked_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol['purpose'] != 'random_legal_target_coverage_only_no_learning':
        raise ValueError('wrong measurement scope')
    for name, expected in protocol['required_fingerprints'].items():
        source = Path(name) if Path(name).is_absolute() else ROOT/name
        if pool.profile.digest(source) != expected:
            raise ValueError(f'preregistered source/evidence changed: {name}')
    native, tasks = pool.checked_protocol(protocol['pool_protocol'])
    selected = [t for t in tasks if set(t['deck']).intersection(protocol['target_plant_types'])]
    if [t['task_id'] for t in selected] != protocol['task_ids']:
        raise ValueError('all source tasks containing target plants must be kept, in order')
    if protocol['seed_count'] not in (2, 64) or protocol['action_seed_offset'] != 290000:
        raise ValueError('explicit paired seed scope changed')
    jobs = [{'job': i, 'task': task, 'seed': seed, 'strategy': 'legal_random'}
            for i, (task, seed) in enumerate((t, s) for t in selected for s in t['seeds'][:protocol['seed_count']])]
    if len(jobs) != protocol['expected_episodes']:
        raise ValueError('measurement inventory changed')
    if protocol['seed_count'] == 64:
        prior = protocol.get('prerequisite_evidence')
        if not prior or pool.profile.digest(prior['path']) != prior['sha256']:
            raise ValueError('full coverage requires immutable complete inspection')
        report = json.loads(Path(prior['path']).read_text())
        if (report['status'] != 'complete' or report['episodes'] != 6
                or report['protocol_sha256'] != prior['protocol_sha256']
                or pool.profile.digest(prior['protocol_path']) != prior['protocol_sha256']):
            raise ValueError('inspection incomplete or changed')
    return protocol, native, jobs


def episode(job, protocol, native, directory, deadline):
    if pool.profile.ENCODER is None:
        pool.profile.load_encoder(native['encoder_root'])
    rng = random.Random(job['seed'] + protocol['action_seed_offset'])
    pool.profile.CHOOSE = lambda observation: choose(observation, rng)
    settings = dict(native, episode_timeout_seconds=protocol['episode_timeout_seconds'],
                    minimum_available_ram_bytes=protocol['minimum_available_ram_bytes'])
    return pool.episode(job, settings, directory, deadline)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol, native, jobs = checked_protocol(args.protocol)
    output = args.output_dir.resolve()
    output.relative_to(ROOT/'artifacts/research')
    output.mkdir(parents=True, exist_ok=False)
    directory = output/'episodes'
    directory.mkdir()
    pool.profile.load_encoder(native['encoder_root'])
    from pvz_research import ResourceMonitor
    monitor = ResourceMonitor()
    records, started = [], time.monotonic()
    report = {'schema_version': 1, 'status': 'running', 'scope': protocol['purpose'],
              'protocol_sha256': pool.profile.digest(args.protocol),
              'pool_protocol_sha256': pool.profile.digest(protocol['pool_protocol']),
              'coexecution': protocol['coexecution'], 'limitations': protocol['limitations']}
    with (output/'.execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor.thread.start()
        executor = ProcessPoolExecutor(max_workers=protocol['workers'], mp_context=multiprocessing.get_context('spawn'))
        pending, inventory = {}, iter(jobs)
        deadline = started + protocol['total_timeout_seconds']
        def submit():
            job = next(inventory, None)
            if job is not None:
                pending[executor.submit(episode, job, protocol, native, str(directory), deadline)] = job['job']
        try:
            for _ in range(protocol['workers']*2):
                submit()
            while pending:
                done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                if time.monotonic() >= deadline:
                    raise RuntimeError('target coverage exceeded frozen deadline')
                for future in done:
                    pending.pop(future)
                    records.append(future.result())
                    submit()
                if done:
                    pool.profile.save(output/'progress.json', {'episodes': len(records), 'expected': len(jobs),
                                      'seconds': time.monotonic()-started})
            records.sort(key=lambda r:r['job'])
            if [r['job'] for r in records] != list(range(len(jobs))):
                raise ValueError('missing/duplicate target coverage jobs')
            pool.profile.save(output/'records.json.gz', records)
            report.update(status='complete', episodes=len(records), summaries=pool.summarize(records),
                          records_sha256=pool.profile.digest(output/'records.json.gz'))
        except BaseException:
            report.update(status='failed', episodes=len(records), error=traceback.format_exc())
            pool.profile.save(output/'failure.json', report)
            raise
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            monitor.stop.set()
            monitor.thread.join()
            report.update(seconds=time.monotonic()-started, resources={'peak_process_tree_rss_bytes': monitor.peak_tree_rss_bytes,
                          'min_system_available_bytes': monitor.min_available_bytes, 'peak_system_swap_used_bytes': monitor.peak_swap_used_bytes})
            pool.profile.save(output/'report.json', report)


if __name__ == '__main__':
    main()
