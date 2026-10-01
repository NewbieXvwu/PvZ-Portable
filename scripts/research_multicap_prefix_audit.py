"""Check the new pool's actual battle prefixes; no learning or evaluator override."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import research_multicap_pool_audit as pool

ROOT = pool.ROOT
PHYSICAL_FIELDS = ('plants', 'projectiles', 'sun', 'coins', 'grid_items', 'defenses', 'packets')


def battle(observation):
    return {**{field: observation[field] for field in PHYSICAL_FIELDS},
            'zombies': [z for z in observation['zombies'] if z['on_board']]}


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def compare(full, capped, cap):
    if full['wave'] > cap:
        return 'next_full_wave'
    if capped['terminal'] and capped['result'] == 1:
        return 'cap_win'
    if full['tick'] != capped['tick'] or battle(full) != battle(capped):
        raise ValueError('physical state/tick differs before intentional horizon boundary')
    if capped['terminal']:
        if not full['terminal'] or full['result'] != capped['result']:
            raise ValueError('normal loss must agree with the full-level prefix')
        return 'normal_terminal'
    if full['terminal']:
        raise ValueError('full task terminated while capped task still active')
    return None


def checked_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol['purpose'] != 'multicap_dynamic_prefix_no_learning':
        raise ValueError('wrong audit purpose')
    for name, expected in protocol['required_fingerprints'].items():
        source = Path(name) if Path(name).is_absolute() else ROOT/name
        if pool.profile.digest(source) != expected:
            raise ValueError(f'preregistered prefix source/evidence changed: {name}')
    native, tasks = pool.checked_protocol(protocol['pool_protocol'])
    pool.make_jobs(tasks, protocol['cases'])
    if protocol['cases']['seeds_per_task'] == 64:
        previous = protocol.get('prerequisite_evidence')
        if not previous or pool.profile.digest(previous['path']) != previous['sha256']:
            raise ValueError('full prefix gate requires preserved complete inspection')
        result = json.loads(Path(previous['path']).read_text())
        if (result['status'] != 'complete' or result['gate_result'] != 'pass' or result['cases'] != 120
                or result['protocol_sha256'] != previous['protocol_sha256']
                or pool.profile.digest(previous['protocol_path']) != previous['protocol_sha256']):
            raise ValueError('prefix inspection failed or changed')
    return protocol, native, tasks


def episode(job, protocol, native, directory, global_deadline):
    if pool.profile.ENCODER is None:
        pool.profile.load_encoder(native['encoder_root'])
    task = job['task']
    directory = Path(directory)
    context = {'job': job['job'], 'task_id': task['task_id'], 'terrain': task['terrain'],
               'cap': task['wave_cap'], 'seed': job['seed'], 'strategy': job['strategy']}
    raw_path = directory/f"job_{job['job']:06d}.steps.json.gz"
    started = time.monotonic()
    deadline = min(global_deadline, started + protocol['episode_timeout_seconds'])
    comparisons = 0
    try:
        with pool.profile.ENV(native['resource_dir'], executable=native['executable']) as full_env, \
             pool.profile.ENV(native['resource_dir'], executable=native['executable']) as cap_env, gzip.open(raw_path, 'wt') as raw:
            kwargs = {'level': task['level'], 'seed': job['seed'], 'playthrough': task['playthrough'],
                      'zombie_count_multiplier': task['zombie_count_multiplier']}
            full, _ = full_env.reset(deck=task['deck'], task=pool.profile.TASK(**kwargs))
            capped, _ = cap_env.reset(deck=task['deck'], task=pool.profile.TASK(**kwargs, wave_cap=task['wave_cap']))
            left = full_env._command('PRIV')['observation']['hidden']['zombies_in_wave']
            right = cap_env._command('PRIV')['observation']['hidden']['zombies_in_wave']
            if left[:task['wave_cap']] != right:
                raise ValueError('actual capped wave table differs from full prefix')
            for index in range(protocol['max_actions'] + 1):
                if time.monotonic() >= deadline or pool.profile.available_ram() < protocol['minimum_available_ram_bytes']:
                    raise RuntimeError('prefix audit time/RAM precondition unmet; partial scene retained')
                try:
                    reason = compare(full, capped, task['wave_cap'])
                except ValueError:
                    pool.profile.save(directory/f"job_{job['job']:06d}.mismatch.json.gz",
                                      {'context': context, 'decision': index, 'full': full, 'capped': capped})
                    raise
                row = {'decision': index, 'full_tick': full['tick'], 'cap_tick': capped['tick'],
                       'full_wave': full['wave'], 'cap_wave': capped['wave'],
                       'full_battle_sha256': sha(battle(full)), 'cap_battle_sha256': sha(battle(capped)),
                       'stop_reason': reason}
                comparisons += reason not in ('next_full_wave', 'cap_win')
                if reason or index == protocol['max_actions'] or full['tick'] >= protocol['max_ticks']:
                    row['stop_reason'] = reason or ('action_horizon' if index == protocol['max_actions'] else 'tick_horizon')
                    raw.write(json.dumps(row, separators=(',', ':'))+'\n')
                    reason = row['stop_reason']
                    break
                action = {'type': 'wait', 'ticks': 300} if job['strategy'] == 'wait' else pool.profile.CHOOSE(full)
                full, _, _, _, left_info = full_env.step(action)
                capped, _, _, _, right_info = cap_env.step(action)
                row.update(action=action, full_info=left_info, cap_info=right_info)
                raw.write(json.dumps(row, separators=(',', ':'))+'\n')
                if not left_info['ok'] or not right_info['ok']:
                    raise ValueError('shared control action was illegal in a prefix')
        record = {**context, 'wave_table_matches_prefix': True, 'physical_comparisons': comparisons,
                  'stop_reason': reason, 'full_wave_count': full['wave_count'], 'cap_wave_count': capped['wave_count'],
                  'full_outcome': {k: full[k] for k in ('terminal', 'result', 'wave', 'tick')},
                  'cap_outcome': {k: capped[k] for k in ('terminal', 'result', 'wave', 'tick')},
                  'raw_path': str(raw_path.relative_to(ROOT)), 'raw_sha256': pool.profile.digest(raw_path),
                  'seconds': time.monotonic()-started}
        pool.profile.save(directory/f"job_{job['job']:06d}.json", record)
        return record
    except BaseException:
        pool.profile.save(directory/f"job_{job['job']:06d}.failure.json",
                          {'context': context, 'error': traceback.format_exc(), 'partial_raw_path': str(raw_path)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol, native, tasks = checked_protocol(args.protocol)
    output = args.output_dir.resolve()
    output.relative_to(ROOT/'artifacts/research')
    output.mkdir(parents=True, exist_ok=False)
    directory = output/'cases'
    directory.mkdir()
    pool.profile.load_encoder(native['encoder_root'])
    from pvz_research import ResourceMonitor
    monitor = ResourceMonitor()
    records, started = [], time.monotonic()
    report = {'schema_version': 1, 'status': 'running', 'gate_result': 'unresolved',
              'scope': protocol['purpose'], 'protocol_sha256': pool.profile.digest(args.protocol),
              'pool_protocol_sha256': pool.profile.digest(protocol['pool_protocol']),
              'coexecution': protocol['coexecution'], 'limitations': protocol['limitations']}
    with (output/'.execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor.thread.start()
        executor = ProcessPoolExecutor(max_workers=protocol['workers'], mp_context=multiprocessing.get_context('spawn'))
        pending, jobs = {}, iter(pool.make_jobs(tasks, protocol['cases']))
        deadline = started + protocol['total_timeout_seconds']
        def submit():
            job = next(jobs, None)
            if job is not None:
                pending[executor.submit(episode, job, protocol, native, str(directory), deadline)] = job['job']
        try:
            for _ in range(protocol['workers'] * 2):
                submit()
            while pending:
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                if time.monotonic() >= deadline:
                    raise RuntimeError('prefix audit exceeded preregistered deadline')
                for future in completed:
                    pending.pop(future)
                    records.append(future.result())
                    submit()
                if completed and (len(records) % 30 == 0 or not pending):
                    pool.profile.save(output/'progress.json', {'cases': len(records),
                        'expected': protocol['cases']['expected_episodes'], 'seconds': time.monotonic()-started})
                    print(f"prefix checked {len(records)}/{protocol['cases']['expected_episodes']}", flush=True)
            records.sort(key=lambda r:r['job'])
            if [r['job'] for r in records] != list(range(protocol['cases']['expected_episodes'])):
                raise ValueError('missing/duplicate prefix cases')
            pool.profile.save(output/'records.json.gz', records)
            report.update(status='complete', gate_result='pass', cases=len(records),
                          physical_comparisons=sum(r['physical_comparisons'] for r in records),
                          records_sha256=pool.profile.digest(output/'records.json.gz'))
        except BaseException:
            report.update(status='failed', gate_result='fail', cases=len(records), error=traceback.format_exc())
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
