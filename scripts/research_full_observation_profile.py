"""Measure complete-level public inputs with frozen non-learning controls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import gzip
import hashlib
import importlib
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
ENCODER = None
ENV = None
TASK = None
CHOOSE = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if path.name.endswith('.gz'):
        with gzip.open(temporary, 'wt') as stream:
            json.dump(value, stream, separators=(',', ':'))
    else:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def checked_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol['purpose'] != 'full_level_public_input_cost_only_no_learning':
        raise ValueError('not a public-input profiling protocol')
    for name, expected in protocol['required_fingerprints'].items():
        source = Path(name) if Path(name).is_absolute() else ROOT / name
        if digest(source) != expected:
            raise ValueError(f'preregistered profiling source changed: {name}')
    gate = json.loads(Path(protocol['native_gate']).read_text())
    if gate['gate_result'] != 'pass' or gate['simulator_sha256'] != digest(protocol['executable']):
        raise ValueError('combined native gate does not permit this executable')
    for name, expected in gate['required_fingerprints'].items():
        if digest(Path(protocol['encoder_root']) / name) != expected:
            raise ValueError(f'native gate source/evidence changed: {name}')
    prior = protocol.get('prerequisite_evidence')
    if protocol['expected_episodes'] > 20 and not prior:
        raise ValueError('full profiling requires complete inspection evidence')
    if prior:
        report = Path(prior['path'])
        if digest(report) != prior['sha256']:
            raise ValueError('inspection evidence changed')
        result = json.loads(report.read_text())
        if result['status'] != 'complete' or result['episodes'] != prior['episodes']:
            raise ValueError('complete inspection required before full profiling')
        if (result['protocol_sha256'] != prior['protocol_sha256']
                or digest(prior['protocol_path']) != prior['protocol_sha256']):
            raise ValueError('inspection belongs to a different protocol')
    make_jobs(protocol)
    return protocol


def make_jobs(protocol):
    jobs = []
    names = set()
    for task in protocol['tasks']:
        if task['task_id'] in names or task['wave_cap'] is not None:
            raise ValueError('unique complete-level tasks required')
        names.add(task['task_id'])
        if len(task['seeds']) != len(set(task['seeds'])):
            raise ValueError('duplicate environment seed')
        for strategy in protocol['strategies']:
            if strategy not in ('wait', 'scripted'):
                raise ValueError('unknown non-learning control')
            for seed in task['seeds']:
                jobs.append({'job': len(jobs), 'task': task, 'seed': seed, 'strategy': strategy})
    if len(jobs) != protocol['expected_episodes']:
        raise ValueError('task/seed/control inventory changed')
    return jobs


def load_encoder(root):
    global ENCODER, ENV, TASK, CHOOSE
    root = Path(root).resolve()
    sys.path[:0] = [str(root / 'python'), str(root / 'scripts')]
    ENCODER = importlib.import_module('pvz_agent_model')
    environment = importlib.import_module('pvz_env')
    controller = importlib.import_module('research_aid_feasibility')
    for module in (ENCODER, environment, controller):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError('profiling imported a different worktree')
    ENCODER.configure_torch_threads(1)
    ENV, TASK, CHOOSE = environment.PvZEnv, environment.TaskSpec, controller.choose


def available_ram():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) * 1024


def encode_stats(observation, flags):
    started = time.monotonic()
    tensors, metadata = ENCODER.observation_tokens(observation, flags)
    packed = ENCODER.pack_tokens(tensors, metadata)
    seconds = time.monotonic() - started
    for action in observation['legal_actions']['plants']:
        if action['row'] * 9 + action['col'] not in metadata['cell_tokens']:
            raise ValueError('profiling encoder omitted a legal action cell')
    inverse = {v: k for k, v in ENCODER.TOKEN_KINDS.items()}
    kinds = Counter(inverse[int(k)] for k in tensors['kinds'])
    return {'tick': observation['tick'], 'wave': observation['wave'],
            'wave_count': observation['wave_count'], 'terminal': observation['terminal'],
            'tokens': len(tensors['kinds']), 'kind_counts': dict(kinds),
            'plants': len(observation['plants']), 'projectiles': len(observation['projectiles']),
            'board_zombies': sum(z['on_board'] for z in observation['zombies']),
            'preview_zombies': sum(not z['on_board'] for z in observation['zombies']),
            'packed_array_bytes': sum(a.nbytes for a in packed.values()),
            'public_observation_json_bytes': len(json.dumps(observation, separators=(',', ':')).encode()),
            'tokenize_pack_seconds': seconds}, packed


def episode(job, protocol, directory, global_deadline):
    if ENCODER is None:
        load_encoder(protocol['encoder_root'])
    directory = Path(directory)
    task = job['task']
    context = {'job': job['job'], 'task_id': task['task_id'], 'terrain': task['terrain'],
               'aid': task['aid'], 'level': task['level'], 'seed': job['seed'], 'strategy': job['strategy']}
    started = time.monotonic()
    deadline = min(global_deadline, started + protocol['episode_timeout_seconds'])
    histogram, late_histogram, by_wave, actions = Counter(), Counter(), defaultdict(Counter), Counter()
    peak, peak_observation, peak_packed = None, None, None
    observation_count = packed_total = public_total = 0
    encoding_seconds = 0.
    raw_path = directory / f"job_{job['job']:06d}.observations.json.gz"
    try:
        with ENV(protocol['resource_dir'], executable=protocol['executable']) as env, gzip.open(raw_path, 'wt') as raw:
            observation, _ = env.reset(deck=task['deck'], task=TASK(level=task['level'], seed=job['seed'],
                wave_cap=None, playthrough=2, zombie_count_multiplier=1.,
                preplanted=tuple(tuple(p) for p in task['preplanted'])))
            if observation['wave_count'] != task['expected_full_waves']:
                raise ValueError('full level wave count changed')
            for index in range(protocol['max_actions'] + 1):
                if time.monotonic() >= deadline or available_ram() < protocol['minimum_available_ram_bytes']:
                    raise RuntimeError('profiling deadline or RAM precondition unmet; scene retained')
                stats, packed = encode_stats(observation, protocol['input_flags'])
                histogram[stats['tokens']] += 1
                by_wave[stats['wave']][stats['tokens']] += 1
                if stats['wave'] >= math.ceil(stats['wave_count'] * protocol['late_wave_fraction']):
                    late_histogram[stats['tokens']] += 1
                raw.write(json.dumps({'observation_index': index, **stats}, separators=(',', ':')) + '\n')
                observation_count += 1
                packed_total += stats['packed_array_bytes']
                public_total += stats['public_observation_json_bytes']
                encoding_seconds += stats['tokenize_pack_seconds']
                if peak is None or stats['tokens'] > peak['tokens']:
                    peak, peak_observation, peak_packed = stats, observation, packed
                if observation['terminal'] or index == protocol['max_actions']:
                    break
                action = {'type': 'wait', 'ticks': 300} if job['strategy'] == 'wait' else CHOOSE(observation)
                observation, _, _, _, info = env.step(action)
                if not info['ok']:
                    raise ValueError(f'illegal control action: {action}')
                actions[action['type']] += 1
        peak_path = directory / f"job_{job['job']:06d}.peak.json.gz"
        save(peak_path, {'context': context, 'stats': peak, 'observation': peak_observation})
        import numpy as np
        packed_path = directory / f"job_{job['job']:06d}.peak.npz"
        np.savez_compressed(packed_path, **peak_packed)
        record = {**context, 'result': observation['result'], 'won': observation['result'] == 1,
                  'terminated': observation['terminal'], 'truncated': not observation['terminal'],
                  'terminal_wave': observation['wave'], 'wave_count': observation['wave_count'],
                  'tick': observation['tick'], 'actions': dict(actions), 'observations': observation_count,
                  'token_histogram': dict(histogram), 'late_token_histogram': dict(late_histogram),
                  'wave_token_histograms': {w: dict(h) for w, h in by_wave.items()},
                  'peak': peak, 'packed_array_bytes_total': packed_total,
                  'public_observation_json_bytes_total': public_total, 'tokenize_pack_seconds': encoding_seconds,
                  'raw_path': str(raw_path.relative_to(ROOT)), 'raw_sha256': digest(raw_path),
                  'peak_path': str(peak_path.relative_to(ROOT)), 'peak_sha256': digest(peak_path),
                  'peak_packed_path': str(packed_path.relative_to(ROOT)), 'peak_packed_sha256': digest(packed_path),
                  'seconds': time.monotonic() - started}
        if record['won'] and record['truncated']:
            raise ValueError('truncated profile cannot be a true win')
        save(directory / f"job_{job['job']:06d}.json", record)
        return record
    except BaseException:
        save(directory / f"job_{job['job']:06d}.failure.json", {'context': context, 'error': traceback.format_exc()})
        raise


def histogram_summary(histogram):
    counts = {int(k): int(v) for k, v in histogram.items()}
    total = sum(counts.values())
    if not total:
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'max': None, 'dense_token_pairs': 0}
    def value_at(index):
        cumulative = 0
        for value, count in sorted(counts.items()):
            cumulative += count
            if index < cumulative:
                return value
        raise ValueError('histogram index out of range')
    def quantile(q):
        index = q * (total - 1)
        lo, hi = math.floor(index), math.ceil(index)
        return value_at(lo) + (value_at(hi) - value_at(lo)) * (index - lo)
    return {'count': total, 'min': min(counts), 'median': quantile(.5), 'p95': quantile(.95),
            'max': max(counts), 'dense_token_pairs': sum(k*k*v for k, v in counts.items())}


def summarize(records):
    cohorts = defaultdict(list)
    for row in records:
        cohorts[row['task_id'] + '/' + row['strategy']].append(row)
    result = {}
    for key, rows in sorted(cohorts.items()):
        hist, late = Counter(), Counter()
        for row in rows:
            hist.update({int(k): v for k, v in row['token_histogram'].items()})
            late.update({int(k): v for k, v in row['late_token_histogram'].items()})
        won, count = sum(row['won'] for row in rows), len(rows)
        z = 1.959963984540054
        p, denominator = won/count, 1+z*z/count
        center = (p+z*z/(2*count))/denominator
        radius = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
        result[key] = {'episodes': count, 'won': won, 'win_rate': p, 'wilson_95': [max(0., center-radius), min(1., center+radius)],
                       'truncated': sum(r['truncated'] for r in rows), 'tokens': histogram_summary(hist),
                       'late_tokens': histogram_summary(late),
                       'episodes_reaching_late_waves': sum(bool(r['late_token_histogram']) for r in rows),
                       'packed_array_bytes_total': sum(r['packed_array_bytes_total'] for r in rows),
                       'public_observation_json_bytes_total': sum(r['public_observation_json_bytes_total'] for r in rows),
                       'tokenize_pack_seconds': sum(r['tokenize_pack_seconds'] for r in rows)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol = checked_protocol(args.protocol)
    output = args.output_dir.resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    directory = output / 'episodes'
    directory.mkdir()
    load_encoder(protocol['encoder_root'])
    from pvz_research import ResourceMonitor
    monitor = ResourceMonitor()
    records, started = [], time.monotonic()
    report = {'schema_version': 1, 'protocol_sha256': digest(args.protocol), 'status': 'running',
              'scope': protocol['purpose'], 'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'coexecution': protocol['coexecution'], 'limitations': protocol['limitations']}
    with (output / '.execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor.thread.start()
        executor = ProcessPoolExecutor(max_workers=protocol['workers'], mp_context=multiprocessing.get_context('spawn'))
        pending = {}
        jobs = iter(make_jobs(protocol))
        deadline = started + protocol['total_timeout_seconds']
        def submit():
            job = next(jobs, None)
            if job is not None:
                pending[executor.submit(episode, job, protocol, str(directory), deadline)] = job['job']
        try:
            for _ in range(protocol['workers'] * 2):
                submit()
            while pending:
                completed, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                if time.monotonic() >= deadline:
                    raise RuntimeError('full profiling exceeded preregistered deadline')
                for future in completed:
                    pending.pop(future)
                    records.append(future.result())
                    submit()
                if completed and (len(records) % 64 == 0 or not pending):
                    save(output / 'progress.json', {'episodes': len(records), 'expected': protocol['expected_episodes'],
                        'seconds': time.monotonic()-started, 'resources': {'peak_process_tree_rss_bytes': monitor.peak_tree_rss_bytes,
                        'min_system_available_bytes': monitor.min_available_bytes, 'peak_system_swap_used_bytes': monitor.peak_swap_used_bytes}})
                    print(f"profiled {len(records)}/{protocol['expected_episodes']}", flush=True)
            records.sort(key=lambda r: r['job'])
            if [r['job'] for r in records] != list(range(protocol['expected_episodes'])):
                raise ValueError('missing or duplicate profiling jobs')
            save(output / 'records.json.gz', records)
            report.update(status='complete', episodes=len(records), summaries=summarize(records),
                          records_sha256=digest(output / 'records.json.gz'))
        except BaseException:
            report.update(status='failed', error=traceback.format_exc(), episodes=len(records))
            save(output / 'failure.json', report)
            raise
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            monitor.stop.set()
            monitor.thread.join()
            report.update(seconds=time.monotonic()-started, resources={'peak_process_tree_rss_bytes': monitor.peak_tree_rss_bytes,
                'min_system_available_bytes': monitor.min_available_bytes, 'peak_system_swap_used_bytes': monitor.peak_swap_used_bytes})
            save(output / 'report.json', report)


if __name__ == '__main__':
    main()
