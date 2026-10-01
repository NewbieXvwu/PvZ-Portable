"""Audit a versioned no-aid task pool without producing PPO training data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import fcntl
import gzip
import importlib
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import time
import traceback

import research_full_observation_profile as profile

ROOT = profile.ROOT


def checked_pool(protocol):
    source = json.loads((ROOT / protocol['original_manifest']).read_text())
    pool = json.loads((ROOT / protocol['manifest']).read_text())
    tasks = pool['tasks']
    if pool['split'] != 'train' or len(tasks) != 30:
        raise ValueError('expected new 30-task training pool')
    if tasks[:20] != source['tasks']:
        raise ValueError('original tasks/seeds must remain unchanged and in order')
    groups, identities, seeds = Counter(), set(), set()
    for task in tasks:
        key = task['task_id']
        if key in identities or len(task['seeds']) != 64 or len(set(task['seeds'])) != 64:
            raise ValueError('unique tasks with all 64 seeds required')
        if seeds.intersection(task['seeds']):
            raise ValueError('training seed blocks overlap')
        if task['preplanted'] or task['sun_start'] != 50 or task['playthrough'] != 2:
            raise ValueError('no-aid original opening required')
        if task['level'] in protocol['adventure_conveyor_levels']:
            raise ValueError('this new pool requires ordinary seed banks')
        identities.add(key)
        seeds.update(task['seeds'])
        groups[(task['terrain'], task['wave_cap'])] += 1
    expected = {(terrain, cap): 2 for terrain in ('day', 'night', 'pool', 'fog', 'roof')
                for cap in (1, 3, 5)}
    if dict(groups) != expected:
        raise ValueError('all 15 terrain/cap groups must contain two tasks')
    for addition, task in zip(pool['generation']['additions'], tasks[20:], strict=True):
        inherited = next(t for t in source['tasks'] if t['task_id'] == addition['source_task_id'])
        intended = dict(inherited, task_id=addition['task_id'], wave_cap=addition['wave_cap'],
                        zombie_count_multiplier=1.0, seeds=addition['seeds'])
        if task != intended:
            raise ValueError('new variant differs from preregistered inheritance')
    for path in protocol['evaluation_manifests']:
        other = json.loads((ROOT / path).read_text())
        for task in other['tasks']:
            if task.get('evaluation_role') == 'training_probe':
                original = next((t for t in source['tasks'] if t['task_id'] == task['task_id']), None)
                if {k: v for k, v in task.items() if k != 'evaluation_role'} != original:
                    raise ValueError('training probe must be an exact original training task')
            elif seeds.intersection(task['seeds']):
                raise ValueError('training seeds overlap validation/heldout seeds')
    return tasks


def make_jobs(tasks, protocol):
    count = protocol['seeds_per_task']
    if count not in (2, 64) or protocol['strategies'] != ['wait', 'scripted']:
        raise ValueError('only explicit inspection/full paired controls allowed')
    jobs = [{'job': i, 'task': task, 'seed': seed, 'strategy': strategy}
            for i, (task, strategy, seed) in enumerate(
                (t, strategy, seed) for t in tasks for strategy in protocol['strategies']
                for seed in t['seeds'][:count])]
    if len(jobs) != protocol['expected_episodes']:
        raise ValueError('episode inventory changed')
    return jobs


def checked_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol['purpose'] != 'multicap_noaid_pool_audit_only_no_learning':
        raise ValueError('wrong experiment purpose')
    for name, expected in protocol['required_fingerprints'].items():
        source = Path(name) if Path(name).is_absolute() else ROOT / name
        if profile.digest(source) != expected:
            raise ValueError(f'preregistered source/evidence changed: {name}')
    gate = json.loads(Path(protocol['native_gate']).read_text())
    if gate['gate_result'] != 'pass' or gate['simulator_sha256'] != profile.digest(protocol['executable']):
        raise ValueError('native gate does not authorize this executable')
    for name, expected in gate['required_fingerprints'].items():
        source = Path(name) if Path(name).is_absolute() else ROOT / name
        if profile.digest(source) != expected:
            raise ValueError(f'native prerequisite changed: {name}')
    tasks = checked_pool(protocol)
    make_jobs(tasks, protocol)
    if protocol['seeds_per_task'] == 64:
        prior = protocol.get('prerequisite_evidence')
        if not prior or profile.digest(prior['path']) != prior['sha256']:
            raise ValueError('full audit requires preserved complete inspection')
        report = json.loads(Path(prior['path']).read_text())
        if (report['status'] != 'complete' or report['episodes'] != 120
                or report['protocol_sha256'] != prior['protocol_sha256']
                or profile.digest(prior['protocol_path']) != prior['protocol_sha256']
                or report['manifest_sha256'] != profile.digest(ROOT / protocol['manifest'])):
            raise ValueError('inspection scope/protocol/pool changed')
    return protocol, tasks


def target_stats(observation, packed):
    references = [{'kind': kind, 'entity_index': index, 'id': entity.get('id'),
                   'target_zombie_id': entity['target_zombie_id']}
                  for kind in ('plants', 'projectiles') for index, entity in enumerate(observation[kind])
                  if entity['target_zombie_id'] not in (0, -1)]
    resolved = int((packed['target_indices'] >= 0).sum())
    if resolved > len(references):
        raise ValueError('encoded target count exceeds public references')
    return references, resolved


def curriculum_mechanics(tasks, protocol):
    course = importlib.import_module('pvz_curriculum')
    if not Path(course.__file__).resolve().is_relative_to(Path(protocol['encoder_root']).resolve()):
        raise ValueError('curriculum imported from a different worktree')
    settings = protocol['mechanics_only_curriculum_settings']
    state = course.initial_state(tasks)
    groups = defaultdict(list)
    for task in tasks:
        groups[(task['terrain'], task['wave_cap'])].append(task['task_id'])
    window = settings['window_episodes']
    for keys in groups.values():
        for index, key in enumerate(keys):
            state['history'][key] = [False] * window + [index == 0] * window
            state['completed'][key] = 2 * window
    balanced, _ = course.probabilities(tasks, state, settings, 'terrain_balanced')
    progress, _ = course.probabilities(tasks, state, settings, 'learning_progress')
    restored, _ = course.probabilities(tasks, json.loads(json.dumps(state)), settings, 'learning_progress')
    if any(not math.isclose(p, 1/30, abs_tol=1e-12) for p in balanced) or progress != restored:
        raise ValueError('uniform or serialized course-state calculation changed')
    for keys in groups.values():
        indices = [next(i for i, t in enumerate(tasks) if t['task_id'] == key) for key in keys]
        if not math.isclose(sum(progress[i] for i in indices), 1/15, abs_tol=1e-12):
            raise ValueError('terrain/cap coverage changed')
    return {'scope': 'synthetic history mechanics only; no actual learned progress',
            'groups': len(groups), 'tasks_per_group': 2, 'balanced_probability': balanced[0],
            'positive_progress_probability_min': min(progress), 'positive_progress_probability_max': max(progress),
            'settings': settings, 'serialized_state_weights_exact': progress == restored}


def episode(job, protocol, directory, global_deadline):
    if profile.ENCODER is None:
        profile.load_encoder(protocol['encoder_root'])
    task, directory = job['task'], Path(directory)
    context = {'job': job['job'], 'task_id': task['task_id'], 'terrain': task['terrain'],
               'level': task['level'], 'cap': task['wave_cap'], 'aid': 'none',
               'seed': job['seed'], 'strategy': job['strategy']}
    started = time.monotonic()
    deadline = min(global_deadline, started + protocol['episode_timeout_seconds'])
    raw_path = directory / f"job_{job['job']:06d}.steps.json.gz"
    histogram, actions, events = Counter(), Counter(), Counter()
    observations = target_states = target_references = target_resolved = 0
    peak = peak_observation = None
    packed_bytes = encoding_seconds = 0
    try:
        with profile.ENV(protocol['resource_dir'], executable=protocol['executable']) as env:
            arguments = {'level': task['level'], 'seed': job['seed'], 'playthrough': task['playthrough'],
                         'zombie_count_multiplier': task['zombie_count_multiplier']}
            full, _ = env.reset(deck=task['deck'], task=profile.TASK(**arguments, wave_cap=None))
            if full['wave_count'] != protocol['expected_full_waves'][str(task['level'])]:
                raise ValueError('actual replay full wave count differs from preregistered count')
            observation, _ = env.reset(deck=task['deck'], task=profile.TASK(**arguments, wave_cap=task['wave_cap']))
            if (observation['wave_count'] != task['wave_cap'] or observation['sun'] != task['sun_start']
                    or [(p['type'], p['row'], p['col']) for p in observation['plants']]
                       != [(p['type'], p['row'], p['col']) for p in full['plants']]
                    or observation['terminal']
                    or [p['type'] for p in observation['packets']] != task['deck']):
                raise ValueError('actual capped opening differs from requested task')
            opening_path = directory / f"job_{job['job']:06d}.opening.json.gz"
            profile.save(opening_path, {'context': context, 'full': full, 'capped': observation})
            with gzip.open(raw_path, 'wt') as raw:
                for index in range(protocol['max_actions'] + 1):
                    if time.monotonic() >= deadline or profile.available_ram() < protocol['minimum_available_ram_bytes']:
                        raise RuntimeError('audit time/RAM precondition unmet; all partial evidence retained')
                    stats, packed = profile.encode_stats(observation, protocol['input_flags'])
                    references, resolved = target_stats(observation, packed)
                    histogram[stats['tokens']] += 1
                    observations += 1
                    target_states += bool(references)
                    target_references += len(references)
                    target_resolved += resolved
                    packed_bytes += stats['packed_array_bytes']
                    encoding_seconds += stats['tokenize_pack_seconds']
                    if peak is None or stats['tokens'] > peak['tokens']:
                        peak, peak_observation = stats, observation
                    row = {'observation_index': index, **stats, 'target_references': references,
                           'resolved_targets': resolved}
                    if references:
                        row['observation'] = observation
                    if observation['terminal'] or index == protocol['max_actions']:
                        raw.write(json.dumps(row, separators=(',', ':')) + '\n')
                        break
                    action = {'type': 'wait', 'ticks': 300} if job['strategy'] == 'wait' else profile.CHOOSE(observation)
                    row['action'] = action
                    observation, _, _, _, info = env.step(action)
                    row['info'] = info
                    raw.write(json.dumps(row, separators=(',', ':')) + '\n')
                    if not info['ok']:
                        raise ValueError(f'illegal control action: {action}')
                    actions[action['type']] += 1
                    events.update(info['events'])
        peak_path = directory / f"job_{job['job']:06d}.peak.json.gz"
        profile.save(peak_path, {'context': context, 'stats': peak, 'observation': peak_observation})
        record = {**context, 'won': observation['result'] == 1, 'result': observation['result'],
                  'terminated': bool(observation['terminal']), 'truncated': not observation['terminal'],
                  'terminal_wave': observation['wave'], 'wave_count': observation['wave_count'],
                  'tick': observation['tick'], 'full_wave_count': full['wave_count'],
                  'actions': dict(actions), 'events': dict(events), 'observations': observations,
                  'token_histogram': dict(histogram), 'late_token_histogram': {},
                  'packed_array_bytes_total': packed_bytes, 'public_observation_json_bytes_total': None,
                  'tokenize_pack_seconds': encoding_seconds, 'target_states': target_states,
                  'target_references': target_references, 'resolved_targets': target_resolved,
                  'peak': peak, 'seconds': time.monotonic() - started}
        for label, path in [('raw', raw_path), ('opening', opening_path), ('peak', peak_path)]:
            record[label + '_path'] = str(path.relative_to(ROOT))
            record[label + '_sha256'] = profile.digest(path)
        profile.save(directory / f"job_{job['job']:06d}.json", record)
        return record
    except BaseException:
        profile.save(directory / f"job_{job['job']:06d}.failure.json",
                     {'context': context, 'error': traceback.format_exc()})
        raise


def summarize(records):
    cohorts = defaultdict(list)
    for record in records:
        cohorts[record['task_id'] + '/' + record['strategy']].append(record)
    result = {}
    for key, rows in sorted(cohorts.items()):
        won, count = sum(r['won'] for r in rows), len(rows)
        histogram = Counter()
        for row in rows:
            histogram.update({int(k): v for k, v in row['token_histogram'].items()})
        z = 1.959963984540054
        p, denominator = won/count, 1 + z*z/count
        center = (p + z*z/(2*count))/denominator
        radius = z*math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denominator
        result[key] = {'episodes': count, 'won': won, 'win_rate': won/count,
                      'wilson_95': [max(0., center-radius), min(1., center+radius)],
                      'truncated': sum(r['truncated'] for r in rows),
                      'tokens': profile.histogram_summary(histogram),
                      'target_states': sum(r['target_states'] for r in rows),
                      'target_references': sum(r['target_references'] for r in rows),
                      'resolved_targets': sum(r['resolved_targets'] for r in rows)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    protocol, tasks = checked_protocol(args.protocol)
    output = args.output_dir.resolve()
    output.relative_to(ROOT / 'artifacts/research')
    output.mkdir(parents=True, exist_ok=False)
    directory = output / 'episodes'
    directory.mkdir()
    profile.load_encoder(protocol['encoder_root'])
    mechanics = curriculum_mechanics(tasks, protocol)
    from pvz_research import ResourceMonitor
    monitor = ResourceMonitor()
    records, started = [], time.monotonic()
    report = {'schema_version': 1, 'status': 'running', 'protocol_sha256': profile.digest(args.protocol),
              'manifest_sha256': profile.digest(ROOT / protocol['manifest']), 'scope': protocol['purpose'],
              'curriculum_mechanics': mechanics,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'coexecution': protocol['coexecution'], 'limitations': protocol['limitations']}
    with (output / '.execution.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        monitor.thread.start()
        executor = ProcessPoolExecutor(max_workers=protocol['workers'], mp_context=multiprocessing.get_context('spawn'))
        pending = {}
        jobs = iter(make_jobs(tasks, protocol))
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
                    raise RuntimeError('pool audit exceeded preregistered deadline')
                for future in completed:
                    pending.pop(future)
                    records.append(future.result())
                    submit()
                if completed and (len(records) % 30 == 0 or not pending):
                    profile.save(output / 'progress.json', {'episodes': len(records), 'expected': protocol['expected_episodes'],
                                 'seconds': time.monotonic()-started})
                    print(f"audited {len(records)}/{protocol['expected_episodes']}", flush=True)
            records.sort(key=lambda r: r['job'])
            if [r['job'] for r in records] != list(range(protocol['expected_episodes'])):
                raise ValueError('missing/duplicate jobs')
            profile.save(output / 'records.json.gz', records)
            report.update(status='complete', episodes=len(records), summaries=summarize(records),
                          records_sha256=profile.digest(output / 'records.json.gz'))
        except BaseException:
            report.update(status='failed', episodes=len(records), error=traceback.format_exc())
            profile.save(output / 'failure.json', report)
            raise
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            monitor.stop.set()
            monitor.thread.join()
            report.update(seconds=time.monotonic()-started, resources={'peak_process_tree_rss_bytes': monitor.peak_tree_rss_bytes,
                          'min_system_available_bytes': monitor.min_available_bytes, 'peak_system_swap_used_bytes': monitor.peak_swap_used_bytes})
            profile.save(output / 'report.json', report)


if __name__ == '__main__':
    main()
