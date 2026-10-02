"""Released-only native macro/reference, snapshot and record-replay audit.

The reference sends raw WAIT 1 commands, never Python step(1). Aggregate income
is annotated once. These audit controls are never PPO demonstration data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'python'))
from pvz_common import canonical_digest, sha256_file
from pvz_env import PvZEnv, TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_seed_jobs import atomic_json
from pvz_wait_events import CONDITIONS, REASON_PRECEDENCE, reference_wait

REQUIRED_COVERAGE = ['terminal','defense_lost','zombie_entered_left_zone','initial_true',
                     'zero_actual_ticks','no_event_equivalent'] + ['condition:'+name for name in CONDITIONS[1:]]


def oracle_macro_wait(env, initial, action, trace):
    """Advance raw ticks with public predicates and exactly one economic annotation."""
    responses = []

    def raw_wait(ticks):
        response = env._command(f'WAIT {ticks}')
        if not response.get('ok') or not isinstance(response.get('observation'), dict):
            raise RuntimeError('raw reference wait failed')
        responses.append(response)
        trace(response)
        return response['observation']

    start = raw_wait(0)
    if start['tick'] != initial['tick']:
        raise RuntimeError('zero wait advanced the reference clock')
    result = reference_wait(start, lambda: raw_wait(1), action['ticks'], action['until'])
    observation = result.pop('observation')
    # sun_spent is max(start.sun-end.sun,0), not the sum of per-tick positive
    # changes. The remaining numeric counters telescope; flags describe final state.
    last = responses[-1]['events']
    events = {name: sum(response['events'][name] for response in responses)
              for name in ('zombies_killed', 'plants_eaten', 'sun_produced', 'mower_triggered', 'waves_started')}
    events.update(sun_spent=max(start['sun']-observation['sun'], 0),
                  level_won=last['level_won'], level_lost=last['level_lost'])
    env._annotate_observation(observation, events)
    env._adopt_tick(observation)
    info = dict(ok=True, events=events, ticks_advanced=result['actual_ticks'], wait_result=result)
    env._record_operation(dict(kind='action', request=dict(action), action=dict(action),
                               ticks_advanced=result['actual_ticks'], wait_result=result), observation, events)
    return observation, 0., observation['terminal'], False, info


def check_protocol(path):
    protocol = json.loads(path.read_text())
    if protocol.get('release_status') != 'released':
        raise RuntimeError('native event audit is draft; publish the released protocol first')
    relative = str(path.resolve().relative_to(ROOT))
    published_ref = 'refs/remotes/delivery/research/event-wait-native-audit-v1'
    published = subprocess.run(['git', '-C', str(ROOT), 'show', published_ref+':'+relative],
                               capture_output=True, check=True).stdout
    if published != path.read_bytes():
        raise RuntimeError('protocol differs from the fetched published branch')
    for relative, expected in protocol['required_fingerprints'].items():
        if sha256_file(ROOT/relative) != expected:
            raise ValueError(f'frozen source differs: {relative}')
        remote_source = subprocess.check_output(['git','-C',str(ROOT),'show',published_ref+':'+relative])
        if canonical_digest(remote_source.hex()) != canonical_digest((ROOT/relative).read_bytes().hex()):
            raise ValueError(f'source differs from fetched publication: {relative}')
    for entry in ('task_manifest', 'reference_executable', 'candidate_executable'):
        if sha256_file(Path(protocol[entry]['path'])) != protocol[entry]['sha256']:
            raise ValueError(f'frozen native input differs: {entry}')
    for name, expected in protocol['resource_fingerprints'].items():
        if sha256_file(Path(protocol['resource_dir'])/name) != expected:
            raise ValueError(f'frozen native resource differs: {name}')
    tasks = json.loads(Path(protocol['task_manifest']['path']).read_text())['tasks']
    if [task['task_id'] for task in tasks] != protocol['all_task_ids_in_order']:
        raise ValueError('native audit task inventory changed')
    if (protocol['conditions'] != list(CONDITIONS) or protocol['durations'] != [60,150,300]
            or protocol['prefix_ticks'] != [0,3000,9000] or protocol['seeds_per_task'] != 2
            or protocol['regimes'] != ['wait_only','one_first_legal_plant']
            or protocol['required_positive_coverage'] != REQUIRED_COVERAGE):
        raise ValueError('frozen audit controls changed')
    return protocol, tasks


def _spec(task, seed):
    return TaskSpec(level=task['level'], seed=seed, playthrough=task['playthrough'],
                    wave_cap=task['wave_cap'], zombie_count_multiplier=task['zombie_count_multiplier'],
                    preplanted=tuple(tuple(p) for p in task['preplanted']))


def run_job(job, protocol, directory):
    """One process/session owns both simulators; full raw traces survive failures."""
    directory = Path(directory)
    os.setsid()
    with (directory/'worker.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        atomic_json(directory/'worker.json', dict(pid=os.getpid(), pgid=os.getpgrp(), job=job))
        try:
            task, seed, regime = job['task'], job['seed'], job['regime']
            started, rows, counts = time.monotonic(), [], Counter()
            with gzip.open(directory/'raw.jsonl.gz','wt',encoding='utf-8') as raw, \
                 PvZEnv(protocol['resource_dir'], executable=protocol['reference_executable']['path'],
                        save_dir=directory/'saves/reference', debug_replay=True) as reference, \
                 EventWaitEnv(protocol['resource_dir'], executable=protocol['candidate_executable']['path'],
                              save_dir=directory/'saves/candidate', debug_replay=True) as candidate:
                def trace(kind, payload):
                    raw.write(json.dumps(dict(kind=kind, payload=payload), separators=(',',':'))+'\n')
                    raw.flush()

                def require_equal(expected, actual, label):
                    trace(label, dict(expected=expected, actual=actual))
                    if expected != actual:
                        raise AssertionError(f'native event mismatch: {label}')

                kwargs = dict(deck=task['deck'], task=_spec(task,seed))
                old, new = reference.reset(**kwargs), candidate.reset(**kwargs)
                require_equal(old,new,'reset')
                initial = new[0]
                if regime == 'one_first_legal_plant' and initial['legal_actions']['plants']:
                    plant = dict(type='plant', **initial['legal_actions']['plants'][0])
                    old, new = reference.step(plant), candidate.step(plant)
                    require_equal(old,new,'first_legal_plant')
                    initial = new[0]
                baseline_ops = 0
                for prefix in protocol['prefix_ticks']:
                    while initial['tick'] < prefix and not initial['terminal']:
                        old, new = reference.step(dict(type='wait',ticks=60)), candidate.step(dict(type='wait',ticks=60))
                        require_equal(old,new,'fixed_prefix')
                        initial = new[0]
                        baseline_ops += 1
                        if baseline_ops > protocol['max_prefix_actions']:
                            raise RuntimeError('frozen prefix action limit reached')
                    parent_ref, parent_cand = reference.snapshot(), candidate.snapshot()
                    require_equal(reference.privileged_state(),candidate.privileged_state(),'prefix_physics_rng')
                    for maximum in protocol['durations']:
                        for condition in protocol['conditions']:
                            # Actual returns are compared against the independent public oracle.
                            reference.restore(parent_ref)
                            candidate.restore(parent_cand)
                            action = dict(type='wait', ticks=maximum, until=condition)
                            before = time.monotonic()
                            expected = oracle_macro_wait(reference, initial, action,
                                                         lambda response: trace('reference_tick',response))
                            oracle_seconds = time.monotonic()-before
                            before = time.monotonic()
                            actual = candidate.step(action)
                            macro_seconds = time.monotonic()-before
                            require_equal(expected,actual,'event_response')
                            require_equal(reference.privileged_state(),candidate.privileged_state(),'event_physics_rng')
                            # The same native snapshot must reproduce time, reasons and every public field.
                            candidate.restore(parent_cand)
                            repeated = candidate.step(action)
                            require_equal(actual,repeated,'snapshot_event_replay')
                            result = actual[-1]['wait_result']
                            counts.update(result['triggered'])
                            counts['condition:'+condition] += 'condition' in result['triggered']
                            counts['initial_true'] += result['initial_condition_satisfied']
                            counts['zero_actual_ticks'] += result['actual_ticks'] == 0
                            counts['no_event_equivalent'] += result['triggered'] == ['max_ticks']
                            if result['triggered'] == ['max_ticks']:
                                reference.restore(parent_ref)
                                fixed = reference.step(dict(type='wait', ticks=maximum))
                                expected_fixed = (*actual[:-1], {key:value for key,value in actual[-1].items()
                                                                 if key != 'wait_result'})
                                require_equal(fixed,expected_fixed,'no_event_fixed_equivalence')
                                require_equal(reference.privileged_state(),candidate.privileged_state(),'no_event_rng')
                            rows.append(dict(prefix_requested=prefix, prefix_actual=initial['tick'],
                                action=action, wait_result=result, terminal=actual[2],
                                oracle_seconds=oracle_seconds, macro_seconds=macro_seconds))
                    initial = candidate.restore(parent_cand)
                    reference.restore(parent_ref)
                    reference.release_snapshot(parent_ref)
                    candidate.release_snapshot(parent_cand)
                # Real record replay checks every stored macro and snapshot operation and debug RNG.
                record = copy.deepcopy(candidate.episode)
                atomic_json(directory/'record.json.gz', record, compressed=True)
                candidate.replay_record(record)
                trace('record_replay', dict(operations=len(record['operations']),passed=True))
            atomic_json(directory/'result.json', dict(job_id=job['job_id'], task_id=task['task_id'], seed=seed,
                regime=regime, rows=rows, counts=dict(counts), record_replay_passed=True,
                seconds=time.monotonic()-started, raw_sha256=sha256_file(directory/'raw.jsonl.gz')))
        except BaseException:
            atomic_json(directory/'failure.json', dict(job=job, error=traceback.format_exc()))
            raise


def stop_owned_job(process):
    if process.pid is None:
        return
    alive = process.is_alive()
    owned_session = getattr(process, '_owned_session', False)
    if alive and not owned_session:
        try:
            owned_session = os.getpgid(process.pid) == process.pid
        except ProcessLookupError:
            alive = False
    if alive or owned_session:
        if owned_session:
            try:
                os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:
            process.kill()  # Before setsid, no native child has been created.
    process.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args = parser.parse_args()
    protocol,tasks = check_protocol(args.protocol.resolve())
    output=args.output_dir.resolve()
    if not output.is_relative_to(Path('/home/newbiexvwu/PvZ-Portable/artifacts/research')):
        raise ValueError('raw native evidence must remain in the ignored research tree')
    output.mkdir(parents=True,exist_ok=False)
    lock=(output/'.execution.lock').open('x')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    from pvz_research import ResourceMonitor
    monitor=ResourceMonitor()
    monitor.thread.start()
    jobs=[dict(job_id=index,task=task,seed=seed,regime=regime)
          for index,(task,seed,regime) in enumerate((t,s,r) for t in tasks
          for s in t['seeds'][:protocol['seeds_per_task']] for r in protocol['regimes'])]
    started,records=time.monotonic(),[]
    process=None
    report=dict(schema_version=1,status='running',scope='native semantics only; no learning acceptance',
                protocol_sha256=sha256_file(args.protocol),expected_jobs=len(jobs))
    try:
        for job in jobs:
            directory=output/f"job_{job['job_id']:04d}"
            directory.mkdir()
            available=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024
            if available < protocol['minimum_available_ram_bytes']:
                raise RuntimeError('native audit memory gate stopped before starting this job')
            process=multiprocessing.get_context('spawn').Process(target=run_job,args=(job,protocol,str(directory)))
            process.start()
            job_started=last_message=time.monotonic()
            while process.is_alive():
                process.join(timeout=1)
                if (directory/'worker.json').exists():
                    ownership=json.loads((directory/'worker.json').read_text())
                    if ownership['pid'] != process.pid or ownership['pgid'] != process.pid:
                        raise RuntimeError('native audit worker session identity differs')
                    process._owned_session=True
                now=time.monotonic()
                if now-started >= protocol['total_timeout_seconds'] or now-job_started >= protocol['job_timeout_seconds']:
                    stop_owned_job(process)
                    raise RuntimeError('native audit exceeded frozen wall deadline; evidence retained')
                if now-last_message >= 30:
                    print(f"native event audit job={job['job_id']} elapsed={now-started:.1f}s",flush=True)
                    last_message=now
            if process.exitcode != 0:
                if (directory/'worker.json').exists():
                    ownership=json.loads((directory/'worker.json').read_text())
                    process._owned_session=(ownership['pid'] == process.pid and ownership['pgid'] == process.pid)
                raise RuntimeError(f"native audit worker failed: job={job['job_id']} exit={process.exitcode}")
            records.append(json.loads((directory/'result.json').read_text()))
            atomic_json(output/'progress.json',dict(completed_jobs=len(records),expected_jobs=len(jobs),seconds=time.monotonic()-started))
            print(f'native event audit completed {len(records)}/{len(jobs)}',flush=True)
        counts=sum((Counter(r['counts']) for r in records),Counter())
        missing=[name for name in protocol['required_positive_coverage'] if counts[name] == 0]
        report.update(status='complete',counts=dict(counts),missing_coverage=missing,
                      gate_result='pass' if not missing else 'fail',jobs=records,
                      limitation='missing real event coverage stops acceptance; no task/seed is deleted')
    except BaseException:
        if process is not None:
            stop_owned_job(process)
        report.update(status='failed',gate_result='fail',jobs=records,error=traceback.format_exc())
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join()
        report.update(seconds=time.monotonic()-started,peak_process_tree_rss_bytes=monitor.peak_tree_rss_bytes,
                      min_system_available_bytes=monitor.min_available_bytes)
        atomic_json(output/'report.json',report)
        lock.close()
    if report['gate_result'] != 'pass':
        raise RuntimeError('native event coverage gate failed; report retained, no learning release')


if __name__ == '__main__':
    main()
