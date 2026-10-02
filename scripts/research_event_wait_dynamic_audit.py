"""Supplement real missing event coverage at public dynamic prefixes.

The original120 jobs/6480 branches remain failed and are required verbatim.
Scout control is a declared engineering WAIT60, never PPO demonstration data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import research_event_wait_native_audit as original
from pvz_common import sha256_file
from pvz_seed_jobs import atomic_json
from pvz_wait_events import public_state

ROOT = original.ROOT
TARGETS = ('defense_lost', 'condition:wave_changed', 'condition:packet_became_ready')


def public_edges(before, after):
    """Only public fields locate a witnessed edge; no hidden future wave access."""
    a, b = public_state(before), public_state(after)
    found = []
    if a.ready_defenses-b.ready_defenses:
        found.append(('defense_lost', 'timeout'))
    if b.wave != a.wave:
        found.append(('condition:wave_changed', 'wave_changed'))
    if b.ready_packets-a.ready_packets:
        found.append(('condition:packet_became_ready', 'packet_became_ready'))
    return found


def check_protocol(path):
    protocol = json.loads(path.read_text())
    if protocol.get('release_status') != 'released':
        raise RuntimeError('dynamic native coverage requires a released published protocol')
    ref = 'refs/remotes/delivery/research/event-wait-native-audit-v1'
    paths = [str(path.resolve().relative_to(ROOT)), *protocol['required_fingerprints']]
    for relative in paths:
        data = (ROOT/relative).read_bytes()
        if relative in protocol['required_fingerprints'] and sha256_file(ROOT/relative) != protocol['required_fingerprints'][relative]:
            raise ValueError('frozen dynamic audit source differs: '+relative)
        if subprocess.check_output(['git','-C',str(ROOT),'show',ref+':'+relative]) != data:
            raise RuntimeError('dynamic audit source/protocol differs from publication: '+relative)
    prior = protocol['prior_audit']
    prior_path, report_path = Path(prior['protocol_path']), Path(prior['report_path'])
    if sha256_file(prior_path) != prior['protocol_sha256'] or sha256_file(report_path) != prior['report_sha256']:
        raise ValueError('original native audit prerequisite bytes differ')
    native, tasks = original.check_protocol(prior_path)
    report = json.loads(report_path.read_text())
    if (report['protocol_sha256'] != prior['protocol_sha256'] or report['status'] != 'complete'
            or report['gate_result'] != 'fail' or report['missing_coverage'] != list(TARGETS)
            or len(report['jobs']) != 120):
        raise ValueError('require the retained complete120-job audit with these exact three missing edges')
    for index, job in enumerate(report['jobs']):
        directory = report_path.parent/f'job_{index:04d}'
        if (job['job_id'] != index or len(job['rows']) != 54 or job['record_replay_passed'] is not True
                or json.loads((directory/'result.json').read_text()) != job
                or sha256_file(directory/'raw.jsonl.gz') != job['raw_sha256']):
            raise ValueError('retained original native branches/raw differ')
        with gzip.open(directory/'record.json.gz','rt') as stream:
            record = json.load(stream)
        manifest = json.loads((directory/record['manifest']['path']).read_text())
        if (original.canonical_digest(manifest) != record['manifest']['sha256']
                or sha256_file(directory/manifest['working_tree_patch']) != manifest['working_tree_patch_sha256']):
            raise ValueError('retained original native manifest/patch chain differs')
    if (protocol['targets'] != list(TARGETS) or protocol['durations'] != [60,150,300]
            or protocol['scout_action'] != dict(type='wait',ticks=60,until='timeout')
            or protocol['maximum_scout_actions'] != 400 or protocol['maximum_scout_ticks'] != 24000
            or protocol['expected_jobs'] != 120 or protocol['job_timeout_seconds'] != 120
            or protocol['total_timeout_seconds'] != 3600
            or protocol['minimum_available_ram_bytes'] != 12*1024**3):
        raise ValueError('frozen supplemental controls/resource limits differ')
    return protocol, native, tasks, report


def run_job(job, protocol, native, directory):
    directory = Path(directory)
    os.setsid()
    with (directory/'worker.log').open('x') as log:
        os.dup2(log.fileno(),1); os.dup2(log.fileno(),2)
        atomic_json(directory/'worker.json',dict(pid=os.getpid(),pgid=os.getpgrp(),job=job))
        try:
            started, counts, rows, found = time.monotonic(), Counter(), [], set()
            task, seed, regime = job['task'], job['seed'], job['regime']
            with gzip.open(directory/'raw.jsonl.gz','wt',encoding='utf-8') as raw, \
                 original.PvZEnv(native['resource_dir'],executable=native['reference_executable']['path'],
                                save_dir=directory/'saves/reference',debug_replay=True) as reference, \
                 original.EventWaitEnv(native['resource_dir'],executable=native['candidate_executable']['path'],
                                     save_dir=directory/'saves/candidate',debug_replay=True) as candidate:
                def trace(kind, payload):
                    raw.write(json.dumps(dict(kind=kind,payload=payload),separators=(',',':'))+'\n'); raw.flush()
                def equal(a,b,label):
                    trace(label,dict(expected=a,actual=b))
                    if a != b:
                        raise AssertionError('dynamic native mismatch: '+label)
                def macro(action):
                    expected=original.oracle_macro_wait(reference,current,action,
                                                        lambda response:trace('reference_tick',response))
                    actual=candidate.step(action)
                    equal(expected,actual,'macro_response')
                    equal(reference.privileged_state(),candidate.privileged_state(),'macro_physics_rng')
                    result=actual[-1]['wait_result']
                    counts.update(result['triggered'])
                    counts['condition:'+action['until']] += 'condition' in result['triggered']
                    return actual
                kwargs=dict(deck=task['deck'],task=original._spec(task,seed))
                a,b=reference.reset(**kwargs),candidate.reset(**kwargs);equal(a,b,'reset');current=b[0]
                if regime=='one_first_legal_plant' and current['legal_actions']['plants']:
                    plant=dict(type='plant',**current['legal_actions']['plants'][0])
                    a,b=reference.step(plant),candidate.step(plant);equal(a,b,'first_legal_plant');current=b[0]
                scouts=0
                while (not current['terminal'] and scouts < protocol['maximum_scout_actions']
                       and current['tick'] < protocol['maximum_scout_ticks']):
                    parent_ref,parent_cand=reference.snapshot(),candidate.snapshot()
                    scout=macro(protocol['scout_action']);scouts+=1
                    edges=[pair for pair in public_edges(current,scout[0]) if pair[0] not in found]
                    for name,condition in edges:
                        for duration in protocol['durations']:
                            reference.restore(parent_ref);candidate.restore(parent_cand)
                            action=dict(type='wait',ticks=duration,until=condition)
                            actual=macro(action)
                            candidate.restore(parent_cand);repeated=candidate.step(action)
                            equal(actual,repeated,'snapshot_macro_replay')
                            rows.append(dict(edge=name,prefix_tick=current['tick'],action=action,
                                             wait_result=actual[-1]['wait_result']))
                        found.add(name)
                    if edges:
                        reference.restore(parent_ref);candidate.restore(parent_cand)
                        repeated=macro(protocol['scout_action']);equal(scout,repeated,'scout_prefix_restore')
                    current=scout[0]
                    reference.release_snapshot(parent_ref);candidate.release_snapshot(parent_cand)
                record=original.saved_replay_record(candidate,directory)
                candidate.replay_record(record,directory)
                trace('record_replay',dict(passed=True,operations=len(record['operations'])))
            atomic_json(directory/'result.json',dict(job_id=job['job_id'],task_id=task['task_id'],seed=seed,
                regime=regime,scouts=scouts,final_tick=current['tick'],terminal=current['terminal'],
                bounded_stop=not current['terminal'],witnessed_edges=sorted(found),rows=rows,counts=dict(counts),
                record_replay_passed=True,seconds=time.monotonic()-started,raw_sha256=sha256_file(directory/'raw.jsonl.gz')))
        except BaseException:
            atomic_json(directory/'failure.json',dict(job=job,error=traceback.format_exc()));raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();protocol,native,tasks,prior=check_protocol(args.protocol.resolve())
    output=args.output_dir.resolve()
    output.relative_to(Path('/home/newbiexvwu/PvZ-Portable/artifacts/research'))
    output.mkdir(parents=True,exist_ok=False)
    lock=(output/'.execution.lock').open('x');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    from pvz_research import ResourceMonitor
    monitor=ResourceMonitor();monitor.thread.start()
    jobs=[dict(job_id=i,task=t,seed=s,regime=r) for i,(t,s,r) in enumerate(
        (t,s,r) for t in tasks for s in t['seeds'][:2] for r in native['regimes'])]
    started,records,process=time.monotonic(),[],None
    report=dict(schema_version=1,status='running',scope='Retained6480 original branches plus public dynamic native coverage; no PPO learning',
                protocol_sha256=sha256_file(args.protocol),expected_jobs=len(jobs),
                prior_report_sha256=protocol['prior_audit']['report_sha256'])
    try:
        for job in jobs:
            available=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024
            if available < protocol['minimum_available_ram_bytes']:
                raise RuntimeError('dynamic native RAM gate stopped before job start')
            directory=output/f"job_{job['job_id']:04d}";directory.mkdir()
            process=multiprocessing.get_context('spawn').Process(target=run_job,args=(job,protocol,native,str(directory)))
            process.start();job_started=time.monotonic()
            while process.is_alive():
                process.join(timeout=1)
                if (directory/'worker.json').exists():
                    owner=json.loads((directory/'worker.json').read_text())
                    if owner['pid']!=process.pid or owner['pgid']!=process.pid:
                        raise RuntimeError('dynamic native session identity differs')
                    process._owned_session=True
                if time.monotonic()-job_started>=protocol['job_timeout_seconds'] or time.monotonic()-started>=protocol['total_timeout_seconds']:
                    raise RuntimeError('dynamic native frozen deadline exceeded')
            if process.exitcode!=0:
                raise RuntimeError(f"dynamic native job failed: {job['job_id']} exit={process.exitcode}")
            records.append(json.loads((directory/'result.json').read_text()))
            atomic_json(output/'progress.json',dict(completed_jobs=len(records),expected_jobs=len(jobs),seconds=time.monotonic()-started))
            print(f'dynamic native {len(records)}/{len(jobs)}',flush=True)
        counts=sum((Counter(r['counts']) for r in records),Counter())
        combined=counts+Counter(prior['counts'])
        missing=[name for name in original.REQUIRED_COVERAGE if combined[name]==0]
        report.update(status='complete',gate_result='pass' if not missing else 'fail',jobs=records,
                      supplemental_counts=dict(counts),combined_counts=dict(combined),missing_coverage=missing,
                      original_gate_result_preserved='fail',original_jobs=120,original_branches=6480)
    except BaseException:
        if process is not None:original.stop_owned_job(process)
        report.update(status='failed',gate_result='fail',jobs=records,error=traceback.format_exc());raise
    finally:
        monitor.stop.set();monitor.thread.join()
        report.update(seconds=time.monotonic()-started,peak_process_tree_rss_bytes=monitor.peak_tree_rss_bytes,
                      min_system_available_bytes=monitor.min_available_bytes)
        atomic_json(output/'report.json',report);lock.close()
    if report['gate_result']!='pass':
        raise RuntimeError('combined native positive coverage failed; all original and supplemental scenes retained')


if __name__=='__main__':main()
