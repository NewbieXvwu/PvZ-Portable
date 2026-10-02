"""Controlled headless intro finalization audit; no learning or injected entities.

The audit-only reference executes the ordinary final two CutScene::Update calls.
The candidate reuses AnimateBoard and RemoveCutsceneZombies before StartPlaying.
Both share unchanged game/task/profile/seed semantics after initialization.
"""
from __future__ import annotations
import argparse
import fcntl
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import time
import traceback
import research_event_wait_native_audit as native
from pvz_common import canonical_digest, sha256_file
from pvz_env import PvZEnv
from pvz_seed_jobs import atomic_json

ROOT = native.ROOT
MAIN = Path('/home/newbiexvwu/PvZ-Portable')


def require_equal(expected, actual, label, trace):
    trace(label, dict(expected=expected, actual=actual))
    if expected != actual:
        raise AssertionError('intro fidelity differs: '+label)


def jobs_for(tasks):
    return [dict(job_id=i,task=t,seed=s,regime=r) for i,(t,s,r) in enumerate(
        (t,s,r) for t in tasks for s in t['seeds'][:2]
        for r in ('wait_only','one_first_legal_plant'))]


def checked_protocol(path):
    p=json.loads(path.read_text())
    if p.get('release_status')!='released':raise RuntimeError('intro audit needs a published released protocol')
    ref='refs/remotes/delivery/research/env-intro-state-v1'
    paths=[str(path.resolve().relative_to(ROOT)),*p['required_fingerprints']]
    for name in paths:
        data=(ROOT/name).read_bytes()
        if name in p['required_fingerprints'] and sha256_file(ROOT/name)!=p['required_fingerprints'][name]:
            raise ValueError('frozen intro source differs: '+name)
        if subprocess.check_output(['git','-C',str(ROOT),'show',ref+':'+name])!=data:
            raise ValueError('intro audit source/protocol differs from publication: '+name)
    for block in ('legacy_executable','candidate_executable','reference_executable','task_manifest'):
        if sha256_file(Path(p[block]['path']))!=p[block]['sha256']:
            raise ValueError('intro audit artifact differs: '+block)
    for name,expected in p['resource_fingerprints'].items():
        if sha256_file(Path(p['resource_dir'])/name)!=expected:raise ValueError('intro resource differs: '+name)
    for expected in p['retained_evidence']:
        if sha256_file(Path(expected['path']))!=expected['sha256']:raise ValueError('retained evidence differs')
    tasks=json.loads(Path(p['task_manifest']['path']).read_text())['tasks']
    if ([t['task_id'] for t in tasks]!=p['all_task_ids_in_order'] or len(tasks)!=30
            or len(jobs_for(tasks))!=120 or p['expected_jobs']!=120
            or p['maximum_actions']!=400 or p['maximum_ticks']!=24000
            or p['action']!=dict(type='wait',ticks=60)
            or p['job_timeout_seconds']!=120 or p['total_timeout_seconds']!=1800
            or p['minimum_available_ram_bytes']!=12*1024**3):
        raise ValueError('frozen intro control/resource inventory differs')
    return p,tasks


def run_job(job,p,directory):
    directory=Path(directory);os.setsid()
    with (directory/'worker.log').open('x') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        atomic_json(directory/'worker.json',dict(pid=os.getpid(),pgid=os.getpgrp(),job=job))
        try:
            started=time.monotonic();task=job['task'];kwargs=dict(deck=task['deck'],task=native._spec(task,job['seed']))
            with gzip.open(directory/'raw.jsonl.gz','wt',encoding='utf-8') as raw, \
                 PvZEnv(p['resource_dir'],executable=p['reference_executable']['path'],save_dir=directory/'reference/saves',debug_replay=True) as reference, \
                 PvZEnv(p['resource_dir'],executable=p['candidate_executable']['path'],save_dir=directory/'candidate/saves',debug_replay=True) as candidate:
                def trace(kind,payload):
                    raw.write(json.dumps(dict(kind=kind,payload=payload),separators=(',',':'))+'\n');raw.flush()
                def equal(a,b,label):require_equal(a,b,label,trace)
                with PvZEnv(p['resource_dir'],executable=p['legacy_executable']['path'],save_dir=directory/'legacy/saves',debug_replay=True) as legacy:
                    old=legacy.reset(**kwargs);old_state=legacy.privileged_state()
                    trace('retained_legacy_reset',dict(response=old,state=old_state))
                a,b=reference.reset(**kwargs),candidate.reset(**kwargs);equal(a,b,'reset_response')
                equal(reference.privileged_state(),candidate.privileged_state(),'reset_physics_rng')
                initial=b[0];trace('intro_initial_defenses',dict(legacy=old[0]['defenses'],candidate=initial['defenses']))
                current=initial;steps=0;physical=1;triggers=0;snapshots=0
                if job['regime']=='one_first_legal_plant' and current['legal_actions']['plants']:
                    action=dict(type='plant',**current['legal_actions']['plants'][0])
                    a,b=reference.step(action),candidate.step(action);equal(a,b,'first_plant_response')
                    equal(reference.privileged_state(),candidate.privileged_state(),'first_plant_physics_rng');physical+=1;current=b[0]
                while not current['terminal'] and steps<p['maximum_actions'] and current['tick']<p['maximum_ticks']:
                    a,b=reference.step(p['action']),candidate.step(p['action']);equal(a,b,'fixed_response')
                    equal(reference.privileged_state(),candidate.privileged_state(),'fixed_physics_rng');physical+=1;current=b[0]
                    triggers+=b[-1]['events']['mower_triggered'];steps+=1
                    if steps in (1,25,100,200):
                        ra,ca=reference.snapshot(),candidate.snapshot()
                        before=reference.privileged_state();reference.restore(ra);candidate.restore(ca)
                        equal(before,reference.privileged_state(),'reference_snapshot_restore')
                        equal(before,candidate.privileged_state(),'candidate_snapshot_restore')
                        reference.release_snapshot(ra);candidate.release_snapshot(ca);snapshots+=1
                for name,env in [('reference',reference),('candidate',candidate)]:
                    record=native.saved_replay_record(env,directory/name)
                    env.replay_record(record,directory/name)
                    trace(name+'_real_record_replay',dict(passed=True,operations=len(record['operations'])))
            atomic_json(directory/'result.json',dict(job_id=job['job_id'],task_id=task['task_id'],seed=job['seed'],regime=job['regime'],
                initial_legacy_defenses=old[0]['defenses'],initial_candidate_defenses=initial['defenses'],
                physics_comparisons=physical,steps=steps,snapshots=snapshots,mower_triggered=triggers,
                final_tick=current['tick'],terminal=current['terminal'],result=current['result'],
                bounded_stop=not current['terminal'],record_replays_passed=2,seconds=time.monotonic()-started,
                raw_sha256=sha256_file(directory/'raw.jsonl.gz')))
        except BaseException:
            atomic_json(directory/'failure.json',dict(job=job,error=traceback.format_exc()));raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args();p,tasks=checked_protocol(args.protocol.resolve());out=args.output_dir.resolve()
    out.relative_to(MAIN/'artifacts/research');out.mkdir(parents=True,exist_ok=False)
    lock=(out/'.execution.lock').open('x');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    from pvz_research import ResourceMonitor
    monitor=ResourceMonitor();monitor.thread.start();started=time.monotonic();records=[];process=None
    report=dict(schema_version=1,status='running',gate_result='unresolved',scope=p['purpose'],protocol_sha256=sha256_file(args.protocol),expected_jobs=120,
                limitations=p['limitations'],binary_fingerprints={k:p[k] for k in ('legacy_executable','reference_executable','candidate_executable')})
    try:
        for job in jobs_for(tasks):
            available=int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024
            if available<p['minimum_available_ram_bytes']:raise RuntimeError('intro audit RAM gate stopped before job start')
            directory=out/f"job_{job['job_id']:04d}";directory.mkdir()
            process=multiprocessing.get_context('spawn').Process(target=run_job,args=(job,p,str(directory)))
            process.start();job_started=time.monotonic()
            while process.is_alive():
                process.join(timeout=1)
                if (directory/'worker.json').exists():
                    owner=json.loads((directory/'worker.json').read_text())
                    if owner['pid']!=process.pid or owner['pgid']!=process.pid:raise RuntimeError('intro audit owned session identity differs')
                    process._owned_session=True
                if time.monotonic()-job_started>=p['job_timeout_seconds'] or time.monotonic()-started>=p['total_timeout_seconds']:
                    raise RuntimeError('intro audit frozen deadline exceeded')
            if process.exitcode!=0:raise RuntimeError(f"intro audit job failed: {job['job_id']} exit={process.exitcode}")
            records.append(json.loads((directory/'result.json').read_text()))
            atomic_json(out/'progress.json',dict(completed_jobs=len(records),expected_jobs=120,seconds=time.monotonic()-started))
            print(f'intro fidelity {len(records)}/120',flush=True)
        triggers=sum(r['mower_triggered'] for r in records)
        report.update(status='complete',gate_result='pass' if triggers else 'fail',jobs=records,
                      missing_coverage=[] if triggers else ['natural_mower_trigger'],mower_triggered=triggers,
                      physical_comparisons=sum(r['physics_comparisons'] for r in records),real_record_replays=240,
                      all_candidate_reference_fields_equal=True)
    except BaseException:
        if process is not None:native.stop_owned_job(process)
        report.update(status='failed',gate_result='fail',jobs=records,error=traceback.format_exc());raise
    finally:
        monitor.stop.set();monitor.thread.join();report.update(seconds=time.monotonic()-started,
            peak_process_tree_rss_bytes=monitor.peak_tree_rss_bytes,min_system_available_bytes=monitor.min_available_bytes,
            peak_swap_used_bytes=monitor.peak_swap_used_bytes)
        atomic_json(out/'report.json',report);lock.close()
    if report['gate_result']!='pass':raise RuntimeError('intro fidelity positive natural mower coverage absent; preserve scene')

if __name__=='__main__':main()
