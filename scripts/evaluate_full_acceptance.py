"""Evaluate the unchanged T7 full-course manifest with resumable seed shards.

One invocation evaluates one immutable research checkpoint in both declared
modes. It reports single-candidate capability checks, never whole-goal success.
Use aggregate_reports on all three initialization reports for the common-mode
two-of-three capability view. Training is not launched or changed here.
"""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]

MANIFEST = ROOT/'experiments/t7/full_level7_v1/acceptance.json'
THRESHOLDS = dict(level7_pass_rate=0.6, terrain_macro_pass_rate=0.6,
                  each_terrain_pass_rate=0.45, weakest_task_pass_rate=0.25,
                  required_initializations=2)
FULL_TASKS = {
    'level7_development':(7,'day',[0,1,2,3,4,5]),
    'full_day8':(8,'day',[0,1,2,3,4,5]),
    'full_night12':(12,'night',[8,9,10,12,14,15]),
    'full_pool26':(26,'pool',[0,1,2,3,4,16]),
    'full_fog31':(31,'fog',[8,9,10,14,15,16]),
    'full_roof41':(41,'roof',[0,1,2,3,4,33]),
}


def validate_manifest(manifest: dict) -> None:
    if manifest['thresholds'] != THRESHOLDS or manifest['modes'] != ['greedy', 'sampled']:
        raise ValueError('Original full-course thresholds/modes must remain unchanged')
    tasks = manifest['tasks']; expected = {t['task_id'] for t in tasks}
    if (len(expected) != len(tasks) or expected != set(FULL_TASKS)
            or any((t['level'],t['terrain'],t['deck']) != FULL_TASKS[t['task_id']]
                   for t in tasks)):
        raise ValueError('Require the unchanged full-course levels, decks and task identities')
    original = next(t for t in tasks if t['task_id'] == 'level7_development')
    terrains = [t for t in tasks if t['task_id'] != original['task_id']]
    if (original['level'] != 7 or original['seeds'] != list(range(30000,30256))
            or len(terrains) != 5
            or {t['terrain'] for t in terrains} != {'day','night','pool','fog','roof'}
            or any(t['wave_cap'] is not None or t['seeds'][:64] != list(range(30000,30064))
                   or len(t['seeds']) != len(set(t['seeds'])) for t in terrains)
            or any(t['wave_cap'] is not None or t['preplanted']
                   or t['sun_start'] != 50 or t['zombie_count_multiplier'] != 1
                   or t['playthrough'] != 2 for t in tasks)):
        raise ValueError('Full original level7 and five-terrain cohort is required')


def summarize(manifest: dict, grouped: dict) -> dict:
    """Compute declared thresholds from complete task/seed/mode outcomes."""
    from research_comparison_summary import outcomes
    validate_manifest(manifest)
    tasks = manifest['tasks']; expected = {t['task_id'] for t in tasks}
    original = next(t for t in tasks if t['task_id'] == 'level7_development')
    terrains = [t for t in tasks if t['task_id'] != original['task_id']]
    if set(grouped) != set(manifest['modes']):
        raise ValueError('Missing declared evaluation modes')
    result = {}
    for mode in manifest['modes']:
        if set(grouped[mode]) != expected:
            raise ValueError('Missing or substituted full-course tasks')
        per_task = {}
        for task in tasks:
            rows = grouped[mode][task['task_id']]
            seeds = [r['seed'] for r in rows]
            if (len(seeds) != len(set(seeds)) or len(seeds) != len(task['seeds'])
                    or set(seeds) != set(task['seeds'])):
                raise ValueError('Missing, repeated or substituted frozen seeds')
            if any(bool(r['terminated']) == bool(r['truncated']) for r in rows):
                raise ValueError('Normal termination and budget truncation must be distinguished')
            per_task[task['task_id']] = outcomes(rows)
        terrain_rates = {t['terrain']:per_task[t['task_id']]['win_rate'] for t in terrains}
        macro = sum(terrain_rates.values())/5
        weakest = min(v['win_rate'] for v in per_task.values())
        checks = dict(level7=per_task[original['task_id']]['win_rate'] >= THRESHOLDS['level7_pass_rate'],
            terrain_macro=macro >= THRESHOLDS['terrain_macro_pass_rate'],
            each_terrain=all(r >= THRESHOLDS['each_terrain_pass_rate'] for r in terrain_rates.values()),
            weakest_task=weakest >= THRESHOLDS['weakest_task_pass_rate'])
        result[mode] = dict(per_task=per_task, terrain_pass_rates=terrain_rates,
            terrain_macro_pass_rate=macro, weakest_task_pass_rate=weakest,
            checks=checks, single_candidate_pass=all(checks.values()))
    return result


def _origin_seed(state: dict, current: int) -> int:
    origin = current
    provenance = state.get('initialization_provenance')
    while provenance:
        origin = provenance['source_initialization_seed']
        provenance = provenance.get('source_initialization_provenance')
    return origin


def aggregate_reports(reports: list[dict]) -> dict:
    """Report two-of-three for each shared mode; never mix modes or origins."""
    if (len(reports) != 3 or any(r['status'] != 'complete' for r in reports)
            or {r['initialization_seed'] for r in reports} != {0,1,2}
            or {r['origin_initialization_seed'] for r in reports} != {0,1,2}):
        raise ValueError('Require complete reports for all three independent initializations')
    first = reports[0]
    validate_manifest(first['manifest'])
    if first['thresholds'] != THRESHOLDS:
        raise ValueError('Aggregate must use the original capability thresholds')
    for report in reports:
        if any(report[key] != first[key] for key in ('manifest','model_config','simulator_sha256','thresholds')):
            raise ValueError('Reports must use the same full manifest, model configuration and native build')
        if set(report['modes']) != {'greedy','sampled'}:
            raise ValueError('Both declared modes must be retained')
    per_mode = {}
    for mode in ('greedy','sampled'):
        passed = [r['initialization_seed'] for r in reports if r['modes'][mode]['single_candidate_pass']]
        per_mode[mode] = dict(passed_initialization_seeds=sorted(passed),
            required_initializations=THRESHOLDS['required_initializations'],
            two_of_three_capability_pass=len(passed) >= THRESHOLDS['required_initializations'])
    return dict(schema_version=1, status='three_initialization_full_capability_summary',
        per_mode=per_mode, scope='Capability checks separately in each common deployment mode. No pooling across modes or selection of different modes per initialization. This summary does not prove T5/T6 or completion of the overall research goal.')


def _worker(job_id: int) -> dict:
    import train_pvz_ppo_task_family as family
    actual_id, episode = family._evaluation_worker(job_id)
    if actual_id != job_id:
        raise ValueError('Evaluation worker returned the wrong job identity')
    return dict(seed=job_id, episode=episode)


def collect(jobs: dict, weights: dict, model_config: dict, resource_dir: Path,
            output: Path, metadata: dict, *, workers: int, worker_threads: int) -> list[dict]:
    import train_pvz_ppo_task_family as family
    from pvz_seed_jobs import run_seed_jobs, seed_job_directory
    directory = seed_job_directory(output, 'full_acceptance', metadata)
    saved = run_seed_jobs(sorted(jobs), directory, metadata, _worker,
        workers=workers, initializer=family._init_evaluation_worker,
        initargs=(str(resource_dir),weights,jobs,worker_threads,'cpu',model_config),
        label='full_acceptance')
    return [row['episode'] for row in saved]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--resource-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--worker-threads',type=int,default=1)
    args = parser.parse_args()
    if args.workers < 1 or args.worker_threads < 1:
        raise ValueError('Worker counts must be positive')
    import torch
    from pvz_agent_model import GameplayModelV1, configure_torch_threads, model_architecture_version
    from pvz_common import ENV_PROTOCOL_VERSION, OBSERVATION_VERSION, TASK_VERSION, sha256_file
    from pvz_research import ResourceMonitor
    from pvz_seed_jobs import atomic_json
    import t4_capability_profile as profile
    configure_torch_threads(1)
    output = args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    with (output/'.execution.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior_path = output/'report.json'
        if not prior_path.exists(): prior_path = output/'progress.json'
        prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
        if prior and prior['status'] == 'complete':
            raise ValueError('Completed evidence is preserved; use its report or a new output directory')
        if prior and prior['checkpoint'] != str(args.checkpoint.resolve()):
            raise ValueError('Resume the same checkpoint; preserve other evidence in its existing directory')
        started = time.monotonic(); monitor = ResourceMonitor(); monitor.thread.start()
        report = dict(prior) if prior else dict(schema_version=1,checkpoint=str(args.checkpoint.resolve()))
        if prior:
            report['prior_attempts'] = prior.get('prior_attempts',[]) + [{k:prior[k] for k in (
                'status','seconds','error','resources') if k in prior}]
        report['status'] = 'running'
        def interrupt(*_): raise KeyboardInterrupt
        for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,interrupt)
        try:
            checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
            if (checkpoint.get('research_version') != 1
                    or checkpoint['model_architecture_version'] != model_architecture_version(checkpoint['config'])
                    or checkpoint.get('value_semantics') != 'research_explicit_return_v1'
                    or checkpoint['provenance']['protocol_version'] != ENV_PROTOCOL_VERSION
                    or checkpoint['provenance']['observation_version'] != OBSERVATION_VERSION
                    or checkpoint['provenance']['task_version'] != TASK_VERSION):
                raise ValueError('Require a compatible immutable research checkpoint')
            native = sha256_file(ROOT/'build/pvz-portable')
            if checkpoint['provenance']['fingerprints']['simulator'] != native:
                raise ValueError('Checkpoint and evaluation native builds differ; preserve and report')
            manifest = json.loads(MANIFEST.read_text()); validate_manifest(manifest)
            state = checkpoint['training_state']
            model = GameplayModelV1(checkpoint['config']).eval(); model.load_state_dict(checkpoint['state_dict'])
            weights = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            identity = profile._state_sha256(weights)
            labels = []; jobs = {}
            for mode in manifest['modes']:
                for task in manifest['tasks']:
                    for seed in task['seeds']:
                        labels.append((mode,task['task_id']))
                        jobs[len(jobs)] = dict(task=task,seed=seed,deterministic=mode=='greedy',
                            max_actions=4000,allow_truncation=True,action_seed=seed+170_000)
            grouped = {mode:{t['task_id']:[] for t in manifest['tasks']} for mode in manifest['modes']}
            metadata = dict(manifest=manifest,model_state_sha256=identity,
                model_config=model.config,simulator_sha256=native,max_actions=4000,
                action_seed_rule='environment_seed+170000',worker_device='cpu',
                evaluation_core_fingerprints={relative:sha256_file(ROOT/relative) for relative in (
                    'python/pvz_agent_model.py','python/pvz_env.py','python/pvz_event_env.py',
                    'python/pvz_wait_events.py','python/train_pvz_ppo_task_family.py',
                    'scripts/t4_capability_profile.py')})
            if prior and any(key in prior and prior[key] != value for key,value in (
                    ('model_state_sha256',identity),('manifest',manifest),('simulator_sha256',native))):
                raise ValueError('Interrupted evidence identity changed; preserve it and use a new directory')
            report.update(initialization_seed=checkpoint['experiment_config']['initialization_seed'],
                origin_initialization_seed=_origin_seed(state,checkpoint['experiment_config']['initialization_seed']),
                manifest=manifest,thresholds=manifest['thresholds'],model_config=model.config,
                simulator_sha256=native,model_state_sha256=identity,
                training_source_revision=checkpoint['provenance']['commit'],
                source_experiment_config=checkpoint['experiment_config'],
                source_counters=state['counters'],source_updates=state['updates'],
                source_active_wall_seconds=state['wall_seconds'],
                initialization_provenance=state.get('initialization_provenance',{'kind':'random'}),
                workers=args.workers,worker_threads=args.worker_threads)
            atomic_json(output/'progress.json',report)
            records = collect(jobs,weights,model.config,args.resource_dir,output,metadata,
                              workers=args.workers,worker_threads=args.worker_threads)
            for (mode,task_id), record in zip(labels,records,strict=True): grouped[mode][task_id].append(record)
            raw_path = output/'seed_results.json.gz'
            atomic_json(raw_path,dict(schema_version=1,manifest=manifest,simulator_sha256=native,
                model_state_sha256=identity,seed_results=grouped),compressed=True)
            report.update(status='complete',modes=summarize(manifest,grouped),
                raw_seed_results_path='seed_results.json.gz',
                scope='All576 frozen cases per mode, all failures and budget truncations retained. Per-mode single-candidate checks only; other initializations, structure comparisons and whole-goal completion require their own evidence. Source training cost and this additional CPU evaluation cost reported separately.')
        except BaseException as error:
            report.update(status='failed_preserved',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            monitor.stop.set(); monitor.thread.join()
            report.update(seconds=time.monotonic()-started,resources=monitor.snapshot(),
                          cuda_initialized=torch.cuda.is_initialized())
            atomic_json(output/'report.json',report); atomic_json(output/'progress.json',report)
        print(json.dumps(dict(status=report['status'],seconds=report['seconds'],
            single_candidate_pass={m:v['single_candidate_pass'] for m,v in report['modes'].items()})),flush=True)


if __name__ == '__main__': main()
