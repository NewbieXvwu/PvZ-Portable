"""Score an immutable width candidate on its unchanged progress cohort.

The caller selects a completed policy before a preregistered wall cap. This
entry scores that policy; it does not select a winner or certify equal spent
time. Extra CPU assessment cost is separate from the source training wall.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--wall-cap-seconds', type=float, required=True)
    parser.add_argument('--resource-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reuse-evaluation', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from pvz_agent_model import configure_torch_threads, model_architecture_version
    from pvz_common import ENV_PROTOCOL_VERSION, OBSERVATION_VERSION, TASK_VERSION, sha256_file
    from pvz_research import ResourceMonitor
    from pvz_value import VALUE_SEMANTICS
    from pvz_seed_jobs import atomic_json, run_seed_jobs, seed_job_directory
    from pvz_progress_metrics import paired_idle_summary
    from evaluate_full_acceptance import _worker
    from research_comparison_summary import summarize_evaluation
    import t4_capability_profile as profile
    import train_pvz_ppo_task_family as family

    configure_torch_threads(1)
    queue = json.loads((ROOT/'experiments/t6/informative_width_v1/comparison_v1.json').read_text())
    if args.wall_cap_seconds not in queue['common_wall_budgets_seconds']:
        raise ValueError('Require an unchanged preregistered width wall cap')
    run = args.run_dir.resolve()
    config = json.loads((run/'experiment_config.json').read_text())
    candidates = [json.loads((ROOT/e['config']).read_text()) for e in queue['order']]
    if config not in candidates:
        raise ValueError('Require one of the nine frozen width candidates')
    output = args.output_dir.resolve()
    output.relative_to(ROOT/'artifacts/research')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.execution.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        prior_path = output/'report.json'
        prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
        if prior and (not args.resume or prior['status'] == 'complete'):
            raise ValueError('Preserve completed/failed evidence; resume only an unfinished score explicitly')
        started = time.monotonic()
        monitor = ResourceMonitor(); monitor.thread.start()
        report = dict(schema_version=1, status='running', checkpoint=str(args.checkpoint.resolve()),
            wall_cap_seconds=args.wall_cap_seconds,
            prior_attempts=(prior.get('prior_attempts', []) + [{k:prior[k] for k in
                ('seconds','status','error') if k in prior}]) if prior else [])
        def interrupt(*_): raise KeyboardInterrupt
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, interrupt)
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
            state, provenance = checkpoint['training_state'], checkpoint['provenance']
            if (checkpoint.get('research_version') != 1
                    or checkpoint['experiment_config'] != config
                    or checkpoint.get('value_semantics') not in {
                        'research_explicit_return_v1', VALUE_SEMANTICS}
                    or checkpoint['model_architecture_version'] != model_architecture_version(checkpoint['config'])
                    or provenance['protocol_version'] != ENV_PROTOCOL_VERSION
                    or provenance['observation_version'] != OBSERVATION_VERSION
                    or provenance['task_version'] != TASK_VERSION):
                raise ValueError('Immutable checkpoint/config/interface mismatch')
            if (state.get('initialization_provenance') or not state.get('invocations')
                    or state['invocations'][0]['kind'] != 'random_initialization'):
                raise ValueError('Require the candidate\'s genuine random origin')
            native = sha256_file(ROOT/'build/pvz-portable')
            if provenance['fingerprints']['simulator'] != native:
                raise ValueError('Checkpoint and scoring native builds differ')
            if state['wall_seconds'] > args.wall_cap_seconds:
                raise ValueError('Selected policy exceeds the frozen wall cap')
            manifest = json.loads((ROOT/config['evaluation']['manifest']).read_text())
            tasks, modes = manifest['tasks'], config['evaluation']['modes']
            if sum(len(t['seeds']) for t in tasks) != 136 or modes != ['greedy','sampled']:
                raise ValueError('Require every unchanged progress task/seed in both modes')
            weights = checkpoint['state_dict']
            metadata = dict(checkpoint_state_sha256=profile._state_sha256(weights),
                model_config=checkpoint['config'], simulator_sha256=native,
                tasks=tasks, modes=modes, max_actions=config['runtime']['max_actions'],
                action_seed_rule='environment_seed+170000', worker_device='cpu')
            metadata['evaluation_core_fingerprints'] = {relative:sha256_file(ROOT/relative)
                for relative in ('python/pvz_agent_model.py','python/pvz_env.py',
                    'python/pvz_event_env.py','python/pvz_wait_events.py',
                    'python/train_pvz_ppo_task_family.py','scripts/t4_capability_profile.py')}
            if any(provenance['fingerprints'].get(path) != value for path, value
                   in metadata['evaluation_core_fingerprints'].items()):
                raise ValueError('Width scoring must retain the training evaluation semantics')
            if prior and (prior['checkpoint'] != report['checkpoint']
                    or prior['wall_cap_seconds'] != args.wall_cap_seconds
                    or prior.get('metadata') != metadata):
                raise ValueError('Resume requires the same policy, cohort and wall cap')
            report.update(metadata=metadata, experiment_id=config['experiment_id'],
                initialization_seed=config['initialization_seed'], source_revision=provenance['commit'],
                source_counters=state['counters'], source_updates=state['updates'],
                source_active_wall_seconds=state['wall_seconds'],
                unused_wall_seconds=args.wall_cap_seconds-state['wall_seconds'],
                source_resources=state['resources'], workers=2, worker_threads=1)
            atomic_json(output/'progress.json', report)
            if args.reuse_evaluation:
                matches = [p for p in state['learning_curve']
                           if p['counters'] == state['counters'] and p['updates'] == state['updates']]
                if len(matches) != 1:
                    raise ValueError('Selected policy has no existing evaluation at its exact counters')
                with gzip.open(run/matches[0]['raw_seed_results_path'], 'rt') as stream:
                    payload = json.load(stream)
                report.update(extra_model_cases=0, reused_evaluation=matches[0]['raw_seed_results_path'])
            else:
                jobs, labels = {}, []
                for mode in modes:
                    for task in tasks:
                        for seed in task['seeds']:
                            labels.append((mode, task['task_id']))
                            jobs[len(jobs)] = dict(task=task, seed=seed, deterministic=mode=='greedy',
                                max_actions=metadata['max_actions'], allow_truncation=True,
                                action_seed=seed+170_000)
                directory = seed_job_directory(output, 'width_wall_score', metadata)
                rows = run_seed_jobs(sorted(jobs), directory, metadata, _worker,
                    workers=2, initializer=family._init_evaluation_worker,
                    initargs=(str(args.resource_dir.resolve()), weights, jobs, 1, 'cpu', checkpoint['config']),
                    label='width_wall_score')
                grouped = {mode:{t['task_id']:[] for t in tasks} for mode in modes}
                for label, row in zip(labels, rows, strict=True):
                    grouped[label[0]][label[1]].append(row['episode'])
                payload = dict(counters=state['counters'], experiment_id=config['experiment_id'],
                               model_config=checkpoint['config'], seed_results=grouped)
                report.update(extra_model_cases=len(jobs), reused_evaluation=None)
            if (payload['counters'] != state['counters'] or payload['experiment_id'] != config['experiment_id']
                    or payload['model_config'] != checkpoint['config']):
                raise ValueError('Raw evaluation belongs to another policy node')
            report['summary'] = summarize_evaluation(payload, tasks, modes, manifest.get('split'))
            with gzip.open(run/'evaluations/idle_control.json.gz', 'rt') as stream:
                idle = json.load(stream)['seed_results']
            report['paired_idle'] = {mode:paired_idle_summary(tasks,payload['seed_results'][mode],idle)
                                     for mode in modes}
            atomic_json(output/'seed_results.json.gz', payload, compressed=True)
            report.update(status='complete', raw_seed_results_path='seed_results.json.gz',
                scope='Scores only the supplied immutable point on all136 frozen cases/mode. Retains failures and truncations; no training mutation, winner selection, identical-spent-time or original full acceptance claim. Report extra assessment cost separately from source training.')
        except BaseException as error:
            report.update(status='failed_preserved', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            monitor.stop.set(); monitor.thread.join()
            report.update(seconds=time.monotonic()-started, resources=monitor.snapshot(),
                          cuda_initialized=torch.cuda.is_initialized())
            atomic_json(output/'report.json', report); atomic_json(output/'progress.json', report)
        print(json.dumps({k:report[k] for k in ('status','seconds','extra_model_cases','cuda_initialized')}), flush=True)


if __name__ == '__main__':
    main()
