"""Released-only full-size event policy collection and CPU/CUDA replay gates.

No learning claim: every optimizer has lr=0. Complete chronological recurrence
is checked without gradients; real PPO uses 128-step windows and recomputed
prefixes. This does not claim full-episode BPTT for long or truncated episodes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import gzip
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]

import numpy as np
import torch
from pvz_agent_model import (GameplayModelV1, configure_torch_threads,
    model_architecture_version, observation_tokens, pack_tokens, replay_log_probs, unpack_tokens)
from pvz_common import sha256_file
from pvz_env import PvZEnv
from pvz_event_env import EventWaitEnv
from pvz_research import ResourceMonitor
from pvz_seed_jobs import atomic_json, atomic_numpy, atomic_write, read_numpy
from pvz_wait_events import CONDITIONS, summarize_wait_records, validate_wait_result
from research_event_wait_native_audit import REQUIRED_COVERAGE, stop_owned_job
from t4_capability_profile import _state_sha256
from train_pvz_ppo import add_advantages, collect_task_episode, episode_digest, train_update

MODEL = dict(layers=6, width=256, heads=8, ff_width=1024, gru_layers=2,
             gru_width=256, critic_width=256, critic_layers=1, input_flags=7)
TASK_IDS = ['train_day_1', 'train_night_1', 'train_pool_1', 'train_fog_1', 'train_roof_1']
REF = 'refs/remotes/delivery/research/event-wait-replay-gates-v1'
RAW_ROOT = Path('/home/newbiexvwu/PvZ-Portable/artifacts/research')
SOURCES = (
    'scripts/research_event_wait_replay_audit.py', 'python/test_event_wait_replay_audit.py',
    'scripts/research_event_wait_native_audit.py', 'python/test_event_wait_native_audit.py',
    'scripts/t4_capability_profile.py', 'scripts/scripted_baseline.py', 'scripts/check_task_manifests.py',
    'python/pvz_agent_model.py', 'python/pvz_observation_features.py', 'python/pvz_env.py',
    'python/pvz_event_env.py', 'python/pvz_wait_events.py', 'python/pvz_common.py',
    'python/pvz_seed_jobs.py', 'python/pvz_value.py', 'python/pvz_research.py', 'python/pvz_curriculum.py',
    'python/train_pvz_ppo.py', 'python/train_pvz_ppo_task_family.py',
    'src/EnvironmentWaitEvents.h', 'src/LawnApp.h', 'src/LawnApp.cpp', 'src/main.cpp',
)
LIMITS = dict(log_prob=3e-6, privileged_value=5e-5, entropy=5e-5,
              condition_logits=5e-5, final_hidden=5e-5)
PPO = dict(lr=0., gamma=.99, shaping_weight=1., gae_lambda=.95, epochs=2,
           sequence_length=128, minibatch_chunks=2, clip_epsilon=.2,
           value_coefficient=.5, entropy_coefficient=.01, target_kl=.03,
           attention_backend='dense')


def published_bytes(root, ref, path):
    relative = str(path.resolve().relative_to(root.resolve()))
    published = subprocess.check_output(['git', '-C', str(root), 'show', ref+':'+relative])
    if published != path.read_bytes():
        raise ValueError(f'local bytes differ from fetched publication: {relative}')


def validate_native_report(report, protocol, directory):
    """Require all native jobs/branches/raw records, never just a pass flag."""
    if report.get('status') != 'complete' or report.get('gate_result') != 'pass':
        raise ValueError('complete native event PASS is required')
    tasks = json.loads(Path(protocol['task_manifest']['path']).read_text())['tasks']
    expected = [(task['task_id'], seed, regime) for task in tasks
                for seed in task['seeds'][:2] for regime in ('wait_only', 'one_first_legal_plant')]
    if len(expected) != 120 or [t['task_id'] for t in tasks] != protocol['all_task_ids_in_order']:
        raise ValueError('native task inventory differs')
    branches = list(itertools.product((0, 3000, 9000), (60, 150, 300), CONDITIONS))
    if report.get('expected_jobs') != 120 or len(report.get('jobs', [])) != 120:
        raise ValueError('native job inventory is incomplete')
    counts, records = Counter(), []
    for index, (job, identity) in enumerate(zip(report['jobs'], expected)):
        if (job.get('job_id'), job.get('task_id'), job.get('seed'), job.get('regime')) != (index, *identity):
            raise ValueError('native job identity differs')
        if job.get('record_replay_passed') is not True or len(job.get('rows', [])) != 54:
            raise ValueError('native branches or record replay are incomplete')
        for row, branch in zip(job['rows'], branches):
            action, result = row['action'], row['wait_result']
            if (row['prefix_requested'], action['ticks'], action['until']) != branch:
                raise ValueError('native branch inventory differs')
            validate_wait_result(action, result, result['actual_ticks'])
            counts.update(result['triggered'])
            counts['condition:'+action['until']] += 'condition' in result['triggered']
            counts['initial_true'] += result['initial_condition_satisfied']
            counts['zero_actual_ticks'] += result['actual_ticks'] == 0
            counts['no_event_equivalent'] += result['triggered'] == ['max_ticks']
        folder = directory/f'job_{index:04d}'
        if json.loads((folder/'result.json').read_text()) != job:
            raise ValueError('native job result differs from report')
        if sha256_file(folder/'raw.jsonl.gz') != job['raw_sha256']:
            raise ValueError('native raw trace differs')
        with gzip.open(folder/'record.json.gz', 'rt') as stream:
            record = json.load(stream)
        if not record.get('operations'):
            raise ValueError('native replay record is empty')
        records.append(dict(job_id=index, raw_sha256=job['raw_sha256'],
                            record_sha256=sha256_file(folder/'record.json.gz')))
    # Counter treats absent zero-valued fields and explicit zeros identically.
    if counts != Counter(report.get('counts', {})) or any(counts[name] <= 0 for name in REQUIRED_COVERAGE):
        raise ValueError('native positive coverage differs or is incomplete')
    if report.get('missing_coverage') != []:
        raise ValueError('native report records missing coverage')
    return records


def check_protocol(path):
    protocol = json.loads(path.read_text())
    if protocol.get('release_status') != 'released':
        raise RuntimeError('event replay audit is draft; publish the released protocol first')
    published_bytes(ROOT, REF, path)
    fingerprints = protocol['required_fingerprints']
    if set(fingerprints) != set(SOURCES):
        raise ValueError('required source inventory differs')
    for name, expected in fingerprints.items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError(f'frozen source differs: {name}')
        published_bytes(ROOT, REF, ROOT/name)
    if (protocol['model_config'] != MODEL or protocol['limits'] != LIMITS or protocol['ppo'] != PPO
            or protocol['model_architecture_version'] != 8 or protocol['selected_task_ids'] != TASK_IDS
            or protocol['sampling_seed_rule'] != 'torch.manual_seed(environment_seed+170000)'
            or protocol['initialization_seeds'] != [0, 1, 2] or protocol['modes'] != ['fixed', 'events']
            or protocol['seeds_per_task'] != 4 or protocol['max_actions'] != 4000
            or protocol['allow_truncation'] is not True or protocol['expected_cases'] != 6
            or protocol['expected_episodes'] != 120 or protocol['parameter_count'] != 7869607):
        raise ValueError('frozen replay configuration differs')
    if (protocol['minimum_available_ram_bytes'] != 12*1024**3
            or protocol['minimum_free_vram_mib'] != 14000
            or protocol['case_timeout_seconds'] != 1200 or protocol['stage_timeout_seconds'] != 7200):
        raise ValueError('frozen resource/time gates differ')
    manifest = protocol['task_manifest']
    task_path = ROOT/manifest['path']
    if manifest['path'] != 'experiments/task_family/train.json' or sha256_file(task_path) != manifest['sha256']:
        raise ValueError('unchanged original task manifest differs')
    published_bytes(ROOT, REF, task_path)
    tasks = [t for t in json.loads(task_path.read_text())['tasks']
             if t['wave_cap'] == 1 and t['zombie_count_multiplier'] == 1]
    if [t['task_id'] for t in tasks] != TASK_IDS or len({t['terrain'] for t in tasks}) != 5:
        raise ValueError('require the five unchanged ordinary cap1 tasks')
    if any(len(t['seeds']) < 4 for t in tasks):
        raise ValueError('original environment seed inventory is incomplete')
    if sha256_file(Path(protocol['executable']['path'])) != protocol['executable']['sha256']:
        raise ValueError('frozen candidate binary differs')
    for name, expected in protocol['resource_fingerprints'].items():
        if sha256_file(Path(protocol['resource_dir'])/name) != expected:
            raise ValueError('frozen resource differs')
    if set(protocol['resource_fingerprints']) != {'main.pak', 'properties/partner.xml'}:
        raise ValueError('resource fingerprint inventory differs')
    native = protocol['native_audit']
    native_path, report_path = Path(native['protocol_path']), Path(native['report_path'])
    if sha256_file(native_path) != native['protocol_sha256'] or sha256_file(report_path) != native['report_sha256']:
        raise ValueError('frozen native prerequisite bytes differ')
    native_protocol = json.loads(native_path.read_text())
    if native_protocol.get('release_status') != 'released':
        raise ValueError('native prerequisite protocol is not released')
    native_root = Path(native['source_root'])
    native_ref = 'refs/remotes/delivery/research/event-wait-native-audit-v1'
    published_bytes(native_root, native_ref, native_path)
    for name, expected in native_protocol['required_fingerprints'].items():
        if fingerprints.get(name) != expected or sha256_file(native_root/name) != expected:
            raise ValueError('native gate and replay candidate source differ')
        published_bytes(native_root, native_ref, native_root/name)
    if (native_protocol['candidate_executable'] != protocol['executable']
            or native_protocol['resource_dir'] != protocol['resource_dir']
            or native_protocol['resource_fingerprints'] != protocol['resource_fingerprints']):
        raise ValueError('native prerequisite uses different binary/resources')
    native_task = native_protocol['task_manifest']
    if sha256_file(Path(native_task['path'])) != native_task['sha256']:
        raise ValueError('native prerequisite tasks changed')
    report = json.loads(report_path.read_text())
    if report.get('protocol_sha256') != native['protocol_sha256']:
        raise ValueError('native report does not bind the frozen protocol')
    protocol['_native_records'] = validate_native_report(report, native_protocol, report_path.parent)
    return protocol, tasks


def configure_runtime(stage):
    # Must precede cuda.is_available(), device construction and RNG seeding.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    configure_torch_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    if stage == 'replay-cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA replay requires CUDA; CPU fallback is prohibited')


def numeric_rows(model, outputs, transitions):
    lp, entropy = replay_log_probs(model, outputs, transitions)
    values = [model.privileged_value_from_extra(out, tr['critic_extra']).reshape(())
              for out, tr in zip(outputs, transitions)]
    logits = torch.stack([out['wait_condition_logits'].reshape(6) for out in outputs])
    rows = torch.cat((lp[:, None], entropy[:, None], torch.stack(values)[:, None], logits), dim=1)
    result = rows.detach().cpu().numpy().copy()
    if not np.isfinite(result).all():
        raise FloatingPointError('non-finite replay log probability/value/entropy/logits')
    return result


def check_rows(actual, expected):
    if actual.shape != expected.shape or actual.ndim != 2 or actual.shape[1] != 9:
        raise ValueError('replay scalar inventory differs')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise FloatingPointError('non-finite replay reference')
    difference = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    errors = dict(log_prob=float(difference[:, 0].max()), entropy=float(difference[:, 1].max()),
                  privileged_value=float(difference[:, 2].max()), condition_logits=float(difference[:, 3:].max()))
    if any(value >= LIMITS[name] for name, value in errors.items()):
        raise AssertionError(f'unchanged replay tolerance exceeded: {errors}')
    return errors


def probe_episode(model, episode, reference=None, window=128):
    """Compare every single step with a chronological chunk replay and sampled scalars."""
    transitions = episode['transitions']
    if not transitions:
        raise ValueError('empty episode cannot pass replay')
    if any(not {'previous_wait_result', 'wait_result'} <= set(tr) for tr in transitions):
        raise ValueError('explicit wait metadata is incomplete')
    device = next(model.parameters()).device
    hidden, rows = None, []
    with torch.no_grad():
        for transition in transitions:
            tensors, metadata = unpack_tokens(transition['tokens'], device)
            output = model.step_tokens(tensors, metadata, transition['wave'], hidden,
                transition['previous_action'], transition['elapsed_since_previous_observation'],
                transition['events'], transition['previous_wait_result'])
            rows.append(numeric_rows(model, [output], [transition]))
            hidden = output['hidden']
        scalars = np.concatenate(rows)
        sampled = scalars.copy()
        sampled[:, 0] = [tr['log_prob'] for tr in transitions]
        sampled[:, 2] = [tr['value'] for tr in transitions]
        sampled_errors = check_rows(scalars, sampled)
        hidden_single = hidden.detach().cpu().numpy().copy()
        replay_hidden, chunk_rows = None, []
        for start in range(0, len(transitions), window):
            chunk = transitions[start:start+window]
            outputs, replay_hidden = model.forward_sequences([chunk], [replay_hidden])
            chunk_rows.append(numeric_rows(model, outputs, chunk))
            replay_hidden = replay_hidden[:, 0, :]
        chunk_errors = check_rows(np.concatenate(chunk_rows), scalars)
        hidden_chunk = replay_hidden.detach().cpu().numpy()[:, None, :]
        hidden_error = float(np.abs(hidden_single-hidden_chunk).max())
        if not math.isfinite(hidden_error) or hidden_error >= LIMITS['final_hidden']:
            raise AssertionError(f'final recurrent state exceeds tolerance: {hidden_error}')
        cross_device = None
        if reference is not None:
            cross_device = check_rows(scalars, reference['scalars'])
            other_hidden = reference['final_hidden']
            if other_hidden.shape != hidden_single.shape or not np.isfinite(other_hidden).all():
                raise ValueError('stored final recurrent state differs')
            final_error = float(np.abs(hidden_single-other_hidden).max())
            if final_error >= LIMITS['final_hidden']:
                raise AssertionError('CPU/CUDA final recurrent state exceeds tolerance')
            cross_device['final_hidden'] = final_error
        bootstrap_error = None
        bootstrap = None
        if 'replay_bootstrap' in episode:
            context = episode['replay_bootstrap']
            bootstrap = 0.
            if context is not None:
                if episode.get('truncated') is not True:
                    raise ValueError('terminal episode cannot carry truncation bootstrap inputs')
                tensors, metadata = unpack_tokens(context['tokens'], device)
                output = model.step_tokens(tensors, metadata, context['wave'], hidden,
                    context['previous_action'], context['elapsed_since_previous_observation'],
                    context['events'], context['previous_wait_result'])
                bootstrap = float(model.privileged_value_from_extra(output, context['critic_extra']).item())
            elif episode.get('truncated') is not False:
                raise ValueError('truncated episode omitted its bootstrap inputs')
            bootstrap_error = abs(bootstrap-episode['bootstrap_value'])
            if not math.isfinite(bootstrap_error) or bootstrap_error >= LIMITS['privileged_value']:
                raise AssertionError('collected bootstrap value exceeds unchanged value tolerance')
            if reference is not None:
                reference_bootstrap = reference['bootstrap_value']
                if reference_bootstrap is None or not math.isfinite(reference_bootstrap):
                    raise ValueError('CPU reference omitted a finite bootstrap value')
                cpu_error = abs(bootstrap-reference_bootstrap)
                if cpu_error >= LIMITS['privileged_value']:
                    raise AssertionError('CPU/CUDA bootstrap value exceeds unchanged value tolerance')
                cross_device['bootstrap_value'] = cpu_error
    return dict(scalars=scalars, final_hidden=hidden_single, bootstrap_value=bootstrap), dict(
        sampled=sampled_errors, chunk=chunk_errors, final_hidden=hidden_error,
        bootstrap_value=bootstrap_error, cpu_cuda=cross_device)


def zero_lr_update(model, episodes, optimizer, label):
    before = _state_sha256(model.state_dict())
    add_advantages(episodes, PPO['gae_lambda'], PPO['gamma'])
    losses = train_update(model, episodes, optimizer, next(model.parameters()).device,
        PPO['epochs'], PPO['sequence_length'], PPO['clip_epsilon'], PPO['value_coefficient'],
        PPO['entropy_coefficient'], minibatch_chunks=PPO['minibatch_chunks'],
        attention_backend=PPO['attention_backend'], label=label, target_kl=PPO['target_kl'])
    if (losses['max_log_prob_change'] >= LIMITS['log_prob'] or losses['clip_fraction'] != 0
            or losses['stopped_for_kl'] or losses['optimizer_steps'] <= 0
            or any(not math.isfinite(value) for value in losses.values())):
        raise AssertionError(f'lr=0 PPO replay gate failed: {losses}')
    if _state_sha256(model.state_dict()) != before:
        raise AssertionError('lr=0 changed model parameters')
    if not optimizer.state:
        raise AssertionError('no actual AdamW state was created')
    return losses


class RecordingMixin:
    """Flush requests before native commands and observations after every macro."""
    def __init__(self, *args, trace, **kwargs):
        self.trace = trace
        super().__init__(*args, **kwargs)

    def step(self, action):
        self.trace('action_request', dict(action))
        response = super().step(action)
        self.last_response = response
        self.trace('action_response', response)
        return response

    def _record_operation(self, operation, observation, events=None):
        super()._record_operation(operation, observation, events)
        self.trace('operation', dict(operation=operation, observation=observation, events=events))


class RecordingFixedEnv(RecordingMixin, PvZEnv):
    pass


class RecordingEventEnv(RecordingMixin, EventWaitEnv):
    pass


def case_jobs(tasks):
    return [dict(job_id=index, task_id=task['task_id'], environment_seed=seed, task=task)
            for index, (task, seed) in enumerate((t, s) for t in tasks for s in t['seeds'][:4])]


def validate_case_manifest(manifest, jobs, seed, mode, protocol_sha):
    if (manifest.get('initialization_seed') != seed or manifest.get('mode') != mode
            or manifest.get('model_config') != {**MODEL, 'wait_mode': mode}
            or manifest.get('protocol_sha256') != protocol_sha or manifest.get('torch_version') != torch.__version__
            or len(manifest.get('episodes', [])) != 20):
        raise ValueError('CPU collection manifest differs or is incomplete')
    for item, job in zip(manifest['episodes'], jobs):
        if any(item.get(key) != job[key] for key in ('job_id', 'task_id', 'environment_seed')):
            raise ValueError('CPU episode inventory differs')
        index = job['job_id']
        if item.get('path') != f'job_{index:04d}/episode.npz' or item.get('reference_path') != f'job_{index:04d}/cpu_reference.npz':
            raise ValueError('CPU evidence path inventory differs')


def verified_file(directory, relative, expected):
    path = (directory/relative).resolve()
    if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected:
        raise ValueError('stored evidence path or SHA differs')
    return path


def validate_cpu_report(report, output, jobs, protocol_sha):
    if (report.get('status') != 'complete' or report.get('gate_result') != 'pass'
            or report.get('stage') != 'collect-cpu' or report.get('protocol_sha256') != protocol_sha
            or report.get('episodes') != 120 or len(report.get('cases', [])) != 6):
        raise ValueError('complete CPU gate on the same published protocol is required')
    for case, (seed, mode) in zip(report['cases'], itertools.product((0, 1, 2), ('fixed', 'events'))):
        if (case.get('initialization_seed'), case.get('mode')) != (seed, mode):
            raise ValueError('CPU case inventory differs')
        base = output/f'{mode}_seed{seed}'
        if json.loads((base/'collect-cpu/result.json').read_text()) != case:
            raise ValueError('CPU case differs from aggregate report')
        manifest = json.loads((base/'collection_manifest.json').read_text())
        validate_case_manifest(manifest, jobs, seed, mode, protocol_sha)
        if (case.get('status') != 'complete' or case.get('gate_result') != 'pass'
                or case.get('model_state_unchanged') is not True or case.get('episodes') != manifest['episodes']
                or case.get('initial_state_sha256') != manifest['initial_state_sha256']
                or case.get('torch_version') != torch.__version__ or len(case.get('optimizer_batches', [])) != 10):
            raise ValueError('CPU collection and real optimizer gate differ')


def run_case(stage, seed, mode, protocol, tasks, output_string):
    output = Path(output_string)
    base, directory = output/f'{mode}_seed{seed}', output/f'{mode}_seed{seed}'/stage
    os.setsid()
    with (directory/'worker.log').open('x') as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        atomic_json(directory/'worker.json', dict(pid=os.getpid(), pgid=os.getpgrp()))
        started, rows, losses = time.monotonic(), [], []
        try:
            configure_runtime(stage)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = GameplayModelV1({**MODEL, 'wait_mode': mode}).eval()
            if sum(p.numel() for p in model.parameters()) != protocol['parameter_count']:
                raise ValueError('full reference model parameter count differs')
            initial_sha = _state_sha256(model.state_dict())
            jobs = case_jobs(tasks)
            checkpoint = base/'initial_model.pt'
            manifest_path = base/'collection_manifest.json'
            if stage == 'collect-cpu':
                atomic_write(checkpoint, lambda path: torch.save(dict(config=model.config,
                    model_architecture_version=model_architecture_version(model.config),
                    state_dict=model.state_dict(), scope='initial engineering replay model; no learning'), path))
            else:
                manifest = json.loads(manifest_path.read_text())
                validate_case_manifest(manifest, jobs, seed, mode, protocol['_protocol_sha256'])
                saved = torch.load(verified_file(base, 'initial_model.pt', manifest['checkpoint_sha256']),
                                   map_location='cpu', weights_only=False)
                if saved['config'] != model.config or saved['model_architecture_version'] != 8:
                    raise ValueError('checkpoint uses a different explicit configuration')
                model.load_state_dict(saved['state_dict'])
                if _state_sha256(model.state_dict()) != initial_sha or initial_sha != manifest['initial_state_sha256']:
                    raise ValueError('paired initialization/checkpoint differs')
                free, _ = torch.cuda.mem_get_info()
                if free < protocol['minimum_free_vram_mib']*1024**2:
                    raise RuntimeError('full-model CUDA memory gate stopped before model allocation')
                torch.cuda.reset_peak_memory_stats()
                model.to('cuda')
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
            pending = []
            for job in jobs:
                index = job['job_id']
                folder = base/f'job_{index:04d}'
                if stage == 'collect-cpu':
                    folder.mkdir()
                    with gzip.open(folder/'raw.jsonl.gz', 'xt', encoding='utf-8', compresslevel=1) as raw:
                        def trace(kind, payload):
                            raw.write(json.dumps(dict(kind=kind, payload=payload), separators=(',', ':'))+'\n')
                            raw.flush()
                        cls = RecordingEventEnv if mode == 'events' else RecordingFixedEnv
                        with cls(protocol['resource_dir'], executable=protocol['executable']['path'],
                                 save_dir=folder/'saves', trace=trace) as env:
                            torch.manual_seed(job['environment_seed']+170000)
                            episode = collect_task_episode(model, env, job['task'], job['environment_seed'],
                                index, 4000, dict(gamma=.99, shaping_weight=1.), allow_truncation=True)
                            episode['replay_bootstrap'] = None
                            if episode['truncated']:
                                observation, _, _, _, info = env.last_response
                                tensors, metadata = observation_tokens(observation, model.config['input_flags'])
                                last = episode['transitions'][-1]
                                roster = env.critic_inputs(observation['wave'])['wave_zombies']
                                episode['replay_bootstrap'] = dict(tokens=pack_tokens(tensors, metadata),
                                    wave=observation['wave'], previous_action=last['action'],
                                    elapsed_since_previous_observation=last['action_duration_ticks'],
                                    events=info['events'], previous_wait_result=last['wait_result'],
                                    critic_extra=model.privileged_extra_from_inputs(observation['wave_timer'], roster))
                                trace('bootstrap_inputs', dict(wave=observation['wave'], actual_tick=observation['tick'],
                                    critic_extra=episode['replay_bootstrap']['critic_extra']))
                            atomic_json(folder/'record.json.gz', copy.deepcopy(env.episode), compressed=True)
                    atomic_numpy(folder/'episode.npz', episode, compressed=True)
                    if episode_digest(read_numpy(folder/'episode.npz')) != episode_digest(episode):
                        raise AssertionError('raw shard changed the collected trajectory')
                    reference, errors = probe_episode(model, episode)
                    atomic_numpy(folder/'cpu_reference.npz', reference, compressed=True)
                    waits = [dict(action=tr['action'], wait_result=tr['wait_result'],
                                  actual_ticks=tr['action_duration_ticks'])
                             for tr in episode['transitions'] if tr['action']['type'] == 'wait']
                    item = {key: job[key] for key in ('job_id', 'task_id', 'environment_seed')}
                    item.update(path=f'job_{index:04d}/episode.npz', sha256=sha256_file(folder/'episode.npz'),
                        reference_path=f'job_{index:04d}/cpu_reference.npz',
                        reference_sha256=sha256_file(folder/'cpu_reference.npz'),
                        raw_sha256=sha256_file(folder/'raw.jsonl.gz'), record_sha256=sha256_file(folder/'record.json.gz'),
                        digest=episode_digest(episode), decisions=len(episode['transitions']),
                        actual_ticks=sum(tr['action_duration_ticks'] for tr in episode['transitions']),
                        won=episode['won'], result=episode['result'], truncated=episode['truncated'],
                        bootstrap_value=episode['bootstrap_value'], wait_summary=summarize_wait_records(waits), errors=errors)
                else:
                    item = manifest['episodes'][index]
                    episode = read_numpy(verified_file(base, item['path'], item['sha256']))
                    reference = read_numpy(verified_file(base, item['reference_path'], item['reference_sha256']))
                    if episode_digest(episode) != item['digest']:
                        raise ValueError('stored episode digest differs')
                    for name in ('raw.jsonl.gz', 'record.json.gz'):
                        verified_file(base, f'job_{index:04d}/{name}', item['raw_sha256' if name.startswith('raw') else 'record_sha256'])
                    _, errors = probe_episode(model, episode, reference)
                    item = dict(item, errors=errors)
                if not math.isfinite(episode['bootstrap_value']):
                    raise FloatingPointError('non-finite truncation bootstrap value')
                rows.append(item)
                pending.append(episode)
                if len(pending) == 2:
                    losses.append(zero_lr_update(model, pending, optimizer, f'{stage} {mode} seed{seed} pair{len(losses)}'))
                    pending.clear()
                atomic_json(directory/'progress.json', dict(episodes=rows, optimizer_batches=losses,
                    completed_episodes=len(rows), expected_episodes=20, seconds=time.monotonic()-started))
                print(f'{stage} {mode} seed={seed} episode={len(rows)}/20 decisions={item["decisions"]}', flush=True)
            if pending or _state_sha256(model.state_dict()) != initial_sha:
                raise AssertionError('incomplete paired batch or model state changed')
            if stage == 'collect-cpu':
                atomic_json(manifest_path, dict(initialization_seed=seed, mode=mode, model_config=model.config,
                    protocol_sha256=protocol['_protocol_sha256'], torch_version=torch.__version__,
                    initial_state_sha256=initial_sha, checkpoint_sha256=sha256_file(checkpoint), episodes=rows))
            atomic_json(directory/'result.json', dict(status='complete', gate_result='pass', stage=stage,
                initialization_seed=seed, mode=mode, episodes=rows, optimizer_batches=losses,
                initial_state_sha256=initial_sha, model_state_unchanged=True,
                protocol_sha256=protocol['_protocol_sha256'], torch_version=torch.__version__,
                seconds=time.monotonic()-started,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if stage == 'replay-cuda' else None,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if stage == 'replay-cuda' else None))
        except BaseException:
            atomic_json(directory/'failure.json', dict(error=traceback.format_exc(), episodes=rows,
                optimizer_batches=losses, seconds=time.monotonic()-started))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=('collect-cpu', 'replay-cuda'))
    args = parser.parse_args()
    protocol, tasks = check_protocol(args.protocol.resolve())  # Before directories or hardware/model allocation.
    protocol['_protocol_sha256'] = sha256_file(args.protocol)
    output = args.output_dir.resolve()
    if not output.is_relative_to(RAW_ROOT):
        raise ValueError('raw replay evidence must remain in the ignored research tree')
    configure_runtime(args.stage)
    if args.stage == 'collect-cpu':
        output.mkdir(parents=True, exist_ok=False)
    else:
        cpu = json.loads((output/'collect-cpu_report.json').read_text())
        validate_cpu_report(cpu, output, case_jobs(tasks), protocol['_protocol_sha256'])
    report_path = output/f'{args.stage}_report.json'
    if report_path.exists():
        raise ValueError('retain existing stage report; choose a fresh evidence directory')
    lock = (output/'.execution.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    monitor = ResourceMonitor()
    monitor.thread.start()
    started, cases, process = time.monotonic(), [], None
    report = dict(schema_version=1, status='running', gate_result='pending', stage=args.stage,
                  protocol_sha256=protocol['_protocol_sha256'], native_records=protocol['_native_records'],
                  scope='actual native engineering replay at full size, lr=0; no learning acceptance')
    try:
        for seed in (0, 1, 2):
            paired_sha = None
            for mode in ('fixed', 'events'):
                directory = output/f'{mode}_seed{seed}'/args.stage
                directory.mkdir(parents=True, exist_ok=False)
                available = int(Path('/proc/meminfo').read_text().split('MemAvailable:')[1].split()[0])*1024
                if available < protocol['minimum_available_ram_bytes']:
                    raise RuntimeError('full-model replay RAM gate stopped before starting case')
                process = multiprocessing.get_context('spawn').Process(target=run_case,
                    args=(args.stage, seed, mode, protocol, tasks, str(output)))
                process.start()
                case_started = last_progress = time.monotonic()
                while process.is_alive():
                    process.join(timeout=1)
                    if (directory/'worker.json').exists():
                        owner = json.loads((directory/'worker.json').read_text())
                        if owner['pid'] != process.pid or owner['pgid'] != process.pid:
                            raise RuntimeError('replay worker session identity differs')
                        process._owned_session = True
                    now = time.monotonic()
                    if now-started >= protocol['stage_timeout_seconds'] or now-case_started >= protocol['case_timeout_seconds']:
                        raise RuntimeError('replay exceeded frozen wall deadline; all evidence retained')
                    if now-last_progress >= 30:
                        print(f'{args.stage} {mode} seed={seed} elapsed={now-started:.1f}s', flush=True)
                        last_progress = now
                if (directory/'worker.json').exists():
                    owner = json.loads((directory/'worker.json').read_text())
                    process._owned_session = owner['pid'] == process.pid and owner['pgid'] == process.pid
                if process.exitcode != 0:
                    raise RuntimeError(f'replay worker failed: {mode} seed={seed} exit={process.exitcode}')
                case = json.loads((directory/'result.json').read_text())
                if case['gate_result'] != 'pass' or len(case['episodes']) != 20:
                    raise ValueError('incomplete replay case')
                if paired_sha is not None and paired_sha != case['initial_state_sha256']:
                    raise ValueError('fixed/events paired initialization differs')
                paired_sha = case['initial_state_sha256']
                cases.append(case)
                atomic_json(output/f'{args.stage}_progress.json', dict(completed_cases=len(cases), expected_cases=6,
                    decisions=sum(e['decisions'] for c in cases for e in c['episodes']), seconds=time.monotonic()-started))
        report.update(status='complete', gate_result='pass', cases=cases,
                      episodes=120, decisions=sum(e['decisions'] for c in cases for e in c['episodes']))
    except BaseException:
        if process is not None:
            stop_owned_job(process)
        report.update(status='failed', gate_result='fail', cases=cases, error=traceback.format_exc())
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join()
        report.update(seconds=time.monotonic()-started, peak_process_tree_rss_bytes=monitor.peak_tree_rss_bytes,
                      min_system_available_bytes=monitor.min_available_bytes)
        atomic_json(report_path, report)
        lock.close()


if __name__ == '__main__':
    main()
