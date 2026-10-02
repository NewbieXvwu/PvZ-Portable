"""Persist numeric synthetic CPU contracts; no native/CUDA/learning pass is granted."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]

import numpy as np
import torch
from pvz_agent_model import (GameplayModelV1, configure_torch_threads, model_architecture_version,
                             replay_log_probs, select_action, unpack_tokens)
from pvz_common import git_metadata, sha256_file
from pvz_research import capture_rng
from pvz_seed_jobs import atomic_json, atomic_numpy, atomic_write, read_numpy
from test_event_wait_policy import fixture, update
from test_observation_context import SMALL
from t4_capability_profile import _state_sha256
from train_pvz_ppo import episode_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    output, report = args.evidence_dir.resolve(), args.report.resolve()
    if output.exists() or report.exists():
        raise ValueError('retain existing evidence; select a fresh output')
    if not any(part in ('research', 'research_evidence') for part in output.parts):
        raise ValueError('raw synthetic checkpoints/shards must use an ignored research directory')
    configure_torch_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    revision, dirty = git_metadata(ROOT)
    sources = ['python/pvz_agent_model.py', 'python/pvz_wait_events.py', 'python/pvz_event_env.py',
               'python/train_pvz_ppo.py', 'python/pvz_research.py', 'python/train_pvz_ppo_task_family.py',
               'scripts/t4_capability_profile.py', 'python/test_event_wait_policy.py',
               'python/test_observation_context.py', 'scripts/research_event_wait_policy_contracts.py']
    output.mkdir(parents=True)
    records = []
    for seed in (0, 1, 2):
        paired_weights = None
        for mode in ('fixed', 'events'):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = GameplayModelV1({**SMALL, 'wait_mode': mode}).eval()
            initial_sha = _state_sha256(model.state_dict())
            paired = paired_weights is None or initial_sha == paired_weights
            if not paired:
                raise AssertionError('fixed/events initialization parameters differ')
            paired_weights = initial_sha
            episodes = fixture(model)
            raw = output/f'{mode}_seed{seed}.npz'
            atomic_numpy(raw, episodes, compressed=True)
            restored = read_numpy(raw)
            if [episode_digest(e) for e in restored] != [episode_digest(e) for e in episodes]:
                raise AssertionError('raw shard changed a synthetic trajectory')
            errors = dict(log_prob=0., entropy=0., privileged_value=0., condition_logits=0., final_hidden=0.)
            with torch.no_grad():
                outputs, batched_hidden = model.forward_sequences([e['transitions'] for e in episodes], [None, None])
                transitions = [tr for e in episodes for tr in e['transitions']]
                lp, entropy = replay_log_probs(model, outputs, transitions)
                offset = 0
                for index, episode in enumerate(episodes):
                    hidden = None
                    for step in episode['transitions']:
                        tensors, metadata = unpack_tokens(step['tokens'], torch.device('cpu'))
                        single = model.step_tokens(tensors, metadata, step['wave'], hidden, step['previous_action'],
                            step['elapsed_since_previous_observation'], step['events'], step['previous_wait_result'])
                        _, step_lp, step_ent = select_action(model, single, step['legal'], action=step['action'])
                        errors['log_prob'] = max(errors['log_prob'], abs(float(lp[offset]-step_lp)))
                        errors['entropy'] = max(errors['entropy'], abs(float(entropy[offset]-step_ent)))
                        errors['privileged_value'] = max(errors['privileged_value'], float((
                            model.privileged_value_from_extra(single, step['critic_extra'])
                            - model.privileged_value_from_extra(outputs[offset], step['critic_extra'])).abs().max()))
                        errors['condition_logits'] = max(errors['condition_logits'], float((
                            single['wait_condition_logits']-outputs[offset]['wait_condition_logits']).abs().max()))
                        hidden, offset = single['hidden'], offset+1
                    errors['final_hidden'] = max(errors['final_hidden'], float((
                        hidden[:, 0]-batched_hidden[:, index]).abs().max()))
            if max(errors.values()) > 5e-5:
                raise AssertionError(f'synthetic replay exceeds existing 5e-5 tolerance: {errors}')
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            initial_condition = model.wait_condition.weight.detach().clone()
            losses = update(model, episodes, optimizer)
            checkpoint = output/f'{mode}_seed{seed}_trained.pt'
            atomic_write(checkpoint, lambda path: torch.save(dict(
                scope='synthetic CPU contract, not a native/CUDA/learning checkpoint',
                config=model.config, model_architecture_version=model_architecture_version(model.config),
                state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                rng_state=capture_rng(random.Random(seed)), losses=losses), path))
            records.append(dict(initialization_seed=seed, mode=mode, config=model.config,
                paired_initial_parameters_identical=paired, initial_state_sha256=initial_sha,
                parameters=sum(p.numel() for p in model.parameters()), episodes=len(episodes),
                decisions=len(transitions), synthetic_actual_ticks=sum(t['action_duration_ticks'] for t in transitions),
                single_batched_max_abs_errors=errors, cpu_synthetic_update_losses=losses,
                condition_head_changed=not torch.equal(initial_condition, model.wait_condition.weight),
                raw_shard=dict(path=str(raw), sha256=sha256_file(raw), bytes=raw.stat().st_size),
                checkpoint=dict(path=str(checkpoint), sha256=sha256_file(checkpoint), bytes=checkpoint.stat().st_size)))
            atomic_json(output/'progress.json', dict(records=records, seconds=time.monotonic()-started))
            print(f'synthetic mode={mode} seed={seed} decisions={len(transitions)} errors={errors}', flush=True)
    atomic_json(report, dict(schema_version=1, completed_at=datetime.now(timezone.utc).isoformat(),
        scope='synthetic CPU policy, recurrence and raw storage contracts; no simulator was started',
        source_commit=revision, source_dirty=dirty,
        source_fingerprints={name: sha256_file(ROOT/name) for name in sources},
        engineering_checks_passed=True, native_audit_passed=False, cuda_update_passed=False,
        actual_sigkill_resume_passed=False, learning_benefit_verified=False, mainline_merged=False,
        torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
        records=records, peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
