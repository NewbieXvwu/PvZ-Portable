"""Synthetic replay/launch/evidence contracts; no simulator or GPU is started."""
import copy
from collections import Counter
import gzip
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import research_event_wait_replay_audit as audit
from pvz_seed_jobs import atomic_json, atomic_numpy, read_numpy
from pvz_wait_events import CONDITIONS
from test_event_wait_policy import fixture, wait_result, SyntheticEnv, SyntheticEventEnv
from test_observation_context import SMALL


def native_fixture(directory):
    """A synthetic manifest, never native physics or event-coverage evidence."""
    tasks = [dict(task_id=f'synthetic_{i}', seeds=[100+i*2, 101+i*2]) for i in range(30)]
    task_path = directory/'tasks.json'
    atomic_json(task_path, dict(tasks=tasks))
    protocol = dict(task_manifest=dict(path=str(task_path)), all_task_ids_in_order=[t['task_id'] for t in tasks])
    identities = [(t['task_id'], s, r) for t in tasks for s in t['seeds']
                  for r in ('wait_only', 'one_first_legal_plant')]
    jobs, counts = [], Counter()
    for index, (task, seed, regime) in enumerate(identities):
        rows = []
        for prefix, maximum, condition in itertools.product((0, 3000, 9000), (60, 150, 300), CONDITIONS):
            action = dict(type='wait', ticks=maximum, until=condition)
            result = wait_result(action, 0 if condition == 'left_zone_occupied' else maximum)
            if index < 3 and len(rows) == 0:
                reason = ('terminal', 'defense_lost', 'zombie_entered_left_zone')[index]
                result = wait_result(action, 1, reason=reason)
            counts.update(result['triggered'])
            counts['condition:'+condition] += 'condition' in result['triggered']
            counts['initial_true'] += result['initial_condition_satisfied']
            counts['zero_actual_ticks'] += result['actual_ticks'] == 0
            counts['no_event_equivalent'] += result['triggered'] == ['max_ticks']
            rows.append(dict(prefix_requested=prefix, action=action, wait_result=result))
        folder = directory/f'job_{index:04d}'
        folder.mkdir()
        with gzip.open(folder/'raw.jsonl.gz', 'wt') as stream:
            stream.write('synthetic fixture only\n')
        atomic_json(folder/'record.json.gz', dict(operations=[dict(scope='synthetic fixture only')]), compressed=True)
        job = dict(job_id=index, task_id=task, seed=seed, regime=regime, rows=rows,
                   record_replay_passed=True, raw_sha256=audit.sha256_file(folder/'raw.jsonl.gz'))
        atomic_json(folder/'result.json', job)
        jobs.append(job)
    report = dict(status='complete', gate_result='pass', expected_jobs=120, jobs=jobs,
                  counts=dict(counts), missing_coverage=[])
    return report, protocol


class EventReplayAuditContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        audit.configure_torch_threads(1)

    def setUp(self):
        torch.manual_seed(19)
        self.model = audit.GameplayModelV1({**SMALL, 'wait_mode': 'events'}).eval()
        self.episodes = fixture(self.model)

    def test_draft_rejection_precedes_output_hardware_or_process_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, output = Path(temporary)/'draft.json', Path(temporary)/'output'
            path.write_text(json.dumps(dict(release_status='draft')))
            with mock.patch.object(sys, 'argv', ['audit', '--protocol', str(path), '--output-dir', str(output), '--stage', 'collect-cpu']), \
                 mock.patch.object(audit, 'published_bytes') as published, \
                 mock.patch.object(audit, 'configure_runtime') as runtime, \
                 mock.patch.object(audit.multiprocessing, 'get_context') as process:
                with self.assertRaisesRegex(RuntimeError, 'publish the released protocol'):
                    audit.main()
                published.assert_not_called(); runtime.assert_not_called(); process.assert_not_called()
            self.assertFalse(output.exists())

    def test_released_label_cannot_omit_required_source_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'protocol.json'
            path.write_text(json.dumps(dict(release_status='released', required_fingerprints={})))
            with mock.patch.object(audit, 'published_bytes'):
                with self.assertRaisesRegex(ValueError, 'source inventory'):
                    audit.check_protocol(path)

    def test_local_protocol_bytes_must_match_fetched_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, path = Path(temporary), Path(temporary)/'protocol.json'
            path.write_bytes(b'local fixture')
            with mock.patch.object(audit.subprocess, 'check_output', return_value=b'other bytes'):
                with self.assertRaisesRegex(ValueError, 'fetched publication'):
                    audit.published_bytes(root, 'synthetic-ref', path)

    def test_cuda_absence_stops_and_never_selects_cpu(self):
        with mock.patch.object(audit.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CPU fallback is prohibited'):
                audit.configure_runtime('replay-cuda')
        self.assertEqual(audit.os.environ['CUBLAS_WORKSPACE_CONFIG'], ':4096:8')

    def test_full_reference_parameter_count_and_identical_paired_initialization(self):
        states = []
        for mode in ('fixed', 'events'):
            torch.manual_seed(0)
            model = audit.GameplayModelV1({**audit.MODEL, 'wait_mode': mode})
            self.assertEqual(sum(p.numel() for p in model.parameters()), 7869607)
            self.assertEqual(audit.model_architecture_version(model.config), 8)
            states.append(audit._state_sha256(model.state_dict()))
        self.assertEqual(states[0], states[1])

    def test_all_steps_and_carried_chunk_states_agree_for_mixed_waits(self):
        for mode in ('fixed', 'events'):
            torch.manual_seed(19)
            model = audit.GameplayModelV1({**SMALL, 'wait_mode': mode}).eval()
            for episode in fixture(model):
                reference, errors = audit.probe_episode(model, episode, window=3)
                self.assertEqual(reference['scalars'].shape, (len(episode['transitions']), 9))
                self.assertLess(errors['chunk']['log_prob'], 3e-6)
                _, same = audit.probe_episode(model, episode, reference, window=3)
                self.assertEqual(same['cpu_cuda']['log_prob'], 0.)  # Both CPU: storage contract only.

    def test_sampled_log_probability_tampering_fails(self):
        episode = copy.deepcopy(self.episodes[0])
        episode['transitions'][5]['log_prob'] += .001
        with self.assertRaisesRegex(AssertionError, 'tolerance exceeded'):
            audit.probe_episode(self.model, episode, window=3)

    def test_sampled_privileged_value_tampering_fails(self):
        episode = copy.deepcopy(self.episodes[0])
        episode['transitions'][5]['value'] += .001
        with self.assertRaisesRegex(AssertionError, 'tolerance exceeded'):
            audit.probe_episode(self.model, episode, window=3)

    def test_missing_last_wait_metadata_and_nonfinite_reference_fail(self):
        episode = copy.deepcopy(self.episodes[0])
        del episode['transitions'][-1]['wait_result']
        with self.assertRaisesRegex(ValueError, 'metadata is incomplete'):
            audit.probe_episode(self.model, episode)
        reference, _ = audit.probe_episode(self.model, self.episodes[0], window=3)
        reference['scalars'][0, 2] = np.nan
        with self.assertRaisesRegex(FloatingPointError, 'non-finite replay reference'):
            audit.probe_episode(self.model, self.episodes[0], reference, window=3)

    def test_cpu_reference_numeric_archive_and_hash_round_trip(self):
        reference, _ = audit.probe_episode(self.model, self.episodes[0], window=3)
        with tempfile.TemporaryDirectory() as temporary:
            folder, path = Path(temporary), Path(temporary)/'reference.npz'
            atomic_numpy(path, reference, compressed=True)
            saved = read_numpy(audit.verified_file(folder, path.name, audit.sha256_file(path)))
            self.assertEqual(audit.check_rows(saved['scalars'], reference['scalars'])['log_prob'], 0.)
            np.testing.assert_array_equal(saved['final_hidden'], reference['final_hidden'])
            with self.assertRaisesRegex(ValueError, 'path or SHA differs'):
                audit.verified_file(folder, path.name, '0'*64)
            with self.assertRaisesRegex(ValueError, 'path or SHA differs'):
                audit.verified_file(folder, '../outside', '0'*64)

    def test_truncation_bootstrap_is_replayed_and_tampering_fails(self):
        episode = copy.deepcopy(self.episodes[0])
        reference, _ = audit.probe_episode(self.model, episode, window=3)
        last = episode['transitions'][-1]
        context = dict(tokens=last['tokens'], wave=last['wave'], previous_action=last['action'],
            elapsed_since_previous_observation=last['action_duration_ticks'], events={},
            previous_wait_result=last['wait_result'], critic_extra=last['critic_extra'])
        tensors, metadata = audit.unpack_tokens(context['tokens'], torch.device('cpu'))
        with torch.no_grad():
            output = self.model.step_tokens(tensors, metadata, context['wave'], torch.from_numpy(reference['final_hidden']),
                context['previous_action'], context['elapsed_since_previous_observation'], {}, context['previous_wait_result'])
            value = float(self.model.privileged_value_from_extra(output, context['critic_extra']).item())
        episode.update(truncated=True, replay_bootstrap=context, bootstrap_value=value)
        saved, errors = audit.probe_episode(self.model, episode, window=3)
        self.assertEqual(errors['bootstrap_value'], 0.)
        self.assertEqual(saved['bootstrap_value'], value)
        _, errors = audit.probe_episode(self.model, episode, saved, window=3)
        self.assertEqual(errors['cpu_cuda']['bootstrap_value'], 0.)
        episode['bootstrap_value'] += .01
        with self.assertRaisesRegex(AssertionError, 'bootstrap value exceeds'):
            audit.probe_episode(self.model, episode, window=3)

    def test_terminal_bootstrap_is_zero_and_truncated_inputs_are_mandatory(self):
        episode = copy.deepcopy(self.episodes[0])
        episode.update(truncated=False, replay_bootstrap=None, bootstrap_value=0.)
        audit.probe_episode(self.model, episode)
        episode['truncated'] = True
        with self.assertRaisesRegex(ValueError, 'omitted its bootstrap inputs'):
            audit.probe_episode(self.model, episode)

    def test_complete_cpu_case_pipeline_uses_synthetic_environment_only(self):
        class Fixed(audit.RecordingMixin, SyntheticEnv):
            def __init__(self, *args, trace, **kwargs):
                super().__init__(trace=trace, terminal_after=3)

        class Event(audit.RecordingMixin, SyntheticEventEnv):
            def __init__(self, *args, trace, **kwargs):
                super().__init__(trace=trace, terminal_after=4)

        tasks = [dict(task_id=name, level=1, playthrough=2, wave_cap=1,
                      zombie_count_multiplier=1., preplanted=[], deck=[0, 1, 2, 3, 4, 5],
                      sun_start=50, seeds=[100+i*4+j for j in range(4)])
                 for i, name in enumerate(audit.TASK_IDS)]
        for mode in ('fixed', 'events'):
            with tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                (folder/f'{mode}_seed0/collect-cpu').mkdir(parents=True)
                protocol = dict(parameter_count=76303, resource_dir='synthetic-no-resources',
                    executable=dict(path='synthetic-no-executable'), _protocol_sha256='synthetic')
                with mock.patch.dict(audit.MODEL, SMALL, clear=True), \
                     mock.patch.object(audit, 'RecordingFixedEnv', Fixed), \
                     mock.patch.object(audit, 'RecordingEventEnv', Event), \
                     mock.patch.object(audit.os, 'setsid'), mock.patch.object(audit.os, 'dup2'):
                    audit.run_case('collect-cpu', 0, mode, protocol, tasks, str(folder))
                    base = folder/f'{mode}_seed0'
                    manifest = json.loads((base/'collection_manifest.json').read_text())
                    audit.validate_case_manifest(manifest, audit.case_jobs(tasks), 0, mode, 'synthetic')
                    result = json.loads((base/'collect-cpu/result.json').read_text())
                    self.assertEqual(result['gate_result'], 'pass')
                    self.assertEqual(len(result['optimizer_batches']), 10)
                    self.assertTrue(all(e['decisions'] == (3 if mode == 'fixed' else 4) for e in result['episodes']))
                    self.assertTrue(all(e['wait_summary']['wait_actions'] == e['decisions'] for e in result['episodes']))

    def test_actual_cpu_adamw_zero_lr_preserves_weights_and_exercises_prefixes(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.)
        # A short synthetic fixture uses 3-step windows to exercise recomputed
        # prefixes; formal protocol remains fixed at 128 and uses the full model.
        with mock.patch.dict(audit.PPO, sequence_length=3):
            losses = audit.zero_lr_update(self.model, self.episodes, optimizer, None)
        self.assertGreater(losses['optimizer_steps'], 2)
        self.assertLess(losses['max_log_prob_change'], 3e-6)
        self.assertEqual(losses['clip_fraction'], 0)
        self.assertTrue(optimizer.state)

    def test_native_pass_flag_does_not_replace_complete_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, protocol = native_fixture(Path(temporary))
            verified = audit.validate_native_report(report, protocol, Path(temporary))
            self.assertEqual(len(verified), 120)
            incomplete = copy.deepcopy(report)
            incomplete['jobs'].pop()
            with self.assertRaisesRegex(ValueError, 'inventory is incomplete'):
                audit.validate_native_report(incomplete, protocol, Path(temporary))
            altered = copy.deepcopy(report)
            altered['jobs'][0]['rows'][0]['action']['ticks'] = 150
            with self.assertRaisesRegex(ValueError, 'branch inventory differs'):
                audit.validate_native_report(altered, protocol, Path(temporary))
            (Path(temporary)/'job_0000/raw.jsonl.gz').write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError, 'raw trace differs'):
                audit.validate_native_report(report, protocol, Path(temporary))

    def test_native_coverage_must_match_rows_not_reported_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            report, protocol = native_fixture(Path(temporary))
            report['counts']['terminal'] += 1
            with self.assertRaisesRegex(ValueError, 'positive coverage differs'):
                audit.validate_native_report(report, protocol, Path(temporary))

    def test_cpu_pass_flag_does_not_replace_six_case_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'complete CPU gate'):
                audit.validate_cpu_report(dict(status='complete', gate_result='pass',
                    stage='collect-cpu', protocol_sha256='synthetic', episodes=120, cases=[]),
                    Path(temporary), [], 'synthetic')

    def test_recording_flushes_request_before_a_failing_environment_step(self):
        class FailureEnv:
            def __init__(self, *args, **kwargs):
                pass

            def step(self, action):
                raise RuntimeError('synthetic worker failure')

        class RecordingFailure(audit.RecordingMixin, FailureEnv):
            pass

        events = []
        env = RecordingFailure(trace=lambda kind, payload: events.append((kind, payload)))
        with self.assertRaisesRegex(RuntimeError, 'synthetic worker failure'):
            env.step(dict(type='wait', ticks=60))
        self.assertEqual(events, [('action_request', dict(type='wait', ticks=60))])


if __name__ == '__main__':
    unittest.main()
