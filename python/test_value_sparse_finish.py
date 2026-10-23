import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import value_sparse_finish as finish


class SparseFinishTest(unittest.TestCase):
    def test_data_gate_rejects_incomplete_or_changed_shards(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out / 'data').mkdir()
            seeds = list(range(64)); plan = dict(out=d, seeds=seeds, held_out_seeds=seeds[-8:])
            (out / 'pipeline_status.json').write_text(json.dumps(dict(stage='data_complete_training_pending')))
            rows = []
            for seed in seeds:
                p = out / 'data' / f'seed_{seed}.pt'; p.write_text(str(seed))
                rows.append(dict(seed=seed, rows=1, sha256=finish.digest(p)))
            exported = dict(status='complete', seeds=rows, held_out_seeds=seeds[-8:])
            (out / 'export.json').write_text(json.dumps(exported))
            self.assertEqual(len(finish.data_gate(plan)['seeds']), 64)
            shard = out / 'data' / 'seed_63.pt'; shard.write_text('changed')
            with self.assertRaises(finish.PipelineStop): finish.data_gate(plan)
            shard.unlink()
            with self.assertRaises(finish.PipelineStop): finish.data_gate(plan)
            (out / 'pipeline_status.json').write_text(json.dumps(dict(stage='stopped')))
            with self.assertRaises(finish.PipelineStop): finish.data_gate(plan)

    def test_low_memory_or_student_training_blocks_training(self):
        with patch.object(finish.Path, 'read_text', return_value='MemAvailable: 10485759 kB\n'), \
             patch.object(finish.subprocess, 'run', return_value=Mock(returncode=1)):
            self.assertFalse(finish.training_ready()[0])
        with patch.object(finish.Path, 'read_text', return_value='MemAvailable: 11534336 kB\n'), \
             patch.object(finish.subprocess, 'run', return_value=Mock(returncode=0)):
            self.assertFalse(finish.training_ready()[0])
        with patch.object(finish.Path, 'read_text', return_value='MemAvailable: 11534336 kB\n'), \
             patch.object(finish.subprocess, 'run', return_value=Mock(returncode=1)):
            self.assertTrue(finish.training_ready()[0])

    def test_failed_dispersion_does_not_skip_any_paired_task(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); ev = out / 'evaluation'; ev.mkdir()
            for label in ('l21', 'day', 'night'):
                (ev / f'ab_{label}').mkdir()
            plan = dict(out=d, training=dict(out=str(out / 'v5')),
                teacher=dict(value='/baseline.pt', executable='/engine'),
                evaluation=dict(calibration_manifest='/A6.json', manifest='/frozen.json'))
            calls = []
            def call(_plan, command, log):
                calls.append(command)
                if log.name == 'calibration.log':
                    (ev / 'calibration').mkdir()
                    (ev / 'calibration' / 'metrics.json').write_text(json.dumps(dict(cohorts={})))
                elif log.name == 'dispersion.log':
                    (ev / 'dispersion.json').write_text(json.dumps(dict(per_level=dict(v4=dict(mean_std=0.001)))))
                elif 'report' in command:
                    directory = Path(command[command.index('--out') + 1])
                    (directory / 'paired.json').write_text(json.dumps(dict(complete=True, results={})))
            with patch.object(finish, 'checked_call', side_effect=call), patch.object(finish, 'verify_models'):
                finish.evaluate(plan, 'modelsha', Mock(), Mock())
            arms = [command[2] for command in calls if command[1].endswith('value_widen_ab.py')]
            self.assertEqual(arms, ['v3', 'v4', 'report'] * 3)
            self.assertTrue(all(command[command.index('--workers') + 1] == '4'
                                for command in calls if '--workers' in command))

    def test_one_failed_gate_prevents_hf_upload(self):
        # An empty plan would fail immediately if any upload work were attempted.
        self.assertIsNone(finish.upload_if_accepted({}, dict(all_three_pass=False)))

    def test_artifact_budget_stops_at_exact_approved_limit(self):
        plan = dict(artifact_root='/artifacts', artifact_budget_bytes=5000000000)
        with patch.object(finish, 'artifact_bytes', return_value=5000000000):
            self.assertEqual(finish.resource_gate(plan), 5000000000)
        with patch.object(finish, 'artifact_bytes', return_value=5000000001):
            with self.assertRaises(finish.PipelineStop): finish.resource_gate(plan)


if __name__ == '__main__': unittest.main()
