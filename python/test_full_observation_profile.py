"""Validate profiling inventory, preserved failures and descriptive counts."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_full_observation_profile as profile


class FullObservationProfileTests(unittest.TestCase):
    def protocol(self):
        return {'tasks': [{'task_id': 'full', 'wave_cap': None, 'seeds': [30000, 30001]}],
                'strategies': ['wait', 'scripted'], 'expected_episodes': 4}

    def test_complete_inventory_keeps_every_seed_and_control(self):
        jobs = profile.make_jobs(self.protocol())
        self.assertEqual([(j['seed'], j['strategy']) for j in jobs],
                         [(30000, 'wait'), (30001, 'wait'), (30000, 'scripted'), (30001, 'scripted')])
        self.assertEqual([j['job'] for j in jobs], list(range(4)))

    def test_duplicates_caps_and_missing_jobs_rejected(self):
        for change in ('duplicate', 'cap', 'missing'):
            p = self.protocol()
            if change == 'duplicate': p['tasks'][0]['seeds'] = [30000, 30000]
            if change == 'cap': p['tasks'][0]['wave_cap'] = 5
            if change == 'missing': p['expected_episodes'] = 5
            with self.assertRaises(ValueError): profile.make_jobs(p)

    def test_histogram_uses_all_states_and_handles_no_late_coverage(self):
        result = profile.histogram_summary({'10': 1, '20': 1, '40': 2})
        self.assertEqual(result['count'], 4)
        self.assertEqual(result['median'], 30)
        self.assertEqual(result['p95'], 40)
        self.assertEqual(result['dense_token_pairs'], 3700)
        self.assertEqual(profile.histogram_summary({})['count'], 0)
        self.assertIsNone(profile.histogram_summary({})['max'])

    def test_cohort_keeps_loss_and_truncation(self):
        common = {'task_id': 'full', 'strategy': 'wait', 'token_histogram': {'10': 2},
                  'late_token_histogram': {}, 'packed_array_bytes_total': 16,
                  'public_observation_json_bytes_total': 40, 'tokenize_pack_seconds': .1}
        rows = [{**common, 'won': True, 'truncated': False},
                {**common, 'won': False, 'truncated': False},
                {**common, 'won': False, 'truncated': True}]
        result = profile.summarize(rows)['full/wait']
        self.assertEqual((result['episodes'], result['won'], result['truncated']), (3, 1, 1))
        self.assertEqual(result['tokens']['count'], 6)
        self.assertEqual(result['episodes_reaching_late_waves'], 0)
        self.assertLess(result['wilson_95'][0], 1/3)
        self.assertGreater(result['wilson_95'][1], 1/3)

    def test_legal_cell_omission_stops_encoding(self):
        observation = {'tick': 0, 'wave': 0, 'wave_count': 30, 'terminal': False,
                       'plants': [], 'zombies': [], 'projectiles': [],
                       'legal_actions': {'plants': [{'row': 0, 'col': 3}]}}
        encoder = SimpleNamespace(observation_tokens=lambda obs, flags: ({'kinds': [0, 2]}, {'cell_tokens': {}}),
                                  pack_tokens=lambda tensors, metadata: {'ids': SimpleNamespace(nbytes=16)},
                                  TOKEN_KINDS={'global': 0, 'cell': 2})
        with patch.object(profile, 'ENCODER', encoder):
            with self.assertRaises(ValueError): profile.encode_stats(observation, 7)

    def test_source_fingerprint_drift_stops_before_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / 'native'
            executable.write_text('fixture')
            gate = root / 'gate.json'
            gate.write_text(json.dumps({'gate_result': 'pass', 'simulator_sha256': profile.digest(executable),
                                        'required_fingerprints': {}}))
            p = {**self.protocol(), 'purpose': 'full_level_public_input_cost_only_no_learning',
                 'required_fingerprints': {'native': profile.digest(executable)},
                 'executable': str(executable), 'native_gate': str(gate), 'encoder_root': str(root)}
            path = root / 'protocol.json'
            path.write_text(json.dumps(p))
            with patch.object(profile, 'ROOT', root):
                self.assertEqual(profile.checked_protocol(path), p)
                full = dict(p, expected_episodes=2048)
                path.write_text(json.dumps(full))
                with self.assertRaisesRegex(ValueError, 'complete inspection'):
                    profile.checked_protocol(path)
                path.write_text(json.dumps(p))
                executable.write_text('changed')
                with self.assertRaises(ValueError): profile.checked_protocol(path)


if __name__ == '__main__':
    unittest.main()
