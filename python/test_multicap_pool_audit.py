"""Keep original tasks, full seed inventories and target-identity semantics."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_multicap_pool_audit as audit


class MulticapPoolAuditTests(unittest.TestCase):
    def fixture(self):
        source, additions = [], []
        for terrain_index, terrain in enumerate(('day', 'night', 'pool', 'fog', 'roof')):
            for index, cap in enumerate((1, 3, 3, 5)):
                start = 60000 + len(source)*64
                source.append({'task_id': f'train_{terrain}_{index+1}', 'terrain': terrain,
                               'level': 1 + terrain_index*10 + index,
                               'deck': [0, 1, 2, 3, 4, 5], 'wave_cap': cap,
                               'zombie_count_multiplier': (1., 1.5, 2., 1.)[index],
                               'sun_start': 50, 'preplanted': [], 'playthrough': 2,
                               'seeds': list(range(start, start+64))})
        tasks = deepcopy(source)
        for terrain in ('day', 'night', 'pool', 'fog', 'roof'):
            for cap, parent_index in ((1, 2), (5, 3)):
                inherited = next(t for t in source if t['task_id'] == f'train_{terrain}_{parent_index}')
                start = 130000 + len(additions)*64
                entry = {'source_task_id': inherited['task_id'], 'task_id': f'new_{terrain}_{cap}',
                         'wave_cap': cap, 'seeds': list(range(start, start+64))}
                additions.append(entry)
                tasks.append(dict(inherited, task_id=entry['task_id'], wave_cap=cap,
                                  zombie_count_multiplier=1., seeds=entry['seeds']))
        return {'tasks': source}, {'split': 'train', 'tasks': tasks, 'generation': {'additions': additions}}

    def checked(self, source, pool, evaluation_tasks=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'source.json').write_text(json.dumps(source))
            (root/'pool.json').write_text(json.dumps(pool))
            protocol = {'original_manifest': 'source.json', 'manifest': 'pool.json',
                        'adventure_conveyor_levels': [], 'evaluation_manifests': []}
            if evaluation_tasks is not None:
                (root/'eval.json').write_text(json.dumps({'tasks': evaluation_tasks}))
                protocol['evaluation_manifests'] = ['eval.json']
            with patch.object(audit, 'ROOT', root):
                return audit.checked_pool(protocol)

    def test_original_and_new_tasks_preserved_with_full_seed_blocks(self):
        source, pool = self.fixture()
        self.assertEqual(self.checked(source, pool), pool['tasks'])

    def test_original_task_or_failed_seed_cannot_be_changed(self):
        source, pool = self.fixture()
        for field, value in [('wave_cap', 5), ('seeds', pool['tasks'][0]['seeds'][:-1])]:
            changed = deepcopy(pool)
            changed['tasks'][0][field] = value
            with self.assertRaises(ValueError):
                self.checked(source, changed)

    def test_unregistered_new_deck_or_aid_rejected(self):
        source, pool = self.fixture()
        for field, value in [('deck', [0, 1, 2, 3, 4, 17]), ('preplanted', [[0, 0, 0]])]:
            changed = deepcopy(pool)
            changed['tasks'][20][field] = value
            with self.assertRaises(ValueError):
                self.checked(source, changed)

    def test_training_probe_is_explicit_and_validation_remains_disjoint(self):
        source, pool = self.fixture()
        probe = dict(source['tasks'][0], evaluation_role='training_probe')
        self.assertEqual(len(self.checked(source, pool, [probe])), 30)
        for changed in (dict(probe, evaluation_role='validation'), dict(probe, wave_cap=5)):
            with self.assertRaises(ValueError):
                self.checked(source, pool, [changed])

    def test_paired_controls_keep_every_seed(self):
        _, pool = self.fixture()
        for count, expected in ((2, 120), (64, 3840)):
            p = {'seeds_per_task': count, 'expected_episodes': expected, 'strategies': ['wait', 'scripted']}
            jobs = audit.make_jobs(pool['tasks'], p)
            self.assertEqual([j['job'] for j in jobs], list(range(expected)))
            self.assertEqual(len({(j['task']['task_id'], j['strategy'], j['seed']) for j in jobs}), expected)
        with self.assertRaises(ValueError):
            audit.make_jobs(pool['tasks'], dict(p, expected_episodes=3839))

    def test_negative_target_id_is_real_identity_except_sentinels(self):
        observation = {'plants': [{'id': 1, 'target_zombie_id': -2}, {'id': 2, 'target_zombie_id': -1}],
                       'projectiles': [{'target_zombie_id': 0}, {'target_zombie_id': 7}]}
        refs, resolved = audit.target_stats(observation, {'target_indices': np.array([-1, 6, -1, 9])})
        self.assertEqual([r['target_zombie_id'] for r in refs], [-2, 7])
        self.assertEqual(resolved, 2)
        self.assertIsNone(refs[1]['id'])
        with self.assertRaises(ValueError):
            audit.target_stats(observation, {'target_indices': np.array([0, 1, 2])})

    def test_losses_and_truncations_remain_in_summary(self):
        base = {'task_id': 'task', 'strategy': 'scripted', 'token_histogram': {'10': 2},
                'target_states': 0, 'target_references': 0, 'resolved_targets': 0}
        rows = [dict(base, won=True, truncated=False), dict(base, won=False, truncated=False),
                dict(base, won=False, truncated=True)]
        summary = audit.summarize(rows)['task/scripted']
        self.assertEqual((summary['episodes'], summary['won'], summary['truncated']), (3, 1, 1))
        self.assertLess(summary['wilson_95'][0], 1/3)
        self.assertGreater(summary['wilson_95'][1], 1/3)

    def test_source_drift_and_missing_inspection_stop_full_release(self):
        _, pool = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root/'native'
            executable.write_text('fixture')
            gate = root/'gate.json'
            gate.write_text(json.dumps({'gate_result': 'pass', 'simulator_sha256': audit.profile.digest(executable),
                                        'required_fingerprints': {}}))
            protocol = {'purpose': 'multicap_noaid_pool_audit_only_no_learning',
                        'required_fingerprints': {'native': audit.profile.digest(executable)},
                        'native_gate': str(gate), 'executable': str(executable),
                        'strategies': ['wait', 'scripted'], 'seeds_per_task': 2, 'expected_episodes': 120}
            path = root/'protocol.json'
            path.write_text(json.dumps(protocol))
            with patch.object(audit, 'ROOT', root), patch.object(audit, 'checked_pool', return_value=pool['tasks']):
                self.assertEqual(audit.checked_protocol(path)[0], protocol)
                path.write_text(json.dumps(dict(protocol, seeds_per_task=64, expected_episodes=3840)))
                with self.assertRaisesRegex(ValueError, 'complete inspection'):
                    audit.checked_protocol(path)
                path.write_text(json.dumps(protocol))
                executable.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'source/evidence changed'):
                    audit.checked_protocol(path)


if __name__ == '__main__':
    unittest.main()
