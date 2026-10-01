import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from research_packet_cost_reevaluation_v2 import check_scope, conveyor_task_ids, persisted_inventory, CONVEYOR_ADVENTURE_LEVELS


class ScopeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tasks = [{'task_id': 'ordinary_roof_label', 'level': 45},
                      {'task_id': 'roof_training', 'level': 42},
                      {'task_id': 'legacy_night', 'level': 20}]
        self.ids = ['ordinary_roof_label', 'legacy_night']
        self.protocol = {'allowed_changed_task_ids': self.ids}
        self.gate = {'conveyor_task_ids': self.ids, 'result': 'pass',
                     'adventure_conveyor_levels': list(CONVEYOR_ADVENTURE_LEVELS),
                     'roof_native_equivalence': {'paired_jobs': 576, 'recorded_traces': 32},
                     'roof_policy_repeatability': {'guarded32_all_observation_output_action_rng_exact': True}}

    def test_native_level_overrides_misleading_task_label(self):
        self.assertEqual(conveyor_task_ids(self.tasks), self.ids)
        check_scope(self.tasks, self.protocol, self.gate)

    def test_original_omission_and_nonconveyor_expansion_rejected(self):
        for ids in [['legacy_night'], ['ordinary_roof_label', 'roof_training', 'legacy_night']]:
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                check_scope(self.tasks, {'allowed_changed_task_ids': ids}, self.gate)

    def test_native_physics_and_policy_proofs_both_required(self):
        for key in ['result', 'native', 'replay', 'policy']:
            gate = copy.deepcopy(self.gate)
            if key == 'result': gate['result'] = 'fail'
            if key == 'native': gate['roof_native_equivalence']['paired_jobs'] = 575
            if key == 'replay': gate['roof_native_equivalence']['recorded_traces'] = 0
            if key == 'policy': gate['roof_policy_repeatability']['guarded32_all_observation_output_action_rng_exact'] = False
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_scope(self.tasks, self.protocol, gate)

    def test_frozen_seed_inventory_survives_json_without_hiding_digest_drift(self):
        current = {'paired_initial_states': {0: 'model0', 1: 'model1', 2: 'model2'},
                   'nodes': [{'checkpoint_sha256': 'checkpoint', 'original_raw_sha256': 'raw'}]}
        saved = {'paired_initial_states': {'0': 'model0', '1': 'model1', '2': 'model2'},
                 'nodes': [{'checkpoint_sha256': 'checkpoint', 'original_raw_sha256': 'raw'}]}
        self.assertEqual(persisted_inventory(current), saved)
        changed = copy.deepcopy(current)
        changed['nodes'][0]['checkpoint_sha256'] = 'replacement'
        self.assertNotEqual(persisted_inventory(changed), saved)


if __name__ == '__main__':
    unittest.main()
