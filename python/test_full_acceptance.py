"""Adversarial checks for complete cohorts and T7 capability interpretation."""
from copy import deepcopy
import json
import unittest

import evaluate_full_acceptance as full


class FullAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(full.MANIFEST.read_text())

    def rows(self, level7_wins=154, terrain_wins=39):
        grouped = {}
        for mode in self.manifest['modes']:
            grouped[mode] = {}
            for task in self.manifest['tasks']:
                won = level7_wins if task['task_id']=='level7_development' else terrain_wins
                grouped[mode][task['task_id']] = [dict(seed=s,won=i<won,
                    result=1 if i<won else 2,terminated=True,truncated=False,
                    terminal_wave=30) for i,s in enumerate(task['seeds'])]
        return grouped

    def reports(self, passes):
        return [dict(status='complete',initialization_seed=s,origin_initialization_seed=s,
            manifest=self.manifest,model_config={'width':256},simulator_sha256='same-native',
            thresholds=full.THRESHOLDS,modes={m:dict(single_candidate_pass=passes[s][i])
                for i,m in enumerate(('greedy','sampled'))}) for s in range(3)]

    def test_empirical_threshold_does_not_require_wilson_lower_bound(self):
        result = full.summarize(self.manifest,self.rows())['greedy']
        self.assertTrue(result['single_candidate_pass'])
        self.assertLess(result['per_task']['level7_development']['wilson_95'][0],0.6)

    def test_level7_large_cohort_does_not_dominate_five_terrain_macro(self):
        result = full.summarize(self.manifest,self.rows(256,29))['greedy']
        self.assertTrue(result['checks']['level7'])
        self.assertTrue(result['checks']['each_terrain'])
        self.assertFalse(result['checks']['terrain_macro'])
        self.assertFalse(result['single_candidate_pass'])

    def test_missing_failure_seed_cannot_be_dropped(self):
        grouped = self.rows(); grouped['greedy']['level7_development'].pop()
        with self.assertRaises(ValueError): full.summarize(self.manifest,grouped)

    def test_repeated_seed_cannot_replace_a_failure(self):
        grouped = self.rows();rows = grouped['sampled']['full_roof41'];rows[-1]=deepcopy(rows[0])
        with self.assertRaises(ValueError): full.summarize(self.manifest,grouped)

    def test_budget_truncation_cannot_be_claimed_as_a_win(self):
        grouped = self.rows();row = grouped['greedy']['full_pool26'][0]
        row.update(terminated=False,truncated=True)
        with self.assertRaises(ValueError): full.summarize(self.manifest,grouped)

    def test_declared_deck_cannot_be_substituted(self):
        manifest = deepcopy(self.manifest);manifest['tasks'][0]['deck']=[0,1]
        with self.assertRaises(ValueError): full.summarize(manifest,self.rows())

    def test_five_terrain_frozen_failure_seed_cannot_be_substituted(self):
        manifest = deepcopy(self.manifest);manifest['tasks'][1]['seeds'][-1]=99999
        with self.assertRaises(ValueError): full.summarize(manifest,self.rows())

    def test_all_failures_are_not_acceptance(self):
        result = full.summarize(self.manifest,self.rows(0,0))
        self.assertTrue(all(not r['single_candidate_pass'] for r in result.values()))

    def test_passes_from_different_modes_are_not_combined(self):
        result = full.aggregate_reports(self.reports([(True,False),(False,True),(False,False)]))
        self.assertTrue(all(not r['two_of_three_capability_pass'] for r in result['per_mode'].values()))

    def test_two_passes_in_same_mode_retain_the_third_failed_initialization(self):
        result = full.aggregate_reports(self.reports([(True,False),(True,False),(False,False)]))
        self.assertEqual(result['per_mode']['greedy']['passed_initialization_seeds'],[0,1])
        self.assertTrue(result['per_mode']['greedy']['two_of_three_capability_pass'])
        self.assertFalse(result['per_mode']['sampled']['two_of_three_capability_pass'])

    def test_transferred_same_origin_cannot_count_as_independent_seeds(self):
        reports = self.reports([(True,True)]*3);reports[1]['origin_initialization_seed']=0
        with self.assertRaises(ValueError): full.aggregate_reports(reports)

    def test_missing_third_initialization_is_incomplete(self):
        with self.assertRaises(ValueError): full.aggregate_reports(self.reports([(True,True)]*3)[:2])

    def test_aggregate_cannot_accept_lowered_thresholds(self):
        reports = self.reports([(True,True)]*3)
        for report in reports:report['thresholds']={**full.THRESHOLDS,'level7_pass_rate':0.1}
        with self.assertRaises(ValueError): full.aggregate_reports(reports)


if __name__ == '__main__': unittest.main()
