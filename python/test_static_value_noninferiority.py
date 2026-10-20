"""Exact cancellation retains the full paired estimand, including both-wrong cases."""
import itertools
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from static_value_noninferiority import correctness_difference, measured_arm


class CancellationTest(unittest.TestCase):
    def test_skipped_agreements_reproduce_full_reference_correctness_differences(self):
        actions = [dict(type='plant', packet=0, row=0, col=c) for c in range(3)]
        actions.append(dict(type='wait', ticks=60))
        for a, b, heavy in itertools.product(actions, repeat=3):
            expected = int(b == heavy) - int(a == heavy)
            supplied = None if a == b else heavy
            self.assertEqual(correctness_difference(a, b, supplied), expected)

    def test_disagreement_cannot_be_inferred_without_heavy_and_both_wrong_is_zero(self):
        a, b, c = [dict(type='plant', packet=0, row=0, col=i) for i in range(3)]
        with self.assertRaisesRegex(ValueError, 'requires'):
            correctness_difference(a, b)
        self.assertEqual(correctness_difference(a, b, c), 0)
        self.assertEqual(correctness_difference(a, b, a), -1)
        self.assertEqual(correctness_difference(a, b, b), 1)

    def test_cpu_includes_worker_and_live_simulator_not_elapsed_wall(self):
        from pvz_cpu_usage import difference
        env = SimpleNamespace(_process=SimpleNamespace(pid=17))
        worker = [{'user_s': 1., 'system_s': .1}, {'user_s': 1.3, 'system_s': .15}]
        sim = [{'user_s': 10., 'system_s': 2.}, {'user_s': 10.8, 'system_s': 2.2}]
        with patch('pvz_cpu_usage.usage', side_effect=worker), \
             patch('pvz_cpu_usage.process_usage', side_effect=sim) as read, \
             patch('static_value_noninferiority.normal', return_value={'seconds': 99., 'chosen': {}}):
            result = measured_arm(env, {}, {}, None, 3)
        self.assertAlmostEqual(result['cpu']['total_s'], 1.35)
        self.assertNotEqual(result['cpu']['total_s'], result['seconds'])
        self.assertEqual(read.call_count, 2)


class NoninferiorityStatisticsTest(unittest.TestCase):
    @staticmethod
    def game(delta):
        return {'points':8,'delta':delta,'A_cost':100,'B3_cost':80,
                'A_cpu':10,'B3_cpu':6,'A_worker_cpu':2,'B3_worker_cpu':1,
                'A_sim_cpu':8,'B3_sim_cpu':5}

    def test_margin_and_formal_gate_do_not_turn_interim_or_deficit_into_success(self):
        from static_value_noninferiority_stats import summarize
        games=[self.game(0)]*128
        interim=summarize(games,formal=False)
        self.assertIsNone(interim['primary']['noninferiority_passed'])
        final=summarize(games*2)
        self.assertTrue(final['primary']['robust_NI'])
        deficit=summarize([self.game(-1)]*256)
        self.assertFalse(deficit['primary']['noninferiority_passed'])

    def test_paired_uncertainty_uses_game_means_and_cpu_ratio_is_not_lane_ratio(self):
        from static_value_noninferiority_stats import summarize
        import numpy as np
        games=[self.game(-1 if i%4==0 else 0) for i in range(256)]
        r=summarize(games)
        self.assertEqual(r['primary']['df'],255)
        self.assertAlmostEqual(r['primary']['difference_B3_minus_A'],-.03125)
        expected=np.std([g['delta']/8 for g in games],ddof=1)/16
        self.assertAlmostEqual(r['primary']['se_of_game_means'],expected)
        self.assertEqual(r['metrics']['B3_cost_saving']['estimate'],1-.8)
        self.assertAlmostEqual(r['metrics']['B3_cpu_saving']['estimate'],.4)
        self.assertFalse(r['primary']['noninferiority_passed'])


if __name__ == '__main__':
    unittest.main()
