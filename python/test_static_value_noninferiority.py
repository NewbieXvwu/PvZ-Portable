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


if __name__ == '__main__':
    unittest.main()
