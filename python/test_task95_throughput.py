import copy
from pathlib import Path
import statistics
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task95_throughput_abba import evaluate


def panel(old=98.0, new=100.0):
    return {'pass': True, 'games': [dict(cell=cell, seed=seed, old_seconds=old,
            new_seconds=new, old_decisions=100, new_decisions=100)
            for cell in ('day', 'pool') for seed in range(8)]}


class Task95ThroughputTest(unittest.TestCase):
    def test_exact_throughput_boundary_passes(self):
        self.assertTrue(evaluate(panel())['pass'])

    def test_legacy_three_percent_rule_does_not_override_task95(self):
        result = evaluate(panel(100.0, 103.0))
        self.assertTrue(result['legacy_a4_time_rule_pass'])
        self.assertFalse(result['pass'])

    def test_uses_median_of_paired_speeds_not_inverse_of_median_time(self):
        data = panel(100.0, 99.0)
        for row in data['games'][8:]:
            row['new_seconds'] = 101.0
        result = evaluate(data)
        self.assertEqual(result['median_throughput_ratio'],
                         statistics.median([100 / 99] * 8 + [100 / 101] * 8))
        self.assertNotEqual(result['median_throughput_ratio'], 1.0)

    def test_decision_difference_fails_even_when_fast(self):
        data = panel(100.0, 90.0)
        data['games'][0]['new_decisions'] = 101
        self.assertFalse(evaluate(data)['pass'])

    def test_incomplete_or_duplicate_panel_is_refused(self):
        data = panel()
        with self.assertRaisesRegex(ValueError, 'complete 16-pair'):
            evaluate({**data, 'games': data['games'][:-1]})
        duplicate = copy.deepcopy(data)
        duplicate['games'][0] = duplicate['games'][1]
        with self.assertRaisesRegex(ValueError, 'complete 16-pair'):
            evaluate(duplicate)


if __name__ == '__main__':
    unittest.main()
