"""The hash suite must reject partial samples and cross-mode divergence."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from task95_hash_suite import stable


class HashSuiteTests(unittest.TestCase):
    def setUp(self):
        row = {'hashes': ['a', 'b', 'c'], 'ticks': [0, 60, 660]}
        self.multi = [copy.deepcopy(row) for _ in range(8)]
        self.same = [copy.deepcopy(row) for _ in range(4)]

    def test_stable_in_both_modes(self):
        self.assertTrue(stable(self.multi, self.same))

    def test_same_mode_stability_does_not_hide_cross_mode_difference(self):
        for row in self.same:
            row['hashes'][2] = 'different'
        self.assertFalse(stable(self.multi, self.same))

    def test_one_process_can_diverge(self):
        self.multi[7]['hashes'][0] = 'different'
        self.assertFalse(stable(self.multi, self.same))

    def test_incomplete_and_missing_samples_fail(self):
        self.assertFalse(stable(self.multi[:-1], self.same))
        self.same[0]['hashes'] = []
        self.assertFalse(stable(self.multi, self.same))


if __name__ == '__main__':
    unittest.main()
