"""Random public controls keep legal coordinates and explicit RNG pairing."""
from collections import Counter
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import research_random_target_coverage as coverage


class RandomTargetCoverageTests(unittest.TestCase):
    def test_only_wait_when_no_instant_actions(self):
        observation = {'legal_actions': {'plants': [], 'shovels': []}}
        rng = random.Random(3)
        for _ in range(20):
            action = coverage.choose(observation, rng)
            self.assertEqual(action['type'], 'wait')
            self.assertIn(action['ticks'], (60, 150, 300))

    def test_coordinates_and_rng_are_reproducible(self):
        observation = {'legal_actions': {'plants': [{'packet': 2, 'col': 7, 'row': 3}],
                                          'shovels': [[4, 1]]}}
        def sample():
            rng = random.Random(7)
            return [coverage.choose(observation, rng) for _ in range(100)]
        actions = sample()
        self.assertEqual(actions, sample())
        self.assertEqual(set(Counter(a['type'] for a in actions)), {'plant', 'wait', 'shovel'})
        for action in actions:
            if action['type'] == 'plant':
                self.assertEqual(action, {'type': 'plant', 'packet': 2, 'col': 7, 'row': 3})
            elif action['type'] == 'shovel':
                self.assertEqual(action, {'type': 'shovel', 'col': 4, 'row': 1})


if __name__ == '__main__':
    unittest.main()
