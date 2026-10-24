"""Preregistered paired-test, multiplicity and cluster-denominator checks."""
from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replace_action_stats as stats


class ReplacementStatisticsTest(unittest.TestCase):
    def test_exact_discordance_and_holm(self):
        self.assertEqual(stats.exact_mcnemar(0, 0), 1)
        self.assertEqual(stats.exact_mcnemar(3, 3), 1)
        self.assertEqual(stats.exact_mcnemar(0, 6), .03125)
        self.assertEqual(stats.exact_mcnemar(6, 0), .03125)
        self.assertEqual(stats.holm([.03125, .04, .9]), [.09375, .09375, .9])

    def game(self, seed, won):
        return dict(seed=seed, won=won, terminal=True,
                    study=dict(verification_by_type={}, replace_events=[], replace_attempts=0,
                               replace_rejected=0, sun_produced=0, sun_spent=0, sun_remaining=0))

    def test_paired_bootstrap_uses_games_and_rejects_unpaired_or_duplicate_seeds(self):
        before = [self.game(seed, False) for seed in range(6)]
        after = [self.game(seed, True) for seed in reversed(range(6))]
        with patch.object(stats, 'BOOTSTRAPS', 1000):
            panel = stats.paired_panel(before, after, 69)
        self.assertEqual(panel['after_only_win'], 6)
        self.assertEqual(panel['win_rate_difference'], dict(mean=1.0, ci95=[1.0, 1.0]))
        self.assertEqual(panel['mcnemar_two_sided_p'], .03125)
        with self.assertRaisesRegex(ValueError, 'seed sets'):
            stats.paired_panel(before, after[:-1], 69)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            stats.paired_panel(before + before, after + after, 69)

    def test_revoke_ratio_pools_counts_with_game_resampling_and_preserves_empty_denominator(self):
        games = [self.game(1, False), self.game(2, True)]
        games[0]['study']['verification_by_type']['replace'] = dict(revoked=1, attempts=2)
        games[1]['study']['verification_by_type']['replace'] = dict(revoked=0, attempts=8)
        rng = np.random.default_rng(69)
        with patch.object(stats, 'BOOTSTRAPS', 1000):
            ratio = stats.verification_summary(games, 'replace', rng)
            missing = stats.verification_summary(games, 'plant', rng)
        self.assertEqual((ratio['revoked'], ratio['attempts'], ratio['ratio']), (1, 10, .1))
        self.assertEqual(ratio['ci95'], [0, .5])
        self.assertIsNone(missing['ratio'])
        self.assertIsNone(missing['ci95'])


if __name__ == '__main__':
    unittest.main()
