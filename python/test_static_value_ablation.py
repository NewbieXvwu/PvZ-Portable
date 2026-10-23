"""The experimental generators must skip static branches and retain the policy anchor."""
from collections import Counter
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import search_play as t


class GeneratorTest(unittest.TestCase):
    def run_generator(self, mode, k=2, policy=None):
        # Global top3 deliberately all belong to one card. The policy can be a
        # different legal plant: ranked joint probabilities need not match act's
        # hierarchical greedy packet/cell choice.
        plants = [dict(type='plant', packet=p, row=0, col=c) for p in (0, 1) for c in range(3)]
        ranked = [(1-i/10, a) for i, a in enumerate(plants)]
        policy = policy or t.NOOP
        args = SimpleNamespace(candidate_generator=mode, student_candidates=k,
                               per_card=1, min_gap=0, shovel_candidates=0, halving=True)
        env = Mock()
        stats = {'searched': 0, 'overrides': 0, 'override_kinds': Counter()}
        with patch.dict(t._W, {'fast': object(), 'student_frame': {'tick': 60}}, clear=True), \
             patch.dict(sys.modules, {'pvz_rollout_policy': SimpleNamespace(ranked_plants=Mock(return_value=ranked))}), \
             patch.object(t, '_static_scores', side_effect=AssertionError('static branches forbidden')), \
             patch.object(t, '_halving_choice', side_effect=lambda *a: a[3]) as halve:
            candidates = t._search(env, {'tick':60, 'legal_actions': {'plants':plants}}, policy, args, 17, stats)
            self.assertEqual(halve.call_count, 1)
        env.release_snapshot.assert_called_once()
        self.assertEqual(len({str(sorted(a.items())) for a in candidates}), len(candidates))
        return candidates, plants

    def test_global_topk_has_no_per_card_floor_and_k_is_not_additional(self):
        for k in (2, 3, 4):
            candidates, plants = self.run_generator('student-topk', k)
            self.assertEqual(candidates, plants[:k] + [t.NOOP])

    def test_student_per_card_and_two_additional_distinct_actions(self):
        candidates, plants = self.run_generator('student-per-card')
        self.assertEqual(candidates, [plants[0], plants[3], t.NOOP, plants[1], plants[2]])

    def test_policy_anchor_is_retained_even_outside_global_rank_prefix(self):
        policy = dict(type='plant', packet=1, row=0, col=2)
        candidates, plants = self.run_generator('student-topk', policy=policy)
        self.assertEqual(candidates, plants[:2] + [t.NOOP, policy])

    def test_unknown_mode_fails_and_releases_root(self):
        args = SimpleNamespace(candidate_generator='typo', student_candidates=2, per_card=1,
                               min_gap=0, shovel_candidates=0, halving=True)
        env = Mock()
        with self.assertRaisesRegex(ValueError, 'unknown'):
            t._search(env, {'tick':60,'legal_actions':{'plants':[{'packet':0,'row':0,'col':0}]}},
                      t.NOOP, args, 17, {'searched':0})
        env.release_snapshot.assert_called_once()


class AblationStatisticsTest(unittest.TestCase):
    def test_paired_cost_ratio_noise_and_decision_family_use_whole_games(self):
        from static_value_ablation_stats import clustered
        game = {'points': 8, 'A_frozen': 6, 'A_expanded': 6, 'A_noise': 5,
                'A_cost': 80, 'A_reseed_cost': 96, 'A_reseed_frozen': 5, 'A_reseed_expanded': 5}
        for name in ('B2', 'B3', 'B4', 'C'):
            for suffix in ('', '_reseed'):
                game[name + suffix + '_frozen'] = game['A' + suffix + '_frozen']
                game[name + suffix + '_expanded'] = game['A' + suffix + '_expanded']
                game[name + suffix + '_cost'] = game['A' + suffix + '_cost'] / 2
            game[name + '_noise'] = 6
        result = clustered([game, game])
        self.assertEqual(result['games'], 2)
        self.assertEqual(result['points'], 16)
        comparison = result['paired_vs_A']['B2']
        self.assertEqual(comparison['_cost_ratio_to_A']['game_cluster_ci98_75'], [.5, .5])
        self.assertEqual(comparison['noise_agreement_minus_A']['estimate'], 1/8)
        self.assertTrue(comparison['recommendation_gate'])
        # A higher point estimate alone must not pass noninferiority or superiority.
        weaker = {**game, 'B2_frozen': 5}
        self.assertFalse(clustered([weaker, weaker])['paired_vs_A']['B2']['recommendation_gate'])

    def test_counts_compare_same_root_different_futures_and_both_references(self):
        from static_value_ablation import ARMS
        from static_value_ablation_stats import point_counts
        plant = dict(type='plant', packet=0, row=0, col=0)
        row = {'heavy_label': plant, 'policy':t.NOOP, 'actions':[plant,t.NOOP],
               'expanded':{'label':t.NOOP,'new_actions':[], 'new_continuations':0,
                           'verification_continuations':36}, 'arms':{}}
        for name in ARMS:
            for suffix in ('', '_noise'):
                row['arms'][name+suffix] = {'chosen':plant if not suffix else t.NOOP,
                    'candidates':[plant,t.NOOP], 'continuations':24,
                    'halving_continuations':6, 'verification_continuations':18}
        counts = point_counts(row)
        self.assertEqual(counts['A_frozen'], 1)
        self.assertEqual(counts['A_expanded'], 0)
        self.assertEqual(counts['A_reseed_expanded'], 1)
        self.assertEqual(counts['A_noise'], 0)
        self.assertEqual(counts['expanded_label_changes'], 1)


if __name__ == '__main__':
    unittest.main()
