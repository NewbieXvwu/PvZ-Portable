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
             patch.object(t, '_static_frames', side_effect=AssertionError('static branches forbidden')), \
             patch.object(t, '_values', side_effect=AssertionError('static scoring forbidden')), \
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


if __name__ == '__main__':
    unittest.main()
