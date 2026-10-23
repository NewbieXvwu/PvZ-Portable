"""task42 comparison arm: exact cancellation, pool symmetry, dispatch and stats gates."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import b3_noninferiority as b3
import b3_noninferiority_stats as stats
import search_play

WAIT = {"type": "wait", "ticks": 60, "until": "timeout"}
PLANT = {"type": "plant", "packet": 0, "row": 0, "col": 0}


class CancellationTest(unittest.TestCase):
    def test_identical_choices_cancel_without_a_heavy_label(self):
        self.assertEqual(b3.correctness_difference(PLANT, dict(PLANT)), 0)
        self.assertEqual(b3.correctness_difference(WAIT, dict(WAIT)), 0)

    def test_both_wrong_is_zero_and_missing_heavy_is_an_error(self):
        other = {"type": "plant", "packet": 0, "row": 0, "col": 1}
        with self.assertRaisesRegex(ValueError, 'requires'):
            b3.correctness_difference(PLANT, other)
        self.assertEqual(b3.correctness_difference(PLANT, other, WAIT), 0)
        self.assertEqual(b3.correctness_difference(PLANT, other, PLANT), -1)
        self.assertEqual(b3.correctness_difference(PLANT, other, other), 1)


class PoolTest(unittest.TestCase):
    def test_pool_keeps_both_arms_top2_per_card_and_is_model_order_independent(self):
        cells = [{"type": "plant", "packet": 0, "row": 0, "col": c} for c in range(4)]
        cells += [{"type": "plant", "packet": 1, "row": 0, "col": 0}]
        a = {"candidates": [cells[0], WAIT]}
        b = {"candidates": [cells[1], WAIT]}
        scored = [(1.0 - i * .1, cell) for i, cell in enumerate(cells)]
        pool = b3.pool_actions(scored, {"A": a, "B3": b}, cells, 7)
        keys = [b3.key(x) for x in pool]
        self.assertEqual(keys, sorted(keys))
        for cell in (cells[0], cells[1], cells[4]):
            self.assertIn(b3.key(cell), keys)
        self.assertIn(b3.key(WAIT), keys)
        reverse = b3.pool_actions(list(reversed(scored)), {"B3": b, "A": a}, cells, 7)
        self.assertEqual([b3.key(x) for x in reverse], keys)


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.legal = [{"packet": 0, "row": 0, "col": 0}, {"packet": 0, "row": 0, "col": 1}]
        self.obs = {"tick": 100, "legal_actions": {"plants": self.legal}, "sun_income_rate": 2.0}
        self.args = SimpleNamespace(min_gap=300, per_card=1, student_candidates=0,
                                    halving=True, samples=3, horizon=2500, verify_samples=0)

    def _run(self, mode):
        captured = []
        with patch.dict(search_play._W, {"candidate_mode": mode, "student_frame": {}, "fast": object()},
                        clear=True), \
             patch.object(search_play, "_student_topk_candidates",
                          return_value=([PLANT], {b3.key(PLANT)})), \
             patch.object(search_play, "_static_scores", return_value=[]), \
             patch.object(search_play, "_halving_choice",
                          side_effect=lambda env, obs, root, candidates, *a: captured.extend(candidates) or WAIT):
            env = SimpleNamespace(snapshot_fast=lambda: 1, restore_fast=lambda root: None,
                                  release_snapshot=lambda root: None)
            search_play._search(env, self.obs, WAIT, self.args, 300000000,
                                {"searched": 0, "overrides": 0, "override_kinds": {}})
        return captured

    def test_student_topk_mode_never_touches_the_static_path(self):
        candidates = self._run('student-topk')
        self.assertEqual(candidates, [PLANT, search_play.NOOP])

    def test_default_mode_is_the_formal_static_path(self):
        captured = self._run('static')
        self.assertEqual(captured, [search_play.NOOP])


class StatisticsTest(unittest.TestCase):
    @staticmethod
    def game(delta, points=8):
        return {'points': points, 'delta': delta, 'identical': 0, 'judged': points,
                'only_A_correct': 0, 'only_B3_correct': 0, 'disagreed_both_wrong': 0,
                'A_cost': 100, 'B3_cost': 80, 'A_cpu': 10, 'B3_cpu': 6,
                'A_worker_cpu': 2, 'B3_worker_cpu': 1, 'A_sim_cpu': 8, 'B3_sim_cpu': 5,
                'A_halving': 82, 'B3_halving': 80, 'A_verify': 18, 'B3_verify': 0,
                'A_candidates': 6, 'B3_candidates': 4,
                'truth_continuations': 0, 'heavy_verification_continuations': 0}

    def test_uncertainty_is_clustered_by_game_and_both_intervals_gate(self):
        games = [self.game(-1 if i % 4 == 0 else 0) for i in range(64)]
        result = stats.summarize(games)
        self.assertEqual(result['primary']['df'], 63)
        self.assertAlmostEqual(result['primary']['difference_B3_minus_A'], -.03125)
        self.assertFalse(result['primary']['robust_NI'])
        roomy = stats.summarize([self.game(0)] * 64)
        self.assertTrue(roomy['primary']['robust_NI'])
        self.assertAlmostEqual(roomy['metrics']['B3_cost_saving']['estimate'], .2)
        self.assertAlmostEqual(roomy['metrics']['B3_cpu_saving']['estimate'], .4)

    def test_one_family_below_the_margin_fails_the_combined_judgement(self):
        games = [self.game(-1 if i < 12 else 0) for i in range(64)]
        result = stats.summarize(games)
        self.assertLess(result['primary']['ci95'][0], -.03)
        self.assertFalse(result['primary']['robust_NI'])


if __name__ == '__main__':
    unittest.main()
