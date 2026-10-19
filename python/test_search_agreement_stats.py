import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from search_agreement_stats import agreement_intervals, wilson
import sim_skip_agreement as diagnostic
import search_play

WAIT = {"type": "wait", "ticks": 60, "until": "timeout"}
PLANT = {"type": "plant", "packet": 0, "row": 0, "col": 0}


class AgreementStatsTest(unittest.TestCase):
    def test_whole_game_bootstrap_does_not_count_repeated_points_as_new_games(self):
        rows = [{"seed": s, "exact": PLANT, "approx": PLANT,
                 "exact_other_futures": PLANT if s < 5 else WAIT}
                for s in range(10) for _ in range(10)]
        first = agreement_intervals(rows)
        repeated = agreement_intervals(rows * 100)
        self.assertEqual(first["difference"], .5)
        self.assertAlmostEqual(first["difference_cluster_ci95"][0], .2)
        self.assertAlmostEqual(first["difference_cluster_ci95"][1], .8)
        self.assertEqual(first["difference_cluster_ci95"], repeated["difference_cluster_ci95"])
        self.assertEqual(first["games"], 10)
        self.assertTrue(first["gate_passed"])

    def test_clear_loss_and_too_few_independent_games_do_not_pass(self):
        rows = [{"seed": s, "exact": PLANT, "approx": WAIT, "exact_other_futures": PLANT}
                for s in range(8)]
        result = agreement_intervals(rows)
        self.assertEqual(result["difference_cluster_ci95"], [-1., -1.])
        self.assertFalse(result["gate_passed"])
        one = [{"seed": 1, "exact": WAIT, "approx": WAIT, "exact_other_futures": WAIT}] * 10000
        self.assertFalse(agreement_intervals(one)["gate_passed"])

    def test_wilson_and_empty_input(self):
        lower, upper = wilson(5, 10)
        self.assertAlmostEqual(lower, .236593090512564, places=10)
        self.assertAlmostEqual(upper, .763406909487436, places=10)
        with self.assertRaises(ValueError):
            agreement_intervals([])


class FormalTeacherDiagnosticTest(unittest.TestCase):
    def test_all_three_arms_use_k2_verify9_and_same_root_with_independent_noise_futures(self):
        args = argparse.Namespace(samples=3, horizon=2500, margin=.02, per_card=1, base_halving=True,
            student_candidates=2, search_verify_samples=9, approx_per_card=None, approx_samples=None,
            approx_horizon=1800, approx_halving=False, label_noise=False, high_samples=12, dagger_beta=0,
            jsonl_dir=None, approx_static_value=None, max_actions=1, min_gap=300, cuda_rollouts=False,
            base_mask=61, mask=61, base_cpp_rollouts=True, cpp_rollouts=True)
        obs = {"tick": 0, "terminal": False, "legal_actions": {"plants": [PLANT]}}
        env = Mock()
        env.step.return_value = ({"terminal": True}, 0, True, False, {"ok": True})
        seen = []
        def search(environment, current, policy, config, seed, stats):
            self.assertIs(current, obs)
            self.assertIs(environment, env)
            seen.append((config.horizon, config.student_candidates, config.verify_samples, seed))
            stats.update(searched=1, last_targets=[(PLANT, .5)], last_best=PLANT,
                         last_verification={"samples": 9}, student_candidate_added=2)
            return WAIT
        with patch.dict(diagnostic._W, {"args": args, "env": env}, clear=True), \
                patch.dict(search_play._W, {}, clear=True), patch.object(diagnostic, "_reset", return_value=obs), \
                patch.object(search_play, "_fast_action", return_value=PLANT), \
                patch.object(search_play, "_search", side_effect=search):
            rows = diagnostic._search_decisions(30000)
        self.assertEqual(seen, [(2500, 2, 9, 300000000), (1800, 2, 9, 300000000),
                                (2500, 2, 9, 307777777)])
        self.assertEqual(len(rows), 1)
        self.assertTrue(all(a["verification_samples_used"] == 9 for a in rows[0]["teacher_audit"].values()))
        # Agreement is measured on student trajectories, regardless of the teachers' returned label.
        env.step.assert_called_once_with(PLANT)


if __name__ == "__main__":
    unittest.main()
