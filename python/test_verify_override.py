import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import search_play

WAIT = dict(search_play.NOOP)
PLANT = {"type": "plant", "packet": 0, "row": 0, "col": 0}
OTHER = {**PLANT, "packet": 1}


class VerifyOverrideTest(unittest.TestCase):
    def choose(self, samples, returns, means=None):
        stats = {"overrides": 0, "override_kinds": Counter()}
        args = SimpleNamespace(samples=3, horizon=2500, margin=.02, verify_samples=samples)
        means = [.4, .5, .9] if means is None else means
        with patch.object(search_play, "_rollout_policy", side_effect=returns) as rollout:
            action = search_play._verified_choice(None, {"tick": 60}, 7, [WAIT, PLANT, OTHER],
                                                   means, [3, 3, 1], 1, 0, WAIT, args, 123, stats)
        return action, stats, rollout.call_args_list

    def test_disabled_preserves_original_choice_and_values(self):
        action, stats, calls = self.choose(0, [])
        self.assertEqual(action, PLANT)
        self.assertFalse(calls)
        self.assertEqual(stats["last_targets"], [(WAIT, .4), (PLANT, .5), (OTHER, .9)])
        self.assertNotIn("last_verification", stats)

    def test_new_independent_futures_cancel_override(self):
        action, stats, calls = self.choose(3, [{"rollout_result": .8}] * 3 + [{"rollout_result": .2}] * 3)
        self.assertEqual(action, WAIT)
        self.assertEqual(stats["overrides"], 0)
        self.assertEqual([call.args[-1] for call in calls], [123003, 123004, 123005] * 2)
        self.assertEqual([call.args[2] for call in calls], [WAIT] * 3 + [PLANT] * 3)
        self.assertAlmostEqual(stats["last_targets"][0][1], .6)
        self.assertAlmostEqual(stats["last_targets"][1][1], .35)
        self.assertEqual(stats["last_targets"][2][1], .9)
        self.assertTrue(stats["last_verification"]["before_override"])
        self.assertFalse(stats["last_verification"]["after_override"])

    def test_combines_old_and_new_samples_before_margin(self):
        action, stats, _calls = self.choose(3, [{"rollout_result": .5}] * 3 + [{"rollout_result": .48}] * 3)
        self.assertEqual(action, PLANT)
        self.assertEqual(stats["overrides"], 1)
        self.assertAlmostEqual(stats["last_targets"][1][1] - stats["last_targets"][0][1], .04)

    def test_non_override_runs_no_extra_futures(self):
        action, stats, calls = self.choose(9, [], [.4, .41, .9])
        self.assertEqual(action, WAIT)
        self.assertFalse(calls)


if __name__ == "__main__":
    unittest.main()
