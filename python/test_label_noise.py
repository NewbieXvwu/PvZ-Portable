import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import label_noise
import search_play


WAIT = {"type": "wait", "ticks": 60, "until": "timeout"}
PLANT = {"type": "plant", "packet": 0, "row": 0, "col": 0}
OTHER = {"type": "plant", "packet": 1, "row": 0, "col": 0}


def arm(label, values, best=None):
    return {"label": label, "best": best or label,
            "values": [(json.dumps(action, sort_keys=True), value) for action, value in values]}


def row(student, a_label, b_label, h_label, wave=5):
    return {"seed": 1, "decision": 0, "wave": wave, "student": student, "margin": 0.02,
            "arms": {"A": arm(a_label, [(WAIT, .4), (PLANT, .46), (OTHER, .9)], PLANT),
                     "B": arm(b_label, [(WAIT, .4), (PLANT, .46), (OTHER, .3)]),
                     "H": arm(h_label, [(WAIT, .4), (PLANT, .5), (OTHER, .55)])}}


class LabelNoiseTest(unittest.TestCase):
    def test_conditional_counts_and_value_losses(self):
        rows = [row(WAIT, PLANT, PLANT, PLANT), row(WAIT, PLANT, WAIT, OTHER),
                row(WAIT, PLANT, OTHER, WAIT), row(PLANT, PLANT, WAIT, WAIT, 20)]
        result = label_noise.summarize(rows)
        group = result["groups"]["all"]
        self.assertEqual(group["a_override"]["rate"], .75)
        self.assertEqual(group["b_override_given_a"]["count"], 2)
        self.assertEqual(group["same_action_given_both"]["rate"], .5)
        for key in ("h_same_given_a", "h_different_given_a", "h_no_override_given_a"):
            self.assertEqual(group[key]["rate"], 1 / 3)
        self.assertEqual(group["h_override_given_not_a"]["rate"], 1)
        self.assertAlmostEqual(group["a_value_loss"]["mean"], .05)
        self.assertAlmostEqual(group["student_value_loss"]["mean"], .125)
        self.assertAlmostEqual(group["gain_h_agrees"]["mean"], .1)
        self.assertEqual(result["groups"]["plant/20-30"]["searches"], 1)
        self.assertIsNone(result["groups"]["10-19"]["a_override"]["rate"])

    def test_margin_uses_surviving_best_and_strict_threshold(self):
        record = row(WAIT, PLANT, WAIT, PLANT)
        self.assertEqual(label_noise.relabel(record, "A", .05), label_noise.action_key(PLANT))
        self.assertEqual(label_noise.relabel(record, "A", .1), "wait")
        record["arms"]["A"]["values"] = [(json.dumps(WAIT), 0), (json.dumps(PLANT), .02)]
        self.assertEqual(label_noise.relabel(record, "A", .02), "wait")
        self.assertEqual(label_noise.action_key({"type": "wait", "ticks": 60}), "wait")

    def test_search_records_actual_halving_winner(self):
        candidates = [WAIT, PLANT, OTHER, {**OTHER, "packet": 2}, {**OTHER, "packet": 3}]
        args = SimpleNamespace(samples=3, horizon=2500, margin=.02)
        stats = {"overrides": 0, "override_kinds": Counter()}
        outcomes = {(0, 0): .36, (0, 1): .36, (0, 2): .36,
                    (1, 0): .99, (1, 1): 0, (1, 2): 0,
                    (2, 0): .9, (2, 1): 0, (2, 2): 0,
                    (3, 0): .8, (3, 1): 0, (4, 0): .1}

        def rollout(env, obs, action, until, root, seed):
            return {"rollout_result": outcomes[(candidates.index(action), seed % 1000)]}

        with patch.object(search_play, "_rollout_policy", side_effect=rollout):
            chosen = search_play._halving_choice(None, {"tick": 0}, 1, candidates, WAIT,
                                                 json.dumps(WAIT, sort_keys=True), args, 0, stats)
        self.assertEqual(chosen, WAIT)
        self.assertEqual(stats["last_best"], WAIT)
        self.assertGreater(dict((label_noise.action_key(action), value) for action, value in stats["last_targets"])
                           [label_noise.action_key(candidates[3])], .36)


if __name__ == "__main__":
    unittest.main()
