"""Opt-in search knobs: --halving-first-samples and --margin-relative (task214b).

Both defaults must leave the historical teacher untouched: first round 1 future and the
absolute 0.02 override margin.  The tests pin the defaults and the opt-in arithmetic; they
do not start an engine.
"""
from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_play as sp

WAIT = dict(sp.NOOP)
PLANT = {"type": "plant", "packet": 0, "row": 0, "col": 0}
OTHER = {"type": "plant", "packet": 1, "row": 0, "col": 1}
THIRD = {"type": "plant", "packet": 2, "row": 0, "col": 2}


def _args(**overrides):
    base = dict(samples=3, horizon=2500, verify_samples=0, save_candidates=0,
                halving_first_samples=1, margin_relative=0.0)
    base.update(overrides)
    return SimpleNamespace(**base)


def _key(action):
    return json.dumps(action, sort_keys=True)


class MarginTest(unittest.TestCase):
    def test_default_is_the_exact_absolute_margin(self):
        self.assertEqual(sp._margin_threshold(SimpleNamespace(), 0.5), sp.MARGIN)
        self.assertEqual(sp._margin_threshold(SimpleNamespace(margin_relative=0.0), 0.5), sp.MARGIN)

    def test_relative_uses_floor_and_fraction(self):
        self.assertAlmostEqual(sp._margin_threshold(SimpleNamespace(margin_relative=0.1), 0.5), 0.05)
        self.assertAlmostEqual(sp._margin_threshold(SimpleNamespace(margin_relative=0.1), 0.001), sp.MARGIN_FLOOR)

    def test_default_override_still_fires_above_002(self):
        stats = {"overrides": 0, "override_kinds": Counter()}
        action = sp._verified_choice(None, {"tick": 60}, 7, [WAIT, PLANT], [0.40, 0.43],
                                     [3, 3], 1, 0, WAIT, _args(), 123, stats)
        self.assertEqual(action, PLANT)
        self.assertEqual(stats["overrides"], 1)

    def test_relative_margin_suppresses_the_same_003_gap_at_low_values(self):
        stats = {"overrides": 0, "override_kinds": Counter()}
        action = sp._verified_choice(None, {"tick": 60}, 7, [WAIT, PLANT], [0.40, 0.43],
                                     [3, 3], 1, 0, WAIT, _args(margin_relative=0.1), 123, stats)
        self.assertEqual(action, WAIT)
        self.assertEqual(stats["overrides"], 0)

    def test_relative_margin_still_fires_when_the_gap_is_large_enough(self):
        stats = {"overrides": 0, "override_kinds": Counter()}
        action = sp._verified_choice(None, {"tick": 60}, 7, [WAIT, PLANT], [0.10, 0.30],
                                     [3, 3], 1, 0, WAIT, _args(margin_relative=0.5), 123, stats)
        self.assertEqual(action, PLANT)
        self.assertEqual(stats["overrides"], 1)


class HalvingFirstSamplesTest(unittest.TestCase):
    def _lanes(self, first_samples, samples=3):
        candidates = [WAIT, PLANT, OTHER, THIRD]
        args = _args(samples=samples, halving_first_samples=first_samples)
        rounds = []

        def continuations(env, obs, root, jobs, args, stats):
            rounds.append([(candidate, seed - 7000) for candidate, seed in jobs])
            return [{"value": 1.0 - index * 0.1} for index, _ in enumerate(jobs)]

        stats = {}
        with patch.object(sp, "_continuations", side_effect=continuations), \
                patch.object(sp, "_verified_choice", return_value="chosen"):
            out = sp._halving_choice(None, {"tick": 0}, 1, candidates, WAIT, _key(WAIT),
                                     args, 7, stats)
        self.assertEqual(out, "chosen")
        return [len(round_jobs) for round_jobs in rounds], rounds

    def test_default_first_round_keeps_one_future_per_candidate(self):
        lanes, rounds = self._lanes(1)
        self.assertEqual(lanes, [4, 2, 2])
        self.assertEqual(len({_key(c) for c, _k in rounds[0]}), 4)
        self.assertEqual(sorted(k for _c, k in rounds[0]), [0, 0, 0, 0])

    def test_first_round_can_take_two_futures_before_the_first_cut(self):
        lanes, rounds = self._lanes(2)
        self.assertEqual(lanes, [8, 2])
        self.assertEqual(sorted(k for _c, k in rounds[0]), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_first_round_is_clamped_to_samples(self):
        lanes, rounds = self._lanes(9, samples=3)
        self.assertEqual(lanes, [12])
        self.assertEqual(sorted(k for _c, k in rounds[0]), [0] * 4 + [1] * 4 + [2] * 4)


class ParserDefaultsTest(unittest.TestCase):
    def test_new_flags_default_to_the_historical_behaviour(self):
        args = sp.parser().parse_args([
            "--rollout-policy", "student.pt", "--value", "value.pt",
            "--manifest", "manifest.json", "--task-id", "t", "--out", "out",
        ])
        self.assertEqual(args.halving_first_samples, 1)
        self.assertEqual(args.margin_relative, 0.0)


if __name__ == "__main__":
    unittest.main()
