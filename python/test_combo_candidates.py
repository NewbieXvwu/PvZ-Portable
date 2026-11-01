"""--combo-candidates: base now, upgrade later, scored as one plan (task214c)."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import search_play as sp

sp._W["fast"] = SimpleNamespace(obs="v1")   # the rollouts read the observation encoding here


def _args(**overrides):
    base = dict(samples=1, horizon=2500, verify_samples=0, save_candidates=0, halving=True,
                halving_first_samples=1, margin_relative=0.0, combo_candidates=0,
                combo_max_wait=2500, min_gap=0, per_card=1)
    base.update(overrides)
    return SimpleNamespace(**base)


def _packets(*types):
    return [{"index": index, "type": kind, "imitater_type": -1} for index, kind in enumerate(types)]


def _plants(*cells):
    return [{"packet": packet, "col": col, "row": row} for packet, col, row in cells]


class FakeEnv:
    def __init__(self):
        self.steps = []
        self.snapshots = 0
        self._last_response = {"decisions": 0, "simulation_seconds": 0.0, "network_seconds": 0.0}
        self.legal, self.ready, self.plants = [], 0, []
        self.tick = 100

    def snapshot_fast(self):
        self.snapshots += 1
        return self.snapshots

    def restore_fast(self, snapshot):
        self.steps.append(("restore", snapshot))

    def release_snapshot(self, snapshot):
        self.steps.append(("release", snapshot))

    def step(self, action):
        self.steps.append(("step", dict(action)))
        self.tick += 50
        obs = {"tick": self.tick, "legal_actions": {"plants": self.legal_for(self.tick)},
               "plants": self.plants, "terminal": False}
        return obs, 0.0, False, False, {"ok": True}

    def legal_for(self, tick):
        return [{"packet": packet, "col": col, "row": row} for packet, col, row in self.legal
                if tick >= self.ready]

    def rollout(self, first, until_tick, **kwargs):
        self.steps.append(("rollout", until_tick))
        self.tick = until_tick
        return {"tick": until_tick, "legal_actions": {"plants": self.legal_for(until_tick)},
                "plants": self.plants, "terminal": False, "value": 0.5}


class SearchCandidateTest(unittest.TestCase):
    def _search(self, combo, legal_actions, packets, static):
        args = _args(combo_candidates=combo)
        env = FakeEnv()
        env.static_scores = lambda root, actions, income: [static.get((a["packet"], a["col"], a["row"]), 0.1)
                                                           for a in actions]
        obs = {"tick": 0, "sun_income_rate": 1.0, "packets": packets,
               "legal_actions": {"plants": legal_actions}, "plants": []}
        stats = {"searched": 0, "overrides": 0, "override_kinds": {}, "last_search_tick": -10 ** 9,
                 "student_candidate_added": 0, "combo_candidate_added": 0}
        captured = {}

        def halving(env, obs, root, candidates, policy, key, args, seed, stats):
            captured["candidates"] = [dict(c) for c in candidates]
            return candidates[0]

        with patch.object(sp, "_halving_choice", side_effect=halving):
            sp._search(env, obs, sp.NOOP, args, 5, stats)
        return captured["candidates"], stats

    def test_off_by_default(self):
        candidates, stats = self._search(0, _plants((0, 3, 2)), _packets(1, 7, 40),
                                         {(0, 3, 2): 0.4})
        self.assertEqual(stats["combo_candidate_added"], 0)
        self.assertFalse(any("combo" in c for c in candidates))

    def test_marks_the_best_base_cell_for_an_unaffordable_upgrade(self):
        # packet 0 = repeater (base of 40), packet 1 = gatling (not legal yet)
        candidates, stats = self._search(1, _plants((0, 3, 2), (0, 4, 2)), _packets(7, 40),
                                         {(0, 3, 2): 0.2, (0, 4, 2): 0.9})
        combos = [c for c in candidates if "combo" in c]
        self.assertEqual(len(combos), 1)
        self.assertEqual((combos[0]["packet"], combos[0]["col"]), (0, 4))
        self.assertEqual(combos[0]["combo"], {"upgrade_packet": 1, "base_type": 7})
        self.assertEqual(stats["combo_candidate_added"], 1)

    def test_skips_pairs_whose_upgrade_is_already_legal(self):
        candidates, stats = self._search(1, _plants((0, 3, 2), (1, 4, 2)), _packets(7, 40),
                                         {(0, 3, 2): 0.2})
        self.assertEqual(stats["combo_candidate_added"], 0)

    def test_caps_the_number_of_plans(self):
        # two base/upgrade pairs: sunflower(1)->twinsunflower(41), repeater(7)->gatling(40)
        candidates, stats = self._search(1, _plants((0, 3, 2), (1, 5, 2)), _packets(1, 7, 40, 41),
                                         {(0, 3, 2): 0.2, (1, 5, 2): 0.3})
        self.assertEqual(stats["combo_candidate_added"], 1)


class ComboRolloutTest(unittest.TestCase):
    def test_waits_for_the_upgrade_and_plants_it_on_the_base_cell(self):
        env = FakeEnv()
        env.legal = [(7, 4, 2)]
        env.ready = 250
        env.plants = []
        candidate = {"type": "plant", "packet": 0, "col": 4, "row": 2,
                     "combo": {"upgrade_packet": 7, "base_type": 1}}
        args = _args(combo_max_wait=500)
        leaf = sp._combo_rollout(env, {"tick": 100}, candidate, 2600, 1, 9, args)
        self.assertEqual(leaf["value"], 0.5)
        steps = [entry for entry in env.steps if entry[0] == "step"]
        self.assertEqual(steps[0][1], {"type": "plant", "packet": 0, "col": 4, "row": 2})
        self.assertTrue(all(step[1]["type"] == "wait" for step in steps[1:-1]))
        self.assertEqual(steps[-1][1], {"type": "plant", "packet": 7, "col": 4, "row": 2})

    def test_scores_the_base_only_when_the_upgrade_never_comes(self):
        env = FakeEnv()
        env.legal = []
        env.ready = 0
        env.plants = []
        candidate = {"type": "plant", "packet": 0, "col": 4, "row": 2,
                     "combo": {"upgrade_packet": 7, "base_type": 1}}
        sp._combo_rollout(env, {"tick": 100}, candidate, 2600, 1, 9, _args(combo_max_wait=100))
        plants = [entry[1] for entry in env.steps if entry[0] == "step" and entry[1]["type"] == "plant"]
        self.assertEqual(plants, [{"type": "plant", "packet": 0, "col": 4, "row": 2}])


class ForcedUpgradeTest(unittest.TestCase):
    def _obs(self, plants, legal, tick=100):
        return {"tick": tick, "plants": plants,
                "legal_actions": {"plants": [{"packet": p, "col": c, "row": r} for p, c, r in legal]}}

    def _stats(self, expires=2000):
        return {"pending_upgrade": {"packet": 40, "base_type": 7, "col": 4, "row": 2,
                                    "expires": expires}}

    def test_fires_on_the_base_cell_only(self):
        stats = self._stats()
        obs = self._obs([{"type": 7, "col": 4, "row": 2}], [(40, 3, 2), (40, 4, 2)])
        self.assertEqual(sp._forced_upgrade(obs, stats, 2500),
                         {"type": "plant", "packet": 40, "col": 4, "row": 2})
        self.assertIsNone(stats["pending_upgrade"])

    def test_waits_while_not_affordable(self):
        self.assertIsNone(sp._forced_upgrade(
            self._obs([{"type": 7, "col": 4, "row": 2}], []), self._stats(), 2500))

    def test_drops_the_plan_when_the_base_died(self):
        stats = self._stats()
        self.assertIsNone(sp._forced_upgrade(self._obs([], [(40, 4, 2)]), stats, 2500))
        self.assertIsNone(stats["pending_upgrade"])

    def test_drops_the_plan_after_the_deadline(self):
        stats = self._stats(expires=150)
        self.assertIsNone(sp._forced_upgrade(
            self._obs([{"type": 7, "col": 4, "row": 2}], [(40, 4, 2)], tick=200), stats, 2500))
        self.assertIsNone(stats["pending_upgrade"])


class ParserTest(unittest.TestCase):
    def test_defaults_are_off(self):
        args = sp.parser().parse_args([
            "--rollout-policy", "student.pt", "--value", "value.pt",
            "--manifest", "manifest.json", "--task-id", "t", "--out", "out",
        ])
        self.assertEqual((args.combo_candidates, args.combo_max_wait), (0, 5000))


if __name__ == "__main__":
    unittest.main()
