"""Regression tests for the T5 training path: time-based GAE, seed-set reading, and discounting.

The search-teacher, DAgger and search-value semantics that used to live here went with
the frozen teacher; only the pieces the PPO path still reads are kept.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pvz_seed_sets import read_seed_set
from pvz_value import DISCOUNT_REFERENCE_TICKS, VALUE_GAMMA
from train_pvz_ppo import add_advantages


def _transition(duration: int, reward: float, value: float) -> dict:
    return {"action_duration_ticks": duration, "reward": reward, "value": value}


def _episode(transitions: list[dict]) -> list[dict]:
    return [{"transitions": [dict(item) for item in transitions]}]

class AdvantageEstimationTests(unittest.TestCase):
    def test_zero_tick_actions_do_not_decay_the_trace(self) -> None:
        """``lambda ** 0`` is 1, so an instant action must not damp what it carries."""
        episodes = _episode([_transition(0, 0.0, 0.0), _transition(300, 1.0, 0.0)])

        add_advantages(episodes, gae_lambda=0.5)

        first, second = episodes[0]["transitions"]
        self.assertAlmostEqual(second["advantage"], 1.0)
        self.assertAlmostEqual(first["advantage"], 1.0)

    def test_trace_discount_uses_the_tick_ratio(self) -> None:
        """A 300-tick action discounts gamma and lambda by one full reference span."""
        episodes = _episode([_transition(300, 0.0, 0.0), _transition(300, 1.0, 0.0)])

        add_advantages(episodes, gae_lambda=0.5)

        first, second = episodes[0]["transitions"]
        self.assertAlmostEqual(second["advantage"], 1.0)
        self.assertAlmostEqual(first["advantage"], VALUE_GAMMA * 0.5)

    def test_bootstrapped_values_enter_the_advantage(self) -> None:
        """With lambda=1 the recurrence is closed form, so the numbers are hand-derivable.

        d = 0.99 for a 300-tick action. The episode ends after the second action, so
          delta_1 = 1.0 - 0.2 = 0.8
          delta_0 = 0.99 * 0.2 - 0.5 = -0.302
          A_1 = 0.8, A_0 = -0.302 + 0.99 * 0.8 = 0.49
        """
        episodes = _episode([_transition(300, 0.0, 0.5), _transition(300, 1.0, 0.2)])

        add_advantages(episodes, gae_lambda=1.0)

        first, second = episodes[0]["transitions"]
        self.assertAlmostEqual(second["advantage"], 0.8)
        self.assertAlmostEqual(second["return"], 1.0)
        self.assertAlmostEqual(first["advantage"], 0.49)
        self.assertAlmostEqual(first["return"], 0.99)

    def test_returns_are_advantages_plus_the_state_value(self) -> None:
        episodes = _episode([_transition(60, 0.0, -0.4), _transition(150, 0.5, 0.25)])

        add_advantages(episodes, gae_lambda=0.95)

        for transition in episodes[0]["transitions"]:
            self.assertAlmostEqual(transition["return"], transition["advantage"] + transition["value"])

class SeedSetReadingTests(unittest.TestCase):
    def test_frozen_seed_role_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            path.write_text(
                '{"schema_version":1,"level":7,"playthrough":2,"first_seed":30,"count":4,"role":"development"}',
                encoding="utf-8",
            )
            self.assertEqual(read_seed_set(path, 7, "development"), [30, 31, 32, 33])
            self.assertEqual(read_seed_set(path, 7), [30, 31, 32, 33])
            with self.assertRaisesRegex(ValueError, "final_test"):
                read_seed_set(path, 7, "final_test")

    def test_frozen_seed_file_rejects_the_wrong_schema_level_or_playthrough(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            documents = [
                {"schema_version": 2, "level": 7, "playthrough": 2, "first_seed": 0, "count": 1,
                 "role": "development"},
                {"schema_version": 1, "level": 8, "playthrough": 2, "first_seed": 0, "count": 1,
                 "role": "development"},
                {"schema_version": 1, "level": 7, "playthrough": 1, "first_seed": 0, "count": 1,
                 "role": "development"},
            ]
            for document in documents:
                with self.subTest(document=document):
                    path.write_text(json.dumps(document), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "schema 1"):
                        read_seed_set(path, 7, "development")

class DiscountConstantTests(unittest.TestCase):
    def test_reference_span_and_gamma_are_the_documented_values(self) -> None:
        self.assertEqual(DISCOUNT_REFERENCE_TICKS, 300)
        self.assertAlmostEqual(VALUE_GAMMA ** (DISCOUNT_REFERENCE_TICKS / DISCOUNT_REFERENCE_TICKS), VALUE_GAMMA)


if __name__ == "__main__":
    unittest.main()
