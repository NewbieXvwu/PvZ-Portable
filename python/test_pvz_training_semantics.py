"""Regression tests for discounted values, time-based GAE, and seed isolation."""

from __future__ import annotations

import unittest

from pvz_agent import VALUE_GAMMA
from train_pvz_agent import validate_seed_sets
from train_pvz_ppo import add_advantages


class TrainingSemanticsTests(unittest.TestCase):
    def test_zero_tick_transition_does_not_decay_trace(self) -> None:
        episodes = [{"transitions": [
            {"action_duration_ticks": 0, "reward": 0.0, "value": 0.0},
            {"action_duration_ticks": 300, "reward": 1.0, "value": 0.0},
        ]}]
        add_advantages(episodes, gae_lambda=0.5)
        transitions = episodes[0]["transitions"]
        self.assertAlmostEqual(transitions[1]["advantage"], VALUE_GAMMA)
        self.assertAlmostEqual(transitions[0]["advantage"], VALUE_GAMMA)

    def test_trace_lambda_is_scaled_by_elapsed_time(self) -> None:
        episodes = [{"transitions": [
            {"action_duration_ticks": 300, "reward": 0.0, "value": 0.0},
            {"action_duration_ticks": 300, "reward": 1.0, "value": 0.0},
        ]}]
        add_advantages(episodes, gae_lambda=0.5)
        transitions = episodes[0]["transitions"]
        self.assertAlmostEqual(transitions[1]["advantage"], VALUE_GAMMA)
        self.assertAlmostEqual(transitions[0]["advantage"], VALUE_GAMMA * 0.5 * VALUE_GAMMA)

    def test_seed_sets_must_be_unique_and_disjoint(self) -> None:
        validate_seed_sets([0, 1], [10, 11], [30, 31])
        with self.assertRaises(ValueError):
            validate_seed_sets([0, 0], [10], [30])
        with self.assertRaises(ValueError):
            validate_seed_sets([0, 1], [1, 2], [30])
        with self.assertRaises(ValueError):
            validate_seed_sets([0, 1], [10, 11], [11, 30])


if __name__ == "__main__":
    unittest.main()
