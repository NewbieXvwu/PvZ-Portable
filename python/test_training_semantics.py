"""Regression tests for discounted values, time-based GAE, seed isolation, and search-value semantics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pvz_search_value import SEARCH_VALUE_FEATURES, SearchValueModel, search_value_features
from pvz_seed_sets import read_seed_set
from pvz_value import VALUE_GAMMA
from pvz_training_artifacts import _jsonable_cli_value
from train_pvz_agent import _search_value_metadata, validate_seed_sets
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

    def test_time_based_lambda_matches_reference_ticks(self) -> None:
        episodes = [{"transitions": [
            {"action_duration_ticks": 300, "reward": 0.0, "value": 0.0},
            {"action_duration_ticks": 300, "reward": 1.0, "value": 0.0},
        ]}]
        add_advantages(episodes, gae_lambda=0.5)
        transitions = episodes[0]["transitions"]
        self.assertAlmostEqual(transitions[1]["advantage"], VALUE_GAMMA)
        self.assertAlmostEqual(transitions[0]["advantage"], VALUE_GAMMA * 0.5 * VALUE_GAMMA)

    def test_all_seed_roles_must_be_unique_and_pairwise_disjoint(self) -> None:
        validate_seed_sets(train=[1, 2], dagger=[3], value_bootstrap=[4], value_refinement=[5],
                           development=[6, 7], final_test=[8, 9])
        with self.assertRaises(ValueError):
            validate_seed_sets(train=[1, 1], dagger=[2])
        with self.assertRaises(ValueError):
            validate_seed_sets(train=[1], dagger=[2], development=[1])
        with self.assertRaises(ValueError):
            validate_seed_sets(value_bootstrap=[4], value_refinement=[4])

    def test_frozen_seed_role_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.json"
            path.write_text(
                '{"schema_version":1,"level":7,"playthrough":2,"first_seed":30,"count":4,"role":"development"}',
                encoding="utf-8",
            )
            self.assertEqual(read_seed_set(path, 7, "development"), [30, 31, 32, 33])
            with self.assertRaises(ValueError):
                read_seed_set(path, 7, "final_test")

    def test_provenance_path_values_are_jsonable(self) -> None:
        self.assertEqual(_jsonable_cli_value(Path("/tmp/seeds.json")), "/tmp/seeds.json")

    def test_search_value_summary_drops_state_dict(self) -> None:
        metadata = _search_value_metadata({"state_dict": {"x": 1}, "bootstrap_seeds": [1], "loss": 0.1})
        self.assertNotIn("state_dict", metadata)
        self.assertEqual(metadata["bootstrap_seeds"], [1])

    def test_search_value_features_have_fixed_width_and_model_is_independent(self) -> None:
        observation = {
            "sun": 50,
            "wave": 1,
            "wave_count": 10,
            "tick": 100,
            "zombie_count_multiplier": 1.0,
            "night": False,
            "pool": False,
            "fog": False,
            "roof": False,
            "plants": [],
            "zombies": [],
            "projectiles": [],
            "packets": [],
            "cells": [{"row": row, "row_type": 1} for row in range(5)],
            "defenses": [],
        }
        features = search_value_features(observation)
        self.assertEqual(tuple(features.shape), (SEARCH_VALUE_FEATURES,))
        model = SearchValueModel()
        value = model(features.unsqueeze(0)).item()
        self.assertGreaterEqual(value, -1.0)
        self.assertLessEqual(value, 1.0)
        self.assertFalse(any("GameplayModel" in type(module).__name__ for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
