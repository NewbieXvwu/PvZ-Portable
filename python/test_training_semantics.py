"""Regression tests for discounted values, time-based GAE, seed isolation, and search-value semantics."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from pvz_search_diagnostics import compare_label_samples
from pvz_search_value import (
    PLANT_TYPE_COUNT,
    SEARCH_VALUE_FEATURE_VERSION,
    SEARCH_VALUE_FEATURES,
    SEARCH_VALUE_VERSION,
    SearchValueModel,
    load_search_value,
    save_search_value,
    search_value_features,
)
from pvz_seed_sets import read_seed_set
from pvz_training_artifacts import _jsonable_cli_value, checkpoint_metadata, task_signature
from pvz_value import DISCOUNT_REFERENCE_TICKS, VALUE_GAMMA
from train_pvz_agent import validate_seed_sets
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


class SeedIsolationTests(unittest.TestCase):
    def test_duplicate_seeds_within_a_role_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "train seed set contains duplicates"):
            validate_seed_sets(train=[1, 1], dagger=[2])

    def test_seed_roles_must_be_pairwise_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "train and development seed sets overlap"):
            validate_seed_sets(train=[1, 2], development=[2, 3])
        with self.assertRaisesRegex(ValueError, "value_bootstrap and value_refinement seed sets overlap"):
            validate_seed_sets(value_bootstrap=[4], value_refinement=[4])

    def test_disjoint_seed_roles_are_accepted(self) -> None:
        validate_seed_sets(train=[1, 2], dagger=[3], value_bootstrap=[4], value_refinement=[5],
                           development=[6, 7], final_test=[8, 9])

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


class ProvenanceHelperTests(unittest.TestCase):
    def test_cli_values_are_made_jsonable(self) -> None:
        values = {"path": Path("/tmp/seeds.json"), "deck": (0, 1, 2), "device": "cpu",
                  "epochs": 8, "ratio": 1.0, "flag": True, "missing": None}

        converted = {key: _jsonable_cli_value(value) for key, value in values.items()}

        self.assertEqual(converted["path"], "/tmp/seeds.json")
        self.assertEqual(converted["deck"], [0, 1, 2])
        for key in ("device", "epochs", "ratio", "flag", "missing"):
            self.assertEqual(converted[key], values[key])
        # The contract is that the converted mapping survives JSON encoding.
        self.assertEqual(json.loads(json.dumps(converted))["deck"], [0, 1, 2])

    def test_checkpoint_metadata_drops_only_the_tensors(self) -> None:
        checkpoint = {"state_dict": {"x": 1}, "bootstrap_seeds": [1], "loss": 0.1}
        untouched = dict(checkpoint)

        metadata = checkpoint_metadata(checkpoint)

        self.assertEqual(metadata, {"bootstrap_seeds": [1], "loss": 0.1})
        self.assertEqual(checkpoint, untouched, "checkpoint_metadata must not mutate its input")

    def test_task_signature_covers_the_resource_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "properties").mkdir()
            (root / "main.pak").write_bytes(b"main")
            (root / "properties" / "partner.xml").write_bytes(b"partner")

            signature = task_signature(7, [0, 1, 2], 1.0, root)

            self.assertEqual(signature["level"], 7)
            self.assertEqual(signature["deck"], [0, 1, 2])
            self.assertEqual(signature["playthrough"], 2)
            self.assertEqual(set(signature["resource_sha256"]), {"main.pak", "properties/partner.xml"})

            (root / "main.pak").write_bytes(b"changed")
            self.assertNotEqual(task_signature(7, [0, 1, 2], 1.0, root), signature)


def _observation(**overrides: object) -> dict:
    observation = {
        "sun": 50, "wave": 1, "wave_count": 10, "tick": 100,
        "zombie_count_multiplier": 1.0,
        "night": False, "pool": False, "fog": False, "roof": False,
        "plants": [], "zombies": [], "projectiles": [], "packets": [],
        "cells": [{"row": row, "row_type": 1} for row in range(5)],
        "defenses": [],
    }
    observation.update(overrides)
    return observation


def _plant(type_id: int) -> dict:
    return {"type": type_id, "row": 1, "col": 2, "health": 100, "max_health": 100}


class SearchValueFeatureTests(unittest.TestCase):
    def test_feature_vector_has_the_declared_fixed_width(self) -> None:
        features = search_value_features(_observation())

        self.assertEqual(tuple(features.shape), (SEARCH_VALUE_FEATURES,))
        self.assertEqual(features.dtype, torch.float32)

    def test_model_input_width_matches_the_feature_vector(self) -> None:
        model = SearchValueModel()
        first_layer = next(model.network.children())

        self.assertEqual(first_layer.in_features, SEARCH_VALUE_FEATURES)
        with torch.no_grad():
            self.assertEqual(tuple(model(search_value_features(_observation()).unsqueeze(0)).shape), (1,))

    def test_value_model_output_depends_on_its_input(self) -> None:
        """A model that always returned the same number would pass a naive range check."""
        torch.manual_seed(0)
        model = SearchValueModel()
        empty = search_value_features(_observation())
        crowded = search_value_features(_observation(
            plants=[_plant(0)],
            zombies=[{"type": 0, "row": 2, "x": 200.0, "body_health": 200,
                      "helm_health": 0, "shield_health": 0}],
        ))

        with torch.no_grad():
            self.assertNotEqual(float(model(empty.unsqueeze(0))), float(model(crowded.unsqueeze(0))))

    def test_unmodelled_type_ids_are_visible_and_share_one_slot(self) -> None:
        """An unmodelled type must not be indistinguishable from "nothing here"."""
        empty = search_value_features(_observation())
        unknown = search_value_features(_observation(plants=[_plant(PLANT_TYPE_COUNT + 7)]))
        also_unknown = search_value_features(_observation(plants=[_plant(PLANT_TYPE_COUNT + 900)]))
        known = search_value_features(_observation(plants=[_plant(3)]))
        first_known_slot = search_value_features(_observation(plants=[_plant(0)]))

        self.assertFalse(torch.equal(empty, unknown))
        # Two out-of-range ids share the dedicated unknown slot...
        self.assertTrue(torch.equal(unknown, also_unknown))
        # ...but that slot must not be a known type's slot, or "unmodelled" collapses
        # back into "looks like plant #0".
        self.assertFalse(torch.equal(unknown, first_known_slot))
        self.assertFalse(torch.equal(known, unknown))

    def test_plant_zombie_and_packet_types_are_distinguished(self) -> None:
        observation = _observation(
            plants=[{"type": 0, "row": 1, "col": 2, "health": 100, "max_health": 100}],
            zombies=[{"type": 0, "row": 2, "x": 420, "body_health": 200,
                      "helm_health": 0, "shield_health": 0}],
            packets=[{"index": 0, "type": 0, "imitater_type": 48, "active": True,
                      "cooldown": 0, "refresh_time": 3000, "cost": 100}],
        )
        changed = {
            **observation,
            "plants": [{**observation["plants"][0], "type": 1}],
            "zombies": [{**observation["zombies"][0], "type": 1}],
            "packets": [{**observation["packets"][0], "type": 1}],
        }

        self.assertFalse(torch.equal(search_value_features(observation), search_value_features(changed)))

    def test_value_checkpoint_round_trips_and_rejects_other_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "properties").mkdir()
            (root / "main.pak").write_bytes(b"main")
            (root / "properties" / "partner.xml").write_bytes(b"partner")
            signature = task_signature(7, [0, 1, 2, 3, 4, 5], 1.0, root)
            checkpoint = root / "value.pt"
            torch.manual_seed(0)
            save_search_value(checkpoint, SearchValueModel(), task_signature=signature, bootstrap_seeds=[1])

            model, metadata = load_search_value(checkpoint, torch.device("cpu"), signature)

            self.assertIsInstance(model, SearchValueModel)
            self.assertEqual(metadata["bootstrap_seeds"], [1])
            with self.assertRaisesRegex(ValueError, "task signature"):
                load_search_value(checkpoint, torch.device("cpu"), {**signature, "level": 8})

    def test_value_checkpoint_rejects_a_stale_feature_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.pt"
            torch.save({
                "search_value_version": SEARCH_VALUE_VERSION,
                "feature_version": SEARCH_VALUE_FEATURE_VERSION - 1,
            }, path)

            with self.assertRaisesRegex(ValueError, "feature version"):
                load_search_value(path, torch.device("cpu"), {})


class SearchValueDeviceTests(unittest.TestCase):
    """``predict`` runs once per search leaf, so it must not re-derive the device each call."""

    def test_the_device_anchor_tracks_the_parameters(self) -> None:
        model = SearchValueModel()
        self.assertEqual(model.device(), next(model.parameters()).device)

        moved = model.to(torch.device("cpu"))
        self.assertEqual(moved.device(), next(moved.parameters()).device)
        self.assertEqual(moved.device().type, "cpu")

    def test_the_device_anchor_stays_out_of_the_state_dict(self) -> None:
        """A persistent anchor would make every existing checkpoint unloadable."""
        keys = set(SearchValueModel().state_dict())

        self.assertNotIn("_device_anchor", keys)
        self.assertTrue(all(key.startswith("network.") for key in keys), sorted(keys))

    def test_predict_matches_an_explicit_device_transfer(self) -> None:
        torch.manual_seed(0)
        model = SearchValueModel().eval()
        observation = _observation()

        with torch.no_grad():
            expected = float(model(search_value_features(observation)
                                   .to(next(model.parameters()).device).unsqueeze(0)).item())
        self.assertEqual(model.predict(observation), expected)


class LabelDiagnosticTests(unittest.TestCase):
    def _sample(self, ticks: int) -> dict:
        action = {"type": "wait", "ticks": ticks}
        return {"visible_state_key": "same", "action": action,
                "policy": [{"action": action, "probability": 1.0}]}

    def test_action_and_policy_disagreement_are_reported(self) -> None:
        result = compare_label_samples({300: [self._sample(60)], 900: [self._sample(300)]})

        self.assertEqual(result["matched_clusters"], 1)
        self.assertEqual(result["horizon_pairs"], 1)
        self.assertEqual(result["action_disagreement_rate"], 1.0)
        self.assertEqual(result["mean_search_policy_total_variation"], 1.0)

    def test_agreeing_horizons_report_no_disagreement(self) -> None:
        result = compare_label_samples({300: [self._sample(60)], 900: [self._sample(60)]})

        self.assertEqual(result["action_disagreement_rate"], 0.0)
        self.assertEqual(result["mean_search_policy_total_variation"], 0.0)

    def test_a_horizon_with_no_samples_for_a_state_is_not_a_match(self) -> None:
        """An empty bucket would otherwise report a spurious 100% disagreement."""
        result = compare_label_samples({300: [self._sample(60)], 900: []})

        self.assertEqual(result["matched_clusters"], 0)
        self.assertEqual(result["horizon_pairs"], 0)
        self.assertIsNone(result["action_disagreement_rate"])
        self.assertIsNone(result["mean_search_policy_total_variation"])


class DiscountConstantTests(unittest.TestCase):
    def test_reference_span_and_gamma_are_the_documented_values(self) -> None:
        self.assertEqual(DISCOUNT_REFERENCE_TICKS, 300)
        self.assertAlmostEqual(VALUE_GAMMA ** (DISCOUNT_REFERENCE_TICKS / DISCOUNT_REFERENCE_TICKS), VALUE_GAMMA)


if __name__ == "__main__":
    unittest.main()
