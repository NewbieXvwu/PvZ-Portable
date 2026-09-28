"""Tests for the structured GameplayModel-v1 network, its token stream, and the imitation loss.

``observation()`` below is a complete, schema-accurate observation: every key the model
reads is present, so a missing-key regression fails here rather than on a cluster.
"""

from __future__ import annotations

import contextlib
import io
import math
import unittest

import torch

from pvz_agent_model import (
    FEATURE_COUNT,
    MODEL_CONFIG,
    TOKEN_KINDS,
    WAIT_TICKS,
    GameplayModelV1,
    hard_behavior_cloning_loss,
    observation_tokens,
    predict_action,
    resolve_device,
    select_action,
)
from pvz_imitation import LANE_COUNT, episode_targets, train

ROW_COUNT = 6
COL_COUNT = 9
CELL_COUNT = ROW_COUNT * COL_COUNT


def _cell(row: int, col: int) -> dict:
    return {"row": row, "col": col, "terrain": 1, "row_type": 1,
            "plant_types": [], "grid_item_types": []}


def _plant(**overrides: object) -> dict:
    plant = {
        "type": 0, "imitater_type": -1, "row": 2, "col": 3,
        "health": 300, "max_health": 300, "state": 0, "state_countdown": 0,
        "launch_counter": 0, "launch_rate": 0, "shooting_counter": 0,
        "wake_up_counter": 0, "asleep": False, "squished": False,
        "bungee_state": 0, "target_zombie_id": -1,
    }
    plant.update(overrides)
    return plant


def _zombie(**overrides: object) -> dict:
    zombie = {
        "type": 0, "row": 2, "x": 700.0, "y": 90.0,
        "body_health": 200, "body_max_health": 200,
        "helm_health": 0, "helm_max_health": 0,
        "shield_health": 0, "shield_max_health": 0,
        "phase": 0, "phase_counter": 0, "velocity_x": -0.5,
        "chilled": 0, "buttered": 0, "ice_trap": 0,
        "has_head": True, "has_arm": True, "has_object": False,
        "is_eating": False, "target_col": -1, "target_row": -1,
    }
    zombie.update(overrides)
    return zombie


def observation(**overrides: object) -> dict:
    base = {
        "terminal": False,
        "result": 0,
        "tick": 600,
        "wave": 3,
        "wave_count": 20,
        "sun": 175,
        "night": False,
        "pool": False,
        "fog": False,
        "roof": False,
        "zombie_count_multiplier": 1.0,
        "player_profile": {
            "playthrough": 2, "seed_slot_count": 6, "owned_upgrade_plants": [],
            "imitater_owned": False, "first_aid_owned": False,
            "pool_cleaner_owned": False, "roof_cleaner_owned": False,
            "rake_charges_remaining": 0,
        },
        "cells": [_cell(row, col) for row in range(ROW_COUNT) for col in range(COL_COUNT)],
        "plants": [_plant()],
        "zombies": [_zombie()],
        "projectiles": [{"type": 0, "row": 2, "x": 420.0, "y": 90.0, "z": 0.0,
                         "vx": 6.0, "vy": 0.0, "vz": 0.0, "motion": 0,
                         "damage": 20, "age": 10, "target_zombie_id": 1}],
        "defenses": [{"type": 0, "row": 0, "state": 1, "x": 40.0, "y": 30.0}],
        "grid_items": [],
        "packets": [{"index": index, "type": index, "imitater_type": -1, "cost": 100,
                     "cooldown": 0, "refresh_time": 3000, "active": True}
                    for index in range(6)],
        "loadout_context": {"zombie_roster": [0, 1, 2]},
        "legal_actions": {
            "plants": [{"packet": packet, "row": 1, "col": 2 + packet} for packet in range(3)],
            "shovels": [[3, 2]],
            "wait": True,
        },
    }
    base.update(overrides)
    return base


class ObservationTokenTests(unittest.TestCase):
    def test_token_stream_covers_every_entity(self) -> None:
        tensors, metadata = observation_tokens(observation())

        # global + profile + 54 cells + plant + zombie + projectile + defense + 6 packets + 3 roster
        expected = 2 + CELL_COUNT + 4 + 6 + 3
        for name in ("kinds", "categories", "variants", "rows", "cols"):
            self.assertEqual(tuple(tensors[name].shape), (expected,), name)
        self.assertEqual(tuple(tensors["features"].shape), (expected, FEATURE_COUNT))
        self.assertEqual(tensors["features"].dtype, torch.float32)

    def test_cell_tokens_are_indexed_by_row_major_position(self) -> None:
        """``select_action`` builds cell ids as ``row * 9 + col``; the tokens must agree."""
        tensors, metadata = observation_tokens(observation())

        self.assertEqual(sorted(metadata["cell_tokens"]), list(range(CELL_COUNT)))
        for key, token in metadata["cell_tokens"].items():
            self.assertEqual(int(tensors["kinds"][token]), TOKEN_KINDS["cell"])
            self.assertEqual(int(tensors["rows"][token]), key // COL_COUNT)
            self.assertEqual(int(tensors["cols"][token]), key % COL_COUNT)

    def test_packet_tokens_are_indexed_by_packet_slot(self) -> None:
        source = observation()
        tensors, metadata = observation_tokens(source)

        self.assertEqual(sorted(metadata["packet_tokens"]), [packet["index"] for packet in source["packets"]])
        for index, token in metadata["packet_tokens"].items():
            self.assertEqual(int(tensors["kinds"][token]), TOKEN_KINDS["seed_packet"])
            self.assertEqual(int(tensors["categories"][token]), source["packets"][index]["type"] + 1)

    def test_entity_types_are_clamped_into_the_embedding_range(self) -> None:
        source = observation(zombies=[_zombie(type=5000)], plants=[_plant(type=-99, imitater_type=9999)])

        tensors, _ = observation_tokens(source)

        for name in ("categories", "variants"):
            self.assertGreaterEqual(int(tensors[name].min()), 0)
            self.assertLess(int(tensors[name].max()), 128)


class GameplayModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = GameplayModelV1().eval()

    def test_step_returns_the_documented_output_shapes(self) -> None:
        source = observation()

        with torch.no_grad():
            output = self.model.step(source)

        self.assertEqual(tuple(output["type_logits"].shape), (3,))
        self.assertEqual(tuple(output["wait_logits"].shape), (len(WAIT_TICKS),))
        self.assertEqual(output["packet_ids"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(tuple(output["packet_logits"].shape), (6,))
        self.assertEqual(tuple(output["cell_tokens"].shape), (CELL_COUNT, MODEL_CONFIG["width"]))
        self.assertEqual(tuple(output["cell_keys"].shape), (CELL_COUNT, MODEL_CONFIG["width"]))
        self.assertEqual(tuple(output["value"].shape), (1,))
        self.assertEqual(output["wave_index"], source["wave"])

    def test_hidden_state_advances_with_the_recurrence(self) -> None:
        source = observation()
        with torch.no_grad():
            first = self.model.step(source)
            second = self.model.step(source, first["hidden"])

        expected = (MODEL_CONFIG["gru_layers"], 1, MODEL_CONFIG["gru_width"])
        self.assertEqual(tuple(first["hidden"].shape), expected)
        self.assertEqual(tuple(second["hidden"].shape), expected)
        self.assertFalse(torch.allclose(first["hidden"], second["hidden"]))

    def test_step_survives_an_empty_board(self) -> None:
        source = observation(plants=[], zombies=[], projectiles=[], defenses=[], packets=[],
                             loadout_context={"zombie_roster": []})

        with torch.no_grad():
            output = self.model.step(source)

        self.assertEqual(output["packet_ids"], [])
        self.assertEqual(tuple(output["packet_logits"].shape), (0,))

    def test_step_rejects_an_unknown_previous_action_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported action type"):
            self.model.step(observation(), None, {"type": "sun"}, 0, {})

    def test_resolve_device_accepts_cpu_and_rejects_unavailable_backends(self) -> None:
        self.assertEqual(resolve_device("cpu").type, "cpu")
        for backend in ("cuda", "mps"):
            available = (torch.cuda.is_available() if backend == "cuda"
                         else torch.backends.mps.is_available())
            if available:
                continue
            with self.assertRaisesRegex(ValueError, backend.upper()):
                resolve_device(backend)

    def test_auto_never_selects_mps(self) -> None:
        """``auto`` must not pick MPS: every model call in this project is batch-of-1.

        Measured on an Apple M5 Pro, ``auto`` landing on MPS made a rollout
        episode 1.71x slower (13.2 s -> 22.6 s) because the search value model is
        evaluated once per leaf.  MPS stays reachable through an explicit
        ``--device mps``.
        """
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(resolve_device("auto").type, expected)
        self.assertNotEqual(resolve_device("auto").type, "mps")
        if torch.backends.mps.is_available():
            self.assertEqual(resolve_device("mps").type, "mps")


class SelectActionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = GameplayModelV1().eval()

    def _output(self, source: dict) -> dict:
        with torch.no_grad():
            return self.model.step(source)

    def test_forced_legal_actions_are_reproduced_with_finite_scores(self) -> None:
        source = observation()
        source["legal_actions"]["plants"] = [{"packet": 1, "row": 1, "col": 3}]
        actions = [
            {"type": "plant", "packet": 1, "row": 1, "col": 3},
            {"type": "shovel", "col": 3, "row": 2},
            {"type": "wait", "ticks": 150},
        ]

        for action in actions:
            with self.subTest(action=action):
                selected, log_prob, entropy = select_action(
                    self.model, self._output(source), source, action=action)
                self.assertEqual(selected, action)
                self.assertTrue(torch.isfinite(log_prob))
                self.assertTrue(torch.isfinite(entropy))

    def test_an_illegal_forced_action_is_rejected(self) -> None:
        source = observation()
        source["legal_actions"]["plants"] = [{"packet": 1, "row": 1, "col": 3}]
        output = self._output(source)

        with self.assertRaisesRegex(ValueError, "unsupported action type"):
            select_action(self.model, output, source, action={"type": "sun"})
        with self.assertRaisesRegex(ValueError, "illegal plant packet"):
            select_action(self.model, output, source, action={"type": "plant", "packet": 5, "row": 1, "col": 3})
        with self.assertRaisesRegex(ValueError, "illegal plant cell"):
            select_action(self.model, output, source, action={"type": "plant", "packet": 1, "row": 4, "col": 4})

    def test_a_plant_is_illegal_when_no_card_can_be_played(self) -> None:
        source = observation()
        source["legal_actions"]["plants"] = []

        with self.assertRaisesRegex(ValueError, "action is illegal"):
            select_action(self.model, self._output(source), source,
                          action={"type": "plant", "packet": 0, "row": 1, "col": 2})

    def test_sampled_actions_are_always_legal(self) -> None:
        source = observation()
        legal = source["legal_actions"]

        for seed in range(8):
            torch.manual_seed(seed)
            action, log_prob, _ = select_action(self.model, self._output(source), source)
            self.assertTrue(torch.isfinite(log_prob))
            if action["type"] == "plant":
                self.assertIn((action["packet"], action["row"], action["col"]),
                              {(item["packet"], item["row"], item["col"]) for item in legal["plants"]})
            elif action["type"] == "shovel":
                self.assertIn([action["col"], action["row"]], legal["shovels"])
            else:
                self.assertIn(action["ticks"], WAIT_TICKS)

    def test_deterministic_selection_is_repeatable(self) -> None:
        source = observation()
        output = self._output(source)

        first, _, _ = select_action(self.model, output, source, deterministic=True)
        second, _, _ = select_action(self.model, output, source, deterministic=True)

        self.assertEqual(first, second)

    def test_predict_action_returns_a_legal_action_and_the_next_hidden_state(self) -> None:
        source = observation()

        action, hidden, output = predict_action(self.model, source, None, None, 0, {})

        self.assertIs(hidden, output["hidden"])
        self.assertEqual(tuple(hidden.shape), (MODEL_CONFIG["gru_layers"], 1, MODEL_CONFIG["gru_width"]))
        self.assertIn(action["type"], {"plant", "shovel", "wait"})


class BehaviorCloningLossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = GameplayModelV1()

    def _loss(self, action: dict, source: dict | None = None) -> torch.Tensor:
        source = observation() if source is None else source
        output = self.model.step(source)
        return hard_behavior_cloning_loss(self.model, output, source, action)

    def test_loss_is_a_differentiable_scalar_for_every_action_kind(self) -> None:
        for action in ({"type": "plant", "packet": 1, "row": 1, "col": 3},
                       {"type": "shovel", "col": 3, "row": 2},
                       {"type": "wait", "ticks": 150}):
            with self.subTest(action=action):
                loss = self._loss(action)
                self.assertEqual(tuple(loss.shape), ())
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(loss.requires_grad)

    def test_an_unsupported_target_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported action type"):
            self._loss({"type": "sun"})
        with self.assertRaisesRegex(ValueError, "wait ticks must be one of"):
            self._loss({"type": "wait", "ticks": 90})
        with self.assertRaises(ValueError):
            self._loss({"type": "plant", "packet": 99, "row": 1, "col": 3})

    def test_the_loss_produces_a_gradient_on_the_action_type_head(self) -> None:
        """A zero type gradient means the model could never learn which action kind to pick."""
        self.model.zero_grad(set_to_none=True)

        self._loss({"type": "wait", "ticks": 150}).backward()

        gradient = self.model.action_type.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_one_optimizer_step_moves_the_policy_towards_the_demonstration(self) -> None:
        """The loss is only useful if minimising it actually favours the shown action."""
        source = observation()
        for action in ({"type": "plant", "packet": 1, "row": 1, "col": 3},
                       {"type": "shovel", "col": 3, "row": 2},
                       {"type": "wait", "ticks": 150}):
            with self.subTest(action=action):
                optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)

                before = self._loss(action, source)
                before.backward()
                optimizer.step()
                after = self._loss(action, source)

                self.assertLess(float(after.detach()), float(before.detach()))


class ImitationTargetTests(unittest.TestCase):
    def test_lane_targets_mark_the_rows_occupied_by_zombies(self) -> None:
        steps = [{"observation": observation(zombies=[_zombie(row=2), _zombie(row=4)]), "events": {}}]

        lanes, next_wave = episode_targets(steps, 0, torch.device("cpu"))

        self.assertEqual(tuple(lanes.shape), (1, LANE_COUNT))
        self.assertEqual(lanes[0].tolist(), [0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        self.assertEqual(float(next_wave), 0.0)

    def test_out_of_range_zombie_rows_are_ignored(self) -> None:
        """A row outside the six lanes must neither raise nor wrap into another lane."""
        steps = [{"observation": observation(zombies=[_zombie(row=9), _zombie(row=-1), _zombie(row=3)]),
                  "events": {}}]

        lanes, _ = episode_targets(steps, 0, torch.device("cpu"))

        self.assertEqual(lanes[0].tolist(), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def test_next_wave_target_reads_the_following_step(self) -> None:
        steps = [{"observation": observation(), "events": {}},
                 {"observation": observation(tick=660), "events": {"waves_started": 1}}]

        _, first = episode_targets(steps, 0, torch.device("cpu"))
        _, last = episode_targets(steps, 1, torch.device("cpu"))

        self.assertEqual(float(first), 1.0)
        self.assertEqual(float(last), 0.0)


class ImitationTrainTests(unittest.TestCase):
    def test_one_epoch_reports_a_finite_loss_and_balances_plant_steps(self) -> None:
        torch.manual_seed(0)
        model = GameplayModelV1()
        episode = {
            "tick": 660,
            "won": True,
            "steps": [
                {"observation": observation(), "action": {"type": "wait", "ticks": 60},
                 "delta_ticks": 60, "events": {}},
                {"observation": observation(tick=660),
                 "action": {"type": "plant", "packet": 1, "row": 1, "col": 3},
                 "delta_ticks": 0, "events": {}},
            ],
        }

        # train() prints its per-epoch progress; keep the test output clean.
        with contextlib.redirect_stdout(io.StringIO()):
            history, plant_weight = train(model, [episode], epochs=1, device=torch.device("cpu"))

        self.assertEqual(len(history), 1)
        self.assertTrue(math.isfinite(history[0]))
        self.assertGreaterEqual(history[0], 0.0)
        # One plant step against one non-plant step balances the two classes exactly.
        self.assertAlmostEqual(plant_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
