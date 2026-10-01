"""Tests for the structured GameplayModel-v1 network, its token stream, and its action losses.

``observation()`` below is a complete, schema-accurate observation: every key the model
reads is present, so a missing-key regression fails here rather than on a cluster.
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pvz_agent_model import (
    DEFAULT_TORCH_THREADS,
    FEATURE_COUNT,
    MODEL_ARCHITECTURE_VERSION,
    MODEL_CONFIG,
    TOKEN_KINDS,
    WAIT_TICKS,
    GameplayModelV1,
    configure_torch_threads,
    derive_observation_features,
    factored_action_log_prob,
    hard_behavior_cloning_loss,
    legal_summary,
    observation_tokens,
    pack_tokens,
    predict_action,
    resolve_device,
    replay_log_probs,
    select_action,
    soft_behavior_cloning_loss,
    unpack_tokens,
)
from pvz_agent_model import _ratio
from train_pvz_ppo import collect_task_episode

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
        "wave_timer": 2400,
        "sun_income_rate": 12.0,
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

        # global + profile + 54 cells + 6 lane summaries + plant + zombie + projectile + defense + 6 packets + 3 roster
        expected = 2 + CELL_COUNT + 6 + 4 + 6 + 3
        for name in ("kinds", "categories", "variants", "rows", "cols"):
            self.assertEqual(tuple(tensors[name].shape), (expected,), name)
        self.assertEqual(tuple(tensors["features"].shape), (expected, FEATURE_COUNT))
        self.assertEqual(tensors["features"].dtype, torch.float32)

    def test_lane_context_vocabulary_bumps_the_architecture_version(self) -> None:
        self.assertEqual(MODEL_ARCHITECTURE_VERSION, 7)
        self.assertEqual(TOKEN_KINDS["lane"], 10)
        self.assertEqual(TOKEN_KINDS["lane_enemy"], 11)
        self.assertEqual(len(TOKEN_KINDS), 12)

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

    def test_lane_tokens_are_six_row_aggregates_and_enter_attention(self) -> None:
        source = observation()
        tensors, metadata = observation_tokens(source)
        lane_positions = [index for index, kind in enumerate(tensors["kinds"])
                          if int(kind) == TOKEN_KINDS["lane"]]

        self.assertEqual(len(lane_positions), ROW_COUNT)
        self.assertEqual(set(metadata["lane_tokens"]), set(range(ROW_COUNT)))
        self.assertEqual([int(tensors["rows"][index]) for index in lane_positions], list(range(ROW_COUNT)))

        seen_attention_kinds = []
        with torch.random.fork_rng(devices=[]):
            model = GameplayModelV1().eval()
            handle = model.encoder[0].attention.register_forward_pre_hook(
                lambda _module, args: seen_attention_kinds.append(args[1].detach().clone()))
            try:
                model.step(source)
            finally:
                handle.remove()
        self.assertEqual(int((seen_attention_kinds[0] == TOKEN_KINDS["lane"]).sum()), ROW_COUNT)

    def test_derived_features_change_with_observation_state(self) -> None:
        source = observation(
            plants=[_plant(type=1, row=2, health=250), _plant(type=0, row=2, health=100)],
            zombies=[_zombie(row=2, x=470.0, body_health=1000, body_max_health=1000)],
            sun_income_rate=25.0, wave=4, wave_count=20, wave_timer=1200)
        features = derive_observation_features(source)
        self.assertGreater(features["lane_features"][2][0], 0.0)
        self.assertLess(features["lane_features"][2][1], 1.0)
        self.assertGreater(features["lane_features"][2][2], 0.0)
        self.assertGreater(features["lane_features"][2][3], 0.0)
        self.assertEqual(features["sun_income_rate"], 25.0)
        self.assertEqual(features["economic_fire_ratio"], 1.0)
        self.assertAlmostEqual(features["wave_progress"], 0.2)
        self.assertAlmostEqual(features["next_wave_distance"], 0.2)

    def test_entity_types_are_clamped_into_the_embedding_range(self) -> None:
        source = observation(zombies=[_zombie(type=5000)], plants=[_plant(type=-99, imitater_type=9999)])

        tensors, _ = observation_tokens(source)

        for name in ("categories", "variants"):
            self.assertGreaterEqual(int(tensors[name].min()), 0)
            self.assertLess(int(tensors[name].max()), 128)

    def test_narrow_packed_tokens_preserve_ids_and_actions(self) -> None:
        source = observation()
        tensors, metadata = observation_tokens(source)
        packed = pack_tokens(tensors, metadata)

        self.assertEqual(packed["ids"].dtype, np.int8)
        self.assertEqual(packed["features"].dtype, np.float16)
        self.assertEqual(packed["cell_index"].dtype, np.uint16)
        self.assertEqual(packed["packet_ids"].dtype, np.uint8)
        restored_tensors, restored_metadata = unpack_tokens(packed, torch.device("cpu"))
        for name in ("kinds", "categories", "variants", "rows", "cols"):
            torch.testing.assert_close(restored_tensors[name], tensors[name], rtol=0, atol=0)
        self.assertLessEqual(float((restored_tensors["features"] - tensors["features"]).abs().max()),
                             5.0e-4)
        self.assertEqual(restored_metadata["cell_tokens"], metadata["cell_tokens"])
        self.assertEqual(restored_metadata["packet_tokens"], metadata["packet_tokens"])

        model = GameplayModelV1().eval()
        with torch.no_grad():
            direct = model.step_tokens(tensors, metadata, source["wave"], None, None, 0, {})
            restored = model.step_tokens(restored_tensors, restored_metadata, source["wave"],
                                         None, None, 0, {})
            legal = legal_summary(source["legal_actions"])
            torch.testing.assert_close(restored["type_logits"], direct["type_logits"], rtol=0, atol=0)
            action_rng_state = torch.random.get_rng_state()
            direct_action, _, _ = select_action(model, direct, legal)
            torch.random.set_rng_state(action_rng_state)
            restored_action, _, _ = select_action(model, restored, legal)
        self.assertEqual(restored_action, direct_action)

        transition = {
            "tokens": packed,
            "previous_action": None,
            "elapsed_since_previous_observation": 0,
            "events": {},
            "wave": source["wave"],
        }
        with torch.no_grad():
            batched, _ = model.forward_sequences([[transition]], [None])
        torch.testing.assert_close(batched[0]["type_logits"], direct["type_logits"],
                                   rtol=1e-5, atol=1e-6)


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

    def test_compact_critic_inputs_match_the_full_privileged_state(self) -> None:
        privileged = {"hidden": {
            "wave_timer": 3400,
            "zombies_in_wave": [[0, 1, 1, 14, 15, 0], [2, 2, 3]],
        }}
        for wave in (-2, 0, 1, 50):
            full = self.model.privileged_extra(privileged, wave)
            selected = privileged["hidden"]["zombies_in_wave"][min(max(0, wave), 1)]
            compact = self.model.privileged_extra_from_inputs(3400, selected)
            self.assertEqual(compact, full)

    def test_batched_replay_masks_work_on_the_model_device(self) -> None:
        source = observation()
        legal = legal_summary(source["legal_actions"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GameplayModelV1().to(device).eval()
        tensors, metadata = observation_tokens(source)
        packed = pack_tokens(tensors, metadata)
        transition = {
            "tokens": packed,
            "previous_action": None,
            "elapsed_since_previous_observation": 0,
            "events": {},
            "wave": source["wave"],
            "legal": legal,
        }
        actions = [
            {"type": "plant", "packet": legal["packets"][0],
             "row": 0, "col": 0},
            {"type": "shovel", "row": 2, "col": 3},
            {"type": "wait", "ticks": 150},
        ]
        # The fixture has at least one legal cell for packet 0, one shovel cell, and wait.
        plant = source["legal_actions"]["plants"][0]
        actions[0].update({"row": plant["row"], "col": plant["col"]})
        with torch.no_grad():
            outputs, _ = model.forward_sequences([[
                {**transition, "action": action} for action in actions
            ]], [None])
            log_probs, entropies = replay_log_probs(
                model, outputs, [{**transition, "action": action} for action in actions])
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.isfinite(entropies).all())

    def test_batched_replay_equals_select_action_per_transition(self) -> None:
        """The batched replay must reproduce ``select_action`` number for number.

        ``replay_log_probs`` exists only to fold ~30 small ops per transition into
        one op per chunk; it is not allowed to change the result.  The two paths
        build their masks in entirely different ways -- ``select_action`` walks a
        Python set and writes one device element at a time, the batched version
        builds one index tensor per chunk -- so a masking slip would surface as a
        silently wrong PPO ratio rather than as a crash.

        The chunk mixes a truncated packet list with full ones, so the padding
        branch of the packet distribution is exercised, and covers all three
        action types.
        """
        full = observation()
        short = observation(packets=full["packets"][:3])
        sources = [full, short, full, full]
        summaries = [legal_summary(source["legal_actions"]) for source in sources]
        tokens = [pack_tokens(*observation_tokens(source)) for source in sources]
        plant = full["legal_actions"]["plants"][0]
        shovel_col, shovel_row = full["legal_actions"]["shovels"][0]
        actions = [
            {"type": "plant", "packet": plant["packet"],
             "row": plant["row"], "col": plant["col"]},
            {"type": "plant", "packet": plant["packet"],
             "row": plant["row"], "col": plant["col"]},
            {"type": "shovel", "row": shovel_row, "col": shovel_col},
            {"type": "wait", "ticks": 150},
        ]
        transitions = [
            {"tokens": packed, "previous_action": None,
             "elapsed_since_previous_observation": 0, "events": {},
             "wave": source["wave"], "legal": summary, "action": action}
            for packed, source, summary, action in zip(tokens, sources, summaries, actions)
        ]
        with torch.no_grad():
            outputs, _ = self.model.forward_sequences([transitions], [None])
            batch_log_probs, batch_entropies = replay_log_probs(
                self.model, outputs, transitions)
            expected = [
                select_action(self.model, outputs[index], transition["legal"],
                              action=transition["action"])
                for index, transition in enumerate(transitions)
            ]
            # The chunk is only meaningful if the packet rows really are ragged.
            self.assertEqual(outputs[0]["packet_logits"].shape[0], 6)
            self.assertEqual(outputs[1]["packet_logits"].shape[0], 3)
            for index, (_, expected_log_prob, expected_entropy) in enumerate(expected):
                self.assertAlmostEqual(float(batch_log_probs[index]),
                                       float(expected_log_prob), places=5)
                self.assertAlmostEqual(float(batch_entropies[index]),
                                       float(expected_entropy), places=5)

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


class TorchThreadPolicyTests(unittest.TestCase):
    """The ``--threads`` knob is a reproducibility choice, not a correctness one.

    Training entry points used to hard-code ``torch.set_num_threads(1)``.  That pin
    is free on Apple Silicon but costs 2.0x on an x86 desktop, so it is now a
    parameter.  What must stay true is that the *resolved* value is the one that
    gets applied and reported -- a run that silently used a different thread count
    than its provenance records would be untraceable.
    """

    def setUp(self) -> None:
        self._original = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self._original)

    def test_zero_and_negative_select_the_measured_default(self) -> None:
        for requested in (0, -1, -64):
            with self.subTest(requested=requested):
                self.assertEqual(configure_torch_threads(requested), DEFAULT_TORCH_THREADS)
                self.assertEqual(torch.get_num_threads(), DEFAULT_TORCH_THREADS)
        self.assertGreaterEqual(DEFAULT_TORCH_THREADS, 1)

    def test_an_explicit_count_is_applied_verbatim_and_returned(self) -> None:
        for requested in (1, 2, 3, 7):
            with self.subTest(requested=requested):
                self.assertEqual(configure_torch_threads(requested), requested)
                self.assertEqual(torch.get_num_threads(), requested)

    def test_a_fixed_thread_count_is_deterministic_across_calls(self) -> None:
        """The documented contract: same machine + same thread count => same bits.

        This is what makes the knob safe to expose at all.  ``--threads 1`` and
        ``--threads 4`` do not agree bit-for-bit on x86, but a run stays
        reproducible *given the thread count its provenance records*.
        """
        torch.manual_seed(3)
        model = GameplayModelV1().eval()
        source = observation()
        outputs = []
        for _ in range(2):
            self.assertEqual(configure_torch_threads(2), torch.get_num_threads())
            with torch.no_grad():
                outputs.append(model.step(source))
        self.assertTrue(torch.equal(outputs[0]["packet_logits"], outputs[1]["packet_logits"]))
        self.assertTrue(torch.equal(outputs[0]["value"], outputs[1]["value"]))


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


class SoftLabelDistillationTests(unittest.TestCase):
    """The teacher writes a distribution over its whole candidate set; these pin that it is used.

    ``collect_search_episode`` records ``candidate_actions`` / ``search_policy`` on every
    step, and until now nothing read them: training minimised the cross-entropy of the
    single executed action and threw the margin between the runner-up lines away.
    """

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = GameplayModelV1()
        self.source = observation()
        self.demonstrated = {"type": "plant", "packet": 0, "row": 1, "col": 2}
        self.alternative = {"type": "wait", "ticks": 300}

    def _output(self) -> dict:
        return self.model.step(self.source)

    def _hard(self, action: dict) -> float:
        return float(hard_behavior_cloning_loss(self.model, self._output(), self.source, action).detach())

    def _log_prob(self, output: dict, action: dict) -> float:
        return float(factored_action_log_prob(self.model, output, self.source, action).detach())

    def test_factored_log_prob_is_the_negative_of_the_hard_loss(self) -> None:
        """The soft term must score a candidate exactly the way the hard loss scores it.

        ``hard_behavior_cloning_loss`` sums the type/packet/cell cross-entropies and
        ``select_action`` multiplies those same three factors, so the two views of one
        action have to agree; if they did not, soft and hard supervision would be
        pulling on two different objectives.
        """
        output = self._output()
        for action in ({"type": "plant", "packet": 1, "row": 1, "col": 3},
                       {"type": "shovel", "col": 3, "row": 2},
                       {"type": "wait", "ticks": 150}):
            with self.subTest(action=action):
                hard = hard_behavior_cloning_loss(self.model, output, self.source, action, plant_weight=1.0)
                self.assertAlmostEqual(-self._log_prob(output, action), float(hard.detach()), places=6)

    def test_a_one_hot_policy_reduces_to_the_hard_loss(self) -> None:
        output = self._output()

        soft = soft_behavior_cloning_loss(self.model, output, self.source, [self.demonstrated], [1.0])

        self.assertAlmostEqual(float(soft.detach()), self._hard(self.demonstrated), places=6)

    def test_the_soft_loss_interpolates_the_candidate_losses(self) -> None:
        output = self._output()

        soft = soft_behavior_cloning_loss(self.model, output, self.source,
                                          [self.demonstrated, self.alternative], [0.75, 0.25])

        expected = 0.75 * self._hard(self.demonstrated) + 0.25 * self._hard(self.alternative)
        self.assertAlmostEqual(float(soft.detach()), float(expected), places=6)

    def test_candidates_with_no_policy_mass_are_dropped(self) -> None:
        output = self._output()

        with_zero = soft_behavior_cloning_loss(self.model, output, self.source,
                                               [self.demonstrated, self.alternative], [1.0, 0.0])
        alone = soft_behavior_cloning_loss(self.model, output, self.source, [self.demonstrated], [1.0])

        self.assertAlmostEqual(float(with_zero.detach()), float(alone.detach()), places=6)

    def test_plant_weight_renormalises_the_soft_target(self) -> None:
        """Upweighting plants must change the mixture, not the total mass of the target."""
        output = self._output()

        weighted = soft_behavior_cloning_loss(self.model, output, self.source,
                                              [self.demonstrated, self.alternative], [0.5, 0.5],
                                              plant_weight=3.0)

        expected = ((3.0 * 0.5) * self._hard(self.demonstrated)
                    + 0.5 * self._hard(self.alternative)) / (3.0 * 0.5 + 0.5)
        self.assertAlmostEqual(float(weighted.detach()), float(expected), places=6)

    def test_malformed_policy_input_is_rejected(self) -> None:
        output = self._output()

        with self.assertRaisesRegex(ValueError, "candidates but"):
            soft_behavior_cloning_loss(self.model, output, self.source, [self.demonstrated], [0.5, 0.5])
        with self.assertRaisesRegex(ValueError, "no probability"):
            soft_behavior_cloning_loss(self.model, output, self.source, [self.demonstrated], [0.0])

class CriticInputCacheTests(unittest.TestCase):
    """The rollout asks for the wave roster once per wave, not once per decision.

    ``CRITIC_INPUTS`` used to be one synchronous round trip per decision, even though
    the roster it returns is fixed for the whole wave (measured: 272 calls over a
    5-wave level produced 4 distinct payloads).  The other field it returns,
    ``wave_timer``, is the same ``mZombieCountDown`` the observation already carries,
    so the trainer must read it from the observation.
    """

    TASK = {"task_id": "scripted", "level": 1, "playthrough": 2,
            "zombie_count_multiplier": 1.0, "wave_cap": 5, "preplanted": (),
            "seeds": [1], "deck": None}

    class _ScriptedEnv:
        """Walks a fixed wave schedule; reports a deliberately wrong wave_timer."""

        WRONG_WAVE_TIMER = 9999

        def __init__(self, waves: list[int], terminal_at: int) -> None:
            self.waves = waves
            self.terminal_at = terminal_at
            self.critic_calls: list[int] = []
            self.index = 0

        def _observe(self) -> dict:
            return observation(wave=self.waves[min(self.index, len(self.waves) - 1)])

        def reset(self, deck=None, task=None):  # noqa: ANN001, ANN201
            self.index = 0
            return self._observe(), {}

        def step(self, action):  # noqa: ANN001, ANN201
            self.index += 1
            current = self._observe()
            done = self.index >= self.terminal_at
            if done:
                current = dict(current, terminal=True, result=1)
            return current, 0.0, done, False, {"ok": True, "ticks_advanced": 30, "events": {}}

        def critic_inputs(self, wave_index: int) -> dict:
            self.critic_calls.append(wave_index)
            return {"ok": True, "wave_timer": self.WRONG_WAVE_TIMER, "wave_zombies": [0, 1]}

    def test_one_request_per_wave_and_wave_timer_comes_from_the_observation(self) -> None:
        waves = [3, 3, 3, 4, 4, 4, 5, 5]
        # The episode ends on the step that returns terminal, so all eight waves
        # entries become one decision each.
        env = self._ScriptedEnv(waves, terminal_at=len(waves))
        episode = collect_task_episode(GameplayModelV1().eval(), env, self.TASK, 1, 0, 64)

        # Three distinct waves, eight decisions.
        self.assertEqual(env.critic_calls, [3, 4, 5])
        self.assertEqual(len(episode["transitions"]), len(waves))

        # critic_extra[0] is _ratio(wave_timer, 6000).  The scripted env reports 9999
        # so a value of 2400/6000 proves the trainer read the observation instead.
        expected = _ratio(observation()["wave_timer"], 6000.0)
        self.assertNotAlmostEqual(expected, _ratio(self._ScriptedEnv.WRONG_WAVE_TIMER, 6000.0))
        for transition in episode["transitions"]:
            self.assertAlmostEqual(transition["critic_extra"][0], expected)

    def test_the_cached_roster_still_reaches_the_critic(self) -> None:
        env = self._ScriptedEnv([3, 3, 3, 3], terminal_at=3)
        episode = collect_task_episode(GameplayModelV1().eval(), env, self.TASK, 1, 0, 64)
        # wave_zombies = [0, 1] -> values[1] and values[2] each gain 0.1.
        for transition in episode["transitions"]:
            self.assertAlmostEqual(transition["critic_extra"][1], 0.1)
            self.assertAlmostEqual(transition["critic_extra"][2], 0.1)


if __name__ == "__main__":
    unittest.main()
