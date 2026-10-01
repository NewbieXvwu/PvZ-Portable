"""Encoding, identity invariance and real PPO replay for the isolated input candidate."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pvz_agent_model import (GameplayModelV1, TOKEN_KINDS, configure_torch_threads,
                             legal_summary, observation_tokens, pack_tokens, select_action, unpack_tokens)
from test_agent_model import observation, _zombie
from train_pvz_ppo import add_advantages, train_update
from pvz_research import load_config

SMALL = {"layers": 2, "width": 32, "heads": 4, "ff_width": 64,
         "gru_layers": 2, "gru_width": 16, "critic_width": 24, "critic_layers": 1,
         "input_flags": 7}


def source() -> dict:
    obs = observation(zombies=[_zombie(id=101, on_board=True),
                               _zombie(id=202, on_board=True, x=310, row=2)])
    obs["plants"][0]["target_zombie_id"] = 101
    obs["projectiles"][0]["target_zombie_id"] = 202
    return obs


def replay_fixture() -> tuple[GameplayModelV1, list[dict]]:
    torch.manual_seed(7)
    model = GameplayModelV1(SMALL).eval()
    episodes = []
    for length in (9, 6):
        transitions, hidden, previous = [], None, None
        for index in range(length):
            obs = source()
            obs["tick"] += index * 73
            if index % 2:
                obs["zombies"].reverse()
            tensors, metadata = observation_tokens(obs, 7)
            legal = legal_summary(obs["legal_actions"])
            duration = (0, 60, 150, 300)[index % 4]
            with torch.no_grad():
                output = model.step_tokens(tensors, metadata, obs["wave"], hidden, previous, duration, {})
                action, log_prob, _ = select_action(model, output, legal)
                value = model.privileged_value_from_extra(output, [0.0] * 16)
            transitions.append({"tokens": pack_tokens(tensors, metadata), "wave": obs["wave"],
                                "previous_action": previous, "elapsed_since_previous_observation": duration,
                                "events": {}, "critic_extra": [0.0] * 16, "legal": legal,
                                "action": action, "log_prob": log_prob.item(), "value": value.item(),
                                "action_duration_ticks": duration,
                                "reward": -1.0 if index == length - 1 else 0.0})
            hidden, previous = output["hidden"], action
        episodes.append({"transitions": transitions})
    add_advantages(episodes, .95)
    return model, episodes


class PublicContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch_threads(1)

    def test_on_board_changes_explicit_input_and_excludes_preview_from_summary(self):
        obs = source()
        before, _ = observation_tokens(obs, 1)
        obs["zombies"][0]["on_board"] = False
        after, metadata = observation_tokens(obs, 1)
        zombie = torch.where(before["kinds"] == TOKEN_KINDS["zombie"])[0][0]
        self.assertEqual(float(before["features"][zombie, 20]), 1)
        self.assertEqual(float(after["features"][zombie, 20]), 0)
        self.assertLess(float(after["features"][0, 10]), float(before["features"][0, 10]))
        self.assertLess(float(after["features"][metadata["lane_tokens"][2], 0]),
                        float(before["features"][metadata["lane_tokens"][2], 0]))

    def test_row_context_has_exact_enemy_type_groups_and_card_cooldown(self):
        obs = source()
        tensors, metadata = observation_tokens(obs, 3)
        enemies = torch.where(tensors["kinds"] == TOKEN_KINDS["lane_enemy"])[0]
        self.assertEqual(len(enemies), 1)
        self.assertEqual(int(tensors["categories"][enemies[0]]), 1)
        changed = copy.deepcopy(obs)
        changed["zombies"][1]["type"] = 2
        other, _ = observation_tokens(changed, 3)
        self.assertEqual(int((other["kinds"] == TOKEN_KINDS["lane_enemy"]).sum()), 2)
        changed = copy.deepcopy(obs)
        changed["packets"][1]["cooldown"] = 1200
        other, _ = observation_tokens(changed, 3)
        lane = metadata["lane_tokens"][2]
        self.assertGreater(int(torch.count_nonzero(tensors["features"][lane] != other["features"][lane])), 0)

    def test_target_identity_is_a_resolved_edge_not_an_integer_feature(self):
        obs = source()
        before, _ = observation_tokens(obs, 7)
        changed = copy.deepcopy(obs)
        changed["plants"][0]["target_zombie_id"] = 202
        after, _ = observation_tokens(changed, 7)
        plant = int(torch.where(before["kinds"] == TOKEN_KINDS["plant"])[0][0])
        self.assertNotEqual(int(before["target_indices"][plant]), int(after["target_indices"][plant]))
        # Arbitrary identity renaming cannot add numerical information.
        renamed = copy.deepcopy(obs)
        for z in renamed["zombies"]:
            z["id"] += 123456
        for kind in ("plants", "projectiles"):
            renamed[kind][0]["target_zombie_id"] += 123456
        renaming, _ = observation_tokens(renamed, 7)
        for key in before:
            torch.testing.assert_close(before[key], renaming[key], rtol=0, atol=0)

    def test_missing_duplicate_and_unresolved_ids_are_explicit(self):
        obs = source()
        obs["zombies"][1]["id"] = 101
        with self.assertRaisesRegex(ValueError, "unique"):
            observation_tokens(obs, 7)
        with self.assertRaisesRegex(ValueError, "zombie.on_board"):
            observation_tokens(observation(), 1)
        obs = source()
        obs["plants"][0]["target_zombie_id"] = 303
        tensors, _ = observation_tokens(obs, 7)
        plant = int(torch.where(tensors["kinds"] == TOKEN_KINDS["plant"])[0][0])
        self.assertEqual(int(tensors["target_indices"][plant]), -1)
        self.assertEqual(float(tensors["features"][plant, 14]), 1)
        self.assertEqual(float(tensors["features"][plant, 15]), 0)

    def test_formal_config_cannot_silently_omit_input_flags(self):
        root = Path(__file__).resolve().parent.parent
        config = json.loads((root / "experiments/t5/reward_comparison_v2/reward_r0_seed0_v2.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "input_flags must be explicit"):
                load_config(path)

    def test_identity_index_above_127_survives_packed_storage(self):
        obs = source()
        obs["zombies"] += [_zombie(id=1000 + i, on_board=True) for i in range(100)]
        obs["plants"][0]["target_zombie_id"] = 1099
        tensors, metadata = observation_tokens(obs, 7)
        plant = int(torch.where(tensors["kinds"] == TOKEN_KINDS["plant"])[0][0])
        self.assertGreater(int(tensors["target_indices"][plant]), 127)
        packed = pack_tokens(tensors, metadata)
        self.assertEqual(packed["target_indices"].dtype, np.int32)
        restored, _ = unpack_tokens(packed, torch.device("cpu"))
        torch.testing.assert_close(restored["target_indices"], tensors["target_indices"], rtol=0, atol=0)

    def test_target_relations_receive_actor_gradient_and_change_behavior(self):
        torch.manual_seed(3)
        model = GameplayModelV1(SMALL).eval()
        output = model.step(source())
        output["type_logits"].sum().backward()
        self.assertGreater(float(model.encoder[0].attention.target_bias.grad.abs().sum()), 0)
        changed = source()
        changed["plants"][0]["target_zombie_id"] = 202
        with torch.no_grad():
            for layer in model.encoder:
                layer.attention.target_bias.fill_(1.0)
            first, second = model.step(source()), model.step(changed)
        self.assertGreater(float((first["type_logits"] - second["type_logits"]).abs().max()), 1e-7)

    def test_entity_permutation_preserves_policy_with_corresponding_edges(self):
        torch.manual_seed(5)
        model = GameplayModelV1(SMALL).eval()
        with torch.no_grad():
            for layer in model.encoder:
                layer.attention.target_bias.fill_(0.7)
            first = model.step(source())
            reordered = source()
            reordered["zombies"].reverse()
            second = model.step(reordered)
        for key in ("belief", "type_logits", "packet_logits", "cell_keys", "wait_logits"):
            torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=2e-6)

    def test_real_ppo_replay_matches_collection_with_new_inputs_and_chunk_boundaries(self):
        model, episodes = replay_fixture()
        for sequence_length in (0, 3):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0)
            losses = train_update(model, copy.deepcopy(episodes), optimizer, torch.device("cpu"),
                                  2, sequence_length, .2, .5, .01, minibatch_chunks=2, attention_backend="dense")
            self.assertLess(losses["max_log_prob_change"], 3e-6)
            self.assertEqual(losses["clip_fraction"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fp32_cuda_replay_matches_cpu_collection_with_new_target_inputs(self):
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model, episodes = replay_fixture()
        model.to("cuda")
        for sequence_length in (0, 3):
            optimizer = torch.optim.AdamW(model.parameters(), lr=0)
            losses = train_update(model, copy.deepcopy(episodes), optimizer, torch.device("cuda"),
                                  2, sequence_length, .2, .5, .01, minibatch_chunks=2, attention_backend="dense")
            self.assertLess(losses["max_log_prob_change"], 3e-6)
            self.assertEqual(losses["clip_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
