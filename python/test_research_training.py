"""Regression checks for actual recurrent PPO replay and complete continuation."""
from __future__ import annotations

import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from pvz_agent_model import (GameplayModelV1, MODEL_CONFIG, configure_torch_threads,
                             legal_summary, observation_tokens, pack_tokens, select_action)
from pvz_research import capture_rng, restore_rng
from train_pvz_ppo import add_advantages, train_update
from test_agent_model import observation

SMALL = {"layers": 1, "width": 32, "heads": 4, "ff_width": 64,
         "gru_layers": 2, "gru_width": 16, "critic_width": 24, "critic_layers": 2}


def episodes_for(model: GameplayModelV1) -> list[dict]:
    episodes = []
    for length in (13, 10, 7):
        transitions, hidden, previous = [], None, None
        for index in range(length):
            obs = observation(tick=index * 73, sun=175 + index, wave=index // 4)
            tokens, metadata = observation_tokens(obs)
            legal = legal_summary(obs["legal_actions"])
            delta = (0, 60, 150, 300)[index % 4]
            with torch.no_grad():
                output = model.step_tokens(tokens, metadata, obs["wave"], hidden, previous, delta, {})
                action, log_prob, _ = select_action(model, output, legal)
                value = model.privileged_value_from_extra(output, [0.0] * 16)
            transitions.append({"tokens": pack_tokens(tokens, metadata),
                "wave": obs["wave"], "previous_action": previous,
                "elapsed_since_previous_observation": delta, "events": {},
                "legal": legal, "action": action, "log_prob": log_prob.item(),
                "value": value.item(), "critic_extra": [0.0] * 16,
                "action_duration_ticks": delta, "reward": -1.0 if index == length - 1 else 0.0})
            hidden, previous = output["hidden"], action
        episodes.append({"transitions": transitions})
    add_advantages(episodes, 0.95)
    return episodes


def update(model, episodes, optimizer, sequence_length=3):
    return train_update(model, episodes, optimizer, torch.device("cpu"), 2, sequence_length,
                        0.2, 0.5, 0.01, minibatch_chunks=2, attention_backend="dense")


class RecurrentReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_torch_threads(1)

    def setUp(self):
        torch.manual_seed(17)
        random.seed(17)
        self.model = GameplayModelV1(SMALL).eval()
        self.episodes = episodes_for(self.model)

    def test_real_optimizer_path_preserves_likelihood_across_all_chunk_boundaries(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0)
        losses = update(self.model, self.episodes, optimizer)
        self.assertLess(losses["max_log_prob_change"], 3e-6)
        self.assertEqual(losses["clip_fraction"], 0.0)
        self.assertGreater(losses["optimizer_steps"], 4)

    def test_prefix_hidden_is_recomputed_after_parameter_updates(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.003)
        original = self.model.forward_sequences
        starts = {id(step): (episode, index) for episode in self.episodes
                  for index, step in enumerate(episode["transitions"])}
        checks = []

        def checked(sequences, hiddens):
            if torch.is_grad_enabled():
                for sequence, hidden in zip(sequences, hiddens):
                    episode, start = starts[id(sequence[0])]
                    if start:
                        with torch.no_grad():
                            _, expected = original([episode["transitions"][:start]], [None])
                        torch.testing.assert_close(hidden, expected[:, 0, :], rtol=1e-5, atol=1e-6)
                        checks.append(start)
            return original(sequences, hiddens)

        with patch.object(self.model, "forward_sequences", side_effect=checked):
            update(self.model, self.episodes, optimizer)
        self.assertGreater(len(checks), 5)

    def test_full_episode_reference_matches_collection(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0)
        losses = update(self.model, self.episodes, optimizer, sequence_length=0)
        self.assertLess(losses["max_log_prob_change"], 3e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA replay requires a GPU")
    def test_fp32_cuda_replay_matches_cpu_collection(self):
        self.model.cuda()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0)
        with torch.backends.cudnn.flags(allow_tf32=False):
            losses = train_update(self.model, self.episodes, optimizer, torch.device("cuda"),
                                  1, 3, 0.2, 0.5, 0.01, minibatch_chunks=2,
                                  attention_backend="dense")
        self.assertLess(losses["max_log_prob_change"], 5e-5)

    def test_full_resume_produces_the_same_next_update(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        update(self.model, self.episodes, optimizer, sequence_length=0)
        assignments = random.Random(91)
        assignments.random()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resume.pt"
            torch.save({"model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": capture_rng(assignments), "updates": 1,
                        "counters": {"episodes": 3, "decisions": 30, "ticks": 1530},
                        "recent_passes": {"test": [False, True]}}, path)
            expected_rng = (random.random(), np.random.rand(), torch.rand(3), assignments.random())
            expected_losses = update(self.model, copy.deepcopy(self.episodes), optimizer, sequence_length=0)
            saved = torch.load(path, weights_only=False)
            restored = GameplayModelV1(SMALL).eval()
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
            restored.load_state_dict(saved["model"])
            restored_optimizer.load_state_dict(saved["optimizer"])
            restored_assignments = random.Random()
            restore_rng(saved["rng"], restored_assignments)
            actual_rng = (random.random(), np.random.rand(), torch.rand(3), restored_assignments.random())
            self.assertEqual(expected_rng[:2], actual_rng[:2])
            torch.testing.assert_close(expected_rng[2], actual_rng[2], rtol=0, atol=0)
            self.assertEqual(expected_rng[3], actual_rng[3])
            actual_losses = update(restored, copy.deepcopy(self.episodes), restored_optimizer, sequence_length=0)
            self.assertEqual(expected_losses, actual_losses)
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(saved["counters"]["decisions"], 30)
            self.assertEqual(saved["recent_passes"]["test"], [False, True])


class ExplicitConfigurationTests(unittest.TestCase):
    def test_two_models_keep_independent_configs_and_critic_dimensions(self):
        model = GameplayModelV1(SMALL)
        other = GameplayModelV1({**SMALL, "width": 64, "critic_width": 48})
        self.assertEqual(model.config["width"], 32)
        self.assertEqual(other.config["width"], 64)
        self.assertEqual(model.privileged_critic[0].out_features, 24)
        self.assertEqual(MODEL_CONFIG["width"], 192)
        with self.assertRaisesRegex(ValueError, "divisible"):
            GameplayModelV1({**SMALL, "width": 33})

    def test_undiscounted_return_and_truncation_bootstrap(self):
        episode = {"bootstrap_value": 0.4, "transitions": [
            {"action_duration_ticks": 0, "reward": 0, "value": 0.1},
            {"action_duration_ticks": 65000, "reward": 0, "value": 0.2}]}
        add_advantages([episode], gae_lambda=1.0, gamma=1.0)
        for step in episode["transitions"]:
            self.assertAlmostEqual(step["return"], 0.4)


if __name__ == "__main__":
    unittest.main()
