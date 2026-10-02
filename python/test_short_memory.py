"""Finite history, causal replay, current-parameter gradients and continuation."""
from __future__ import annotations

import copy
import random
import unittest

import torch

from pvz_agent_model import (GameplayModelV1, model_architecture_version,
    observation_tokens, pack_tokens, policy_legal_summary, replay_log_probs, select_action)
from pvz_research import capture_rng, restore_rng
from test_event_wait_policy import wait_result
from test_observation_context import SMALL, source
from train_pvz_ppo import add_advantages, train_update


def config(history=8):
    return {**SMALL, "wait_mode": "events", "wait_mask": "progress_v1",
            "short_history": {"history_decisions": history, "projection_width": 8,
                              "mlp_width": 24, "mlp_layers": 2}}


def fixture(model, lengths=(13, 10), first_sun=None):
    episodes = []
    for job, length in enumerate(lengths):
        transitions, hidden, previous, result, delta = [], None, None, None, 0
        for index in range(length):
            obs = source(); obs.update(tick=index*150, wave=1+index//4)
            if index == 0 and first_sun is not None:
                obs["sun"] = first_sun
            tensors, metadata = observation_tokens(obs, 7)
            legal = policy_legal_summary(obs, model.config)
            action = dict(type="wait", ticks=150, until="timeout")
            with torch.no_grad():
                output = model.step_tokens(tensors, metadata, obs["wave"], hidden, previous, delta, {}, result)
                _, lp, _ = select_action(model, output, legal, action=action)
                value = model.privileged_value_from_extra(output, [0.]*16)
            wait = wait_result(action, 150)
            transitions.append(dict(tokens=pack_tokens(tensors, metadata), wave=obs["wave"],
                legal=legal, action=action, previous_action=previous, previous_wait_result=result,
                wait_result=wait, elapsed_since_previous_observation=delta,
                action_duration_ticks=150, events={}, log_prob=lp.item(), value=value.item(),
                critic_extra=[0.]*16, reward=1. if index == length-1 else 0.))
            hidden, previous, result, delta = output["hidden"], action, wait, 150
        episodes.append(dict(transitions=transitions, terminated=True, truncated=False, won=True))
    add_advantages(episodes, .95, .99)
    return episodes


def update(model, episodes, optimizer):
    return train_update(model, episodes, optimizer, torch.device("cpu"), 1, 3, .2, .5, .01, 2, "dense")


class ShortHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(0); random.seed(0)

    def test_explicit_configuration_rejects_ambiguous_or_invalid_memory(self):
        self.assertEqual(model_architecture_version(config()), 15)
        for changes in ({"history_decisions": 0}, {"projection_width": True}, {"mlp_layers": -1}):
            cfg=config(); cfg["short_history"].update(changes)
            with self.assertRaises(ValueError): GameplayModelV1(cfg)
        cfg=config(); cfg["slow_memory"]={}
        with self.assertRaisesRegex(ValueError, "cannot also"): GameplayModelV1(cfg)
        model=GameplayModelV1(config())
        self.assertFalse(any(isinstance(m, (torch.nn.GRU, torch.nn.LSTM)) for m in model.modules()))
        self.assertFalse(any(name.startswith("aux_") for name in model.state_dict()))

    def test_ragged_replay_matches_step_collection_for_one_and_eight_frames(self):
        for history in (1, 8):
            model=GameplayModelV1(config(history)).eval(); episodes=fixture(model)
            sequences=[e["transitions"] for e in episodes]
            with torch.no_grad():
                outputs, hidden=model.forward_sequences(sequences, [None]*2)
                lp,_=replay_log_probs(model,outputs,[t for seq in sequences for t in seq])
            expected=torch.tensor([t["log_prob"] for seq in sequences for t in seq])
            self.assertLess((lp-expected).abs().max().item(), 1e-6)
            self.assertEqual(tuple(hidden.shape), (history-1, 2, 8))

    def test_old_observation_changes_policy_only_inside_explicit_history(self):
        model=GameplayModelV1(config(4)).eval()
        first=fixture(model,lengths=(9,),first_sun=50)[0]["transitions"]
        second=fixture(model,lengths=(9,),first_sun=900)[0]["transitions"]
        with torch.no_grad():
            a, ah=model.forward_sequences([first],[None]); b, bh=model.forward_sequences([second],[None])
        self.assertGreater((a[3]["type_logits"]-b[3]["type_logits"]).abs().max().item(),1e-8)
        for index in range(4,9):
            torch.testing.assert_close(a[index]["type_logits"],b[index]["type_logits"],rtol=0,atol=0)
        torch.testing.assert_close(ah,bh,rtol=0,atol=0)

    def test_chunked_history_has_exact_same_policy_as_whole_sequence(self):
        model=GameplayModelV1(config()).eval(); sequence=fixture(model,lengths=(13,))[0]["transitions"]
        with torch.no_grad():
            full,_=model.forward_sequences([sequence],[None]); pieces=[];hidden=None
            for start in range(0, len(sequence), 3):
                out,h=model.forward_sequences([sequence[start:start+3]],[hidden]);pieces.extend(out);hidden=h[:,0]
        torch.testing.assert_close(torch.stack([o["type_logits"] for o in full]),
                                   torch.stack([o["type_logits"] for o in pieces]),rtol=1e-6,atol=1e-7)

    def test_zero_lr_ppo_replays_every_boundary_without_full_prefix_or_duplicate_loss(self):
        model=GameplayModelV1(config()).eval(); episodes=fixture(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0)
        records=[]; original=model.forward_sequences
        def recorded(sequences,hiddens):
            if torch.is_grad_enabled(): records.append(sum(len(s) for s in sequences))
            else: self.fail("Finite history should not rebuild a detached earlier prefix")
            return original(sequences,hiddens)
        model.forward_sequences=recorded
        losses=update(model,episodes,optimizer)
        self.assertLess(losses["max_log_prob_change"],1e-6)
        self.assertEqual(losses["optimizer_steps"],5)
        self.assertGreater(sum(records),23)

    def test_core_actor_loss_trains_the_prior_public_history_projection(self):
        model=GameplayModelV1(config()).eval(); sequence=fixture(model,lengths=(10,))[0]["transitions"]
        outputs,_=model.forward_sequences([sequence[:8]],[None])
        lp,_=replay_log_probs(model,[outputs[-1]],[sequence[7]])
        (-lp.mean()).backward()
        self.assertGreater(model.belief.history_projection[0].weight.grad.abs().sum().item(),0.)

    def test_model_optimizer_and_rng_restore_identical_next_real_update(self):
        model=GameplayModelV1(config()).eval(); episodes=fixture(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4); update(model,episodes,optimizer)
        assignments=random.Random(91)
        saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(assignments)))
        expected=update(model,copy.deepcopy(episodes),optimizer)
        restored=GameplayModelV1(config()).eval(); restored.load_state_dict(saved["model"])
        other=torch.optim.AdamW(restored.parameters(),lr=.01); other.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"],random.Random())
        actual=update(restored,copy.deepcopy(episodes),other)
        self.assertEqual(actual,expected)
        for key,value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)


if __name__ == "__main__": unittest.main()
