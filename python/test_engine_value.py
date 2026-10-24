"""Python oracle for the engine value encoder, forward and static branching (one game process)."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from export_value_net import export_value_net
from pvz_env import PvZEnv, TaskSpec, PlayerProfileContext
from pvz_value_net import ValueNet, encode_frame, observation_context
from render_episode import _frame

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("PVZ_TEST_RESOURCE_DIR"), "set PVZ_TEST_RESOURCE_DIR to run engine oracle")
class EngineValueTest(unittest.TestCase):
    def test_fog_value_features_static_and_rollout_match_public_observation(self):
        from pvz_tasks import deck_for_level, profile_for_deck
        from pvz_rollout_policy import export_student, RolloutNet
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, PvZEnv(
                os.environ["PVZ_TEST_RESOURCE_DIR"],
                executable=os.environ.get("PVZ_TEST_EXECUTABLE", str(ROOT / "build/pvz-portable"))) as env:
            export_student(RolloutNet(13, obs="v2").eval(), Path(tmp) / "student.bin")
            env.load_rollout_net(Path(tmp) / "student.bin")
            deck = deck_for_level(31)
            for version in (1, 2):
                torch.manual_seed(50)
                net = ValueNet(13, context=True, encoder_version=version).eval()
                export_value_net(net, Path(tmp) / "value.bin")
                env.load_value_net(Path(tmp) / "value.bin")
                env.load_value_net(Path(tmp) / "value.bin", static=True)
                for mode in ("human", "xray"):
                    env.reset(deck=deck, task=TaskSpec(level=31, seed=450031,
                              profile=profile_for_deck(deck), fog_observation=mode))
                    obs, *_ = env.step(dict(type="wait", ticks=8000))
                    if mode == "human":
                        self.assertLess(len(obs["zombies"]), len(env.privileged_state()["zombies"]))
                    context = torch.from_numpy(observation_context(obs))

                    def expected(observation):
                        b, s = encode_frame(_frame(observation), version)
                        s = torch.cat([s, context])
                        with torch.inference_mode():
                            return torch.sigmoid(net(b[None], s[None])).item(), b, s

                    probability, board, scalars = expected(obs)
                    actual = env.leaf_values(trace_path=Path(tmp) / "features.bin")[0]
                    np.testing.assert_array_equal(np.fromfile(Path(tmp) / "features.bin", dtype="<f4"),
                                                  np.concatenate([board.flatten().numpy(), scalars.numpy()]))
                    self.assertLess(abs(actual - probability), 1e-5)
                    root = env.snapshot_fast()
                    actions = [{"type": "plant", **a} for a in obs["legal_actions"]["plants"][:8]]
                    actions.append(dict(type="wait", ticks=0))
                    scores = env.static_scores(root, actions, obs["sun_income_rate"])
                    for score, branch in zip(scores, env.branch_snapshot(root, actions, retain_snapshots=False)):
                        branch_obs = branch["observation"]
                        branch_obs["sun_income_rate"] = obs["sun_income_rate"]
                        self.assertLess(abs(score - expected(branch_obs)[0]), 1e-5)
                    leaf = env.rollout(None, 11000, snapshot_id=root, seed=50, sim_skip=61)
                    probability = float(leaf["result"] == 1) if leaf["terminal"] else expected(leaf)[0]
                    scored = env.rollout(None, 11000, snapshot_id=root, seed=50, sim_skip=61, score_value=True)
                    self.assertLess(abs(scored["value"] - probability), 1e-5)
                    env.release_snapshot(root)

    def test_fast_snapshot_restores_board_clock_and_income_history(self):
        with PvZEnv(os.environ["PVZ_TEST_RESOURCE_DIR"], executable=os.environ.get(
                "PVZ_TEST_EXECUTABLE", str(ROOT / "build/pvz-portable"))) as env:
            env.reset(level=7, seed=3717)
            before, *_ = env.step(dict(type="wait", ticks=3000))
            history = list(env._sun_production_history)
            root = env.snapshot_fast()
            env.step(dict(type="wait", ticks=600))
            env.restore_fast(root)
            self.assertEqual(env._tick, before["tick"])
            self.assertEqual(list(env._sun_production_history), history)
            self.assertEqual(env.observe(), before)
            env.release_snapshot(root)
            with self.assertRaises(ValueError):
                env.restore_fast(root)

    def test_scored_rollout_matches_python_leaf_and_preserves_state(self):
        from pvz_rollout_policy import load_student, export_student
        from pvz_value_net import load_value_net
        student_path = os.environ.get("PVZ_TEST_STUDENT")
        value_path = os.environ.get("PVZ_TEST_VALUE")
        if not student_path or not value_path:
            self.skipTest("set PVZ_TEST_STUDENT/PVZ_TEST_VALUE for trained rollout oracle")
        torch.set_num_threads(1)
        net = load_value_net(value_path)
        with tempfile.TemporaryDirectory() as tmp, PvZEnv(
                os.environ["PVZ_TEST_RESOURCE_DIR"],
                executable=os.environ.get("PVZ_TEST_EXECUTABLE", str(ROOT / "build/pvz-portable"))) as env:
            export_student(load_student(student_path), Path(tmp) / "student.bin")
            export_value_net(net, Path(tmp) / "value.bin")
            env.load_rollout_net(Path(tmp) / "student.bin")
            env.load_value_net(Path(tmp) / "value.bin")
            for level in (7, 14, 21, 10):
                deck = (0, 1, 2, 3, 4, 5, 16) if level == 21 else (0, 1, 2, 3, 4, 5)
                task = TaskSpec(level=level, seed=37000+level,
                                profile=PlayerProfileContext(seed_slot_count=len(deck)),
                                conveyor_start="original" if level == 10 else None)
                env.reset(deck=() if level == 10 else deck, task=task)
                env.rollout(None, 4000)
                root = env.snapshot()
                for mask in (0, 61):
                    leaf = env.rollout(None, 9000, sim_skip=mask, snapshot_id=root, seed=1700)
                    full = env.observe()
                    if leaf["terminal"]:
                        expected = float(leaf["result"] == 1)
                    else:
                        b, s = encode_frame(_frame(leaf), net.encoder_version)
                        if net.context:
                            s = torch.cat([s, torch.from_numpy(observation_context(full))])
                        with torch.inference_mode():
                            expected = torch.sigmoid(net(b[None], s[None])).item()
                    scored = env.rollout(None, 9000, sim_skip=mask, snapshot_id=root, seed=1700, score_value=True)
                    self.assertLess(abs(scored["value"] - expected), 1e-5)
                    # Observation comparison includes dynamics, counters and legal actions.
                    # Income is wrapper state, intentionally not adopted by the score-only command.
                    after = env.observe()
                    full.pop("sun_income_rate", None); after.pop("sun_income_rate", None)
                    self.assertEqual(full, after)
                env.restore(root)
                env.release_snapshot(root)

    def test_board_logits_static_rejections_and_restore(self):
        torch.set_num_threads(1)
        maxima = {"features": 0.0, "logits": 0.0, "static": 0.0}
        with tempfile.TemporaryDirectory() as tmp, PvZEnv(
                os.environ["PVZ_TEST_RESOURCE_DIR"],
                executable=os.environ.get("PVZ_TEST_EXECUTABLE", str(ROOT / "build/pvz-portable"))) as env:
            for version in (1, 2):
                torch.manual_seed(37)
                net = ValueNet(64, context=True, encoder_version=version).eval()
                export_value_net(net, Path(tmp) / "value.bin")
                env.load_value_net(Path(tmp) / "value.bin")
                env.load_value_net(Path(tmp) / "value.bin", static=True)
                for level in (7, 14, 21, 10):
                    deck = (0, 1, 2, 3, 4, 5, 16) if level == 21 else (0, 1, 2, 3, 4, 5)
                    kwargs = dict(deck=deck, task=TaskSpec(level=level, seed=3700 + level,
                                  profile=PlayerProfileContext(seed_slot_count=len(deck))))
                    if level == 10:
                        kwargs.update(deck=(), task=TaskSpec(level=level, seed=3700 + level,
                                      conveyor_start="original"))
                    obs, _ = env.reset(**kwargs)
                    for step in range(16):
                        board, scalars = encode_frame(_frame(obs), version)
                        scalars = torch.cat([scalars, torch.from_numpy(observation_context(obs))])
                        env.leaf_values(trace_path=Path(tmp) / "trace.bin")
                        actual = np.fromfile(Path(tmp) / "trace.bin", dtype="<f4")
                        expected = np.concatenate([board.flatten().numpy(), scalars.numpy()])
                        maxima["features"] = max(maxima["features"], float(np.max(np.abs(actual - expected))))
                        np.testing.assert_array_equal(actual, expected)
                        with torch.inference_mode():
                            logit = net(board[None], scalars[None]).item()
                        actual_logit = env._last_response["logits"][0]
                        maxima["logits"] = max(maxima["logits"], abs(logit - actual_logit))
                        self.assertLess(abs(logit - actual_logit), 1e-5)
                        root = env.snapshot()
                        actions = [{"type": "plant", **a} for a in (obs.get("legal_actions") or {}).get("plants", [])[:8]]
                        actions += [{"type": "plant", "packet": 99, "row": 0, "col": 0}]
                        income = obs["sun_income_rate"]
                        scores = env.static_scores(root, actions, income)
                        self.assertIsNone(scores[-1])
                        branches = env.branch_snapshot(root, actions, retain_snapshots=False)
                        for score, branch in zip(scores, branches):
                            if not branch.get("ok"):
                                self.assertIsNone(score)
                                continue
                            frame = _frame(branch["observation"])
                            self.assertEqual(frame["tick"], obs["tick"])
                            frame["sun_income_rate"] = income
                            b, s = encode_frame(frame, version)
                            s = torch.cat([s, scalars[-35:]])
                            with torch.inference_mode():
                                expected_score = torch.sigmoid(net(b[None], s[None])).item()
                            maxima["static"] = max(maxima["static"], abs(score - expected_score))
                            self.assertLess(abs(score - expected_score), 1e-6)
                        restored = env.restore(root)
                        self.assertEqual(_frame(restored), _frame(obs))
                        env.release_snapshot(root)
                        obs, _, done, _, _ = env.step({"type": "wait", "ticks": 120})
                        if done:
                            break
        print("engine_value_oracle", json.dumps(maxima))


if __name__ == "__main__":
    unittest.main()
