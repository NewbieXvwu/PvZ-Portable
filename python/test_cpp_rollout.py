import os
import json
from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
import sys
from itertools import product

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import RolloutNet, act, export_student, features, load_student
from pvz_value_net import observation_context
from pvz_student_observation import StudentHistory
from render_episode import _frame
import search_play

RESOURCE = Path(os.environ.get("PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))
BINARY = Path(os.environ.get("PVZ_EXECUTABLE", ROOT / "build/pvz-portable"))


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / "main.pak").is_file(), "requires simulator and game resources")

def _build_up(obs: dict) -> dict:
    """A deterministic action stream that puts plants on the board: the first legal placement, else wait."""
    legal = obs["legal_actions"]["plants"]
    return {"type": "plant", **legal[0]} if legal else {"type": "wait", "ticks": 60}

class CppRolloutTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.net = RolloutNet(width=13).eval()
        self.path = Path(self.directory.name) / "student with spaces.bin"
        export_student(self.net, self.path)
        self.env = EventWaitEnv(RESOURCE, executable=BINARY)
        self.addCleanup(self.env.close)
        self.env.load_rollout_net(self.path)
        self.obs, _ = self.env.reset(task=TaskSpec(level=7, seed=30000))

    def test_logits_width_and_checkpoint_export(self):
        maximum_error = 0.0
        for pool, with_context, multiplier, version, row_pool in product(("mean", "meanmax"), (False, True), (1.3, 1.5), ("v1", "v2lite", "v2"), (False, True)):
            with self.subTest(pool=pool, context=with_context, multiplier=multiplier, obs=version, row_pool=row_pool):
                self.net = RolloutNet(width=64 if row_pool else 13, pool=pool, context=with_context, obs=version,
                                      row_pool=row_pool).eval()
                weights = dict(self.net.state_dict())
                weights["value.weight"] = torch.zeros(1, self.net.plant.in_features)
                checkpoint = Path(self.directory.name) / "fast.pt"
                torch.save({"state_dict": weights, "pool": pool, "context": with_context, "obs": version,
                            "row_pool": row_pool}, checkpoint)
                export_student(load_student(checkpoint), self.path)
                self.env.load_rollout_net(self.path)
                self.obs, _ = self.env.reset(task=TaskSpec(level=7, seed=30000, zombie_count_multiplier=multiplier))
                if row_pool:
                    while self.obs["tick"] < 6000:
                        self.obs, *_ = self.env.step(_build_up(self.obs))
                task_context = observation_context(self.obs) if with_context else None
                previous = None
                previous_action = None
                history = StudentHistory(version) if version != "v1" else None
                for _ in range(8):
                    frame = _frame(self.obs)
                    frame["action"] = previous_action
                    native = self.env.rollout_predict(previous["tick"] if previous else None,
                                                       bool(previous_action and previous_action["type"] == "plant"),
                                                       student_history=history)
                    board, scalars = (history.features(frame, previous, task_context) if history is not None
                                      else features(frame, previous, task_context))
                    with torch.inference_mode():
                        hidden, context = self.net.encode(board[None], scalars[None])
                        torch.testing.assert_close(self.net.plant(context).flatten(), torch.tensor([native["plant"]]), rtol=0, atol=1e-5)
                        torch.testing.assert_close(self.net.packet(context)[0], torch.tensor(native["packet"]), rtol=0, atol=1e-5)
                        self.assertEqual(int(self.net.packet(context)[0].argmax()), int(np.argmax(native["packet"])))
                        maximum_error = max(maximum_error,
                                            float((self.net.plant(context).flatten() - torch.tensor([native["plant"]])).abs().max()),
                                            float((self.net.packet(context)[0] - torch.tensor(native["packet"])).abs().max()))
                        if native["cell_packet"] >= 0:
                            cells = self.net.cell_logits(hidden, context, torch.tensor([native["cell_packet"]]))[0]
                            torch.testing.assert_close(cells, torch.tensor(native["cell"]), rtol=0, atol=1e-5)
                            self.assertEqual(int(cells.argmax()), int(np.argmax(native["cell"])))
                            maximum_error = max(maximum_error, float((cells - torch.tensor(native["cell"])).abs().max()))
                    action = act(self.net, frame, previous, self.obs["legal_actions"]["plants"], context=task_context, history=history)
                    self.assertEqual(action, native["action"])
                    previous, previous_action = frame, action
                    self.obs, *_ = self.env.step(action)
        print(json.dumps({"rollout_max_abs_error": maximum_error, "configurations": 48}), flush=True)

    def test_v2lite_continuation_features_and_root_history(self):
        self._continuation_features("v2lite")

    def test_v2_continuation_features_and_root_history(self):
        self._continuation_features("v2")

    def _continuation_features(self, version):
        model = ROOT / "artifacts/v2lite_cpp/hf/transfer/d17v2lw2s1_model.pt"
        net = load_student(model) if model.is_file() and version == "v2lite" else RolloutNet(width=13, obs=version).eval()
        export_student(net, self.path)
        self.env.load_rollout_net(self.path)
        history = StudentHistory(version)
        while self.obs["tick"] < 6000:
            history.extra(_frame(self.obs))
            self.obs, *_ = self.env.step(_build_up(self.obs))
        history.extra(_frame(self.obs))
        self.assertEqual(history.wave, self.obs["wave"])
        context = observation_context(self.obs) if net.context else None
        root = self.env.snapshot()
        self.addCleanup(self.env.release_snapshot, root)
        search_play._W.clear()
        search_play._W.update(fast=net, context=context, student_history=history, batch_waits=True, rollout_sim_skip=0)
        self.addCleanup(search_play._W.clear)
        compared = bits = 0
        for seed in (12345, 98765):
            first = {"type": "wait", "ticks": 60, "until": "timeout"}
            trace = Path(self.directory.name) / "features.jsonl"
            until = self.obs["tick"] + 3000 if seed == 12345 else 0
            actual = self.env.rollout(first, until, snapshot_id=root, seed=seed,
                                      student_history=history, trace_path=trace)
            replay = deepcopy(history)
            for line in trace.read_text().splitlines():
                record = json.loads(line)
                frame = _frame(record["observation"])
                frame["sun_income_rate"] = record["income"]
                frame["action"] = {"type": "plant" if record["previous_plant"] else "wait"}
                previous = {"tick": record["previous_tick"]} if record["previous_tick"] >= 0 else None
                board, scalars = replay.features(frame, previous, context)
                for expected, encoded in ((board.flatten(), record["board"]), (scalars, record["scalars"])):
                    np.testing.assert_array_equal(expected.numpy().view(np.uint32), np.array(encoded, np.float32).view(np.uint32))
                    bits += expected.numel()
                compared += 1
            if until:
                expected = search_play._python_rollout_reference(self.env, self.obs, first, until, root, seed)
                self.assertEqual(_frame(actual), _frame(expected))
        self.assertGreater(compared, 50)
        print(json.dumps({"obs": version, "trajectory_decisions": compared, "equal_float32": bits}), flush=True)

    def test_rollout_leaf_income_restore_and_rng(self):
        self.obs, *_ = self.env.step({"type": "wait", "ticks": 6000})
        root = self.env.snapshot()
        self.addCleanup(self.env.release_snapshot, root)
        search_play._W.clear()
        search_play._W.update(fast=self.net, lean_rollouts=True, batch_waits=True, rollout_sim_skip=21)
        self.addCleanup(search_play._W.clear)
        first = {"type": "wait", "ticks": 60, "until": "timeout"}
        expected = search_play._python_rollout_reference(self.env, self.obs, first, self.obs["tick"] + 3000, root, 12345)
        expected_frame = _frame(expected)
        expected_history = self.env._sun_production_history.copy()
        actual = search_play._rollout_policy(self.env, self.obs, first, self.obs["tick"] + 3000, root, 12345)
        self.assertEqual(_frame(actual), expected_frame)
        self.assertEqual(self.env._sun_production_history, expected_history)
        restored = self.env.restore(root)
        self.assertEqual(_frame(restored), _frame(self.obs))


    def test_shovel_first_action_matches_the_python_loop(self):
        """The search may start a continuation with a shovel; C++ and the Python loop must agree."""
        for _ in range(200):
            if self.obs["legal_actions"]["shovels"]:
                break
            self.obs, *_ = self.env.step(_build_up(self.obs))
        self.assertTrue(self.obs["legal_actions"]["shovels"], "no plant appeared to shovel")
        col, row = self.obs["legal_actions"]["shovels"][0]
        first = {"type": "shovel", "col": int(col), "row": int(row)}
        root = self.env.snapshot()
        self.addCleanup(self.env.release_snapshot, root)

        before = {tuple(p[:2]) for p in _frame(self.obs)["plants"]}
        branch = self.env.branch_snapshot(root, [first], retain_snapshots=False)[0]
        self.assertTrue(branch["ok"])
        after = {tuple(p[:2]) for p in _frame(branch["observation"])["plants"]}
        self.assertIn((row, col), before)
        self.assertNotIn((row, col), after)
        self.assertEqual(after, before - {(row, col)})

        search_play._W.clear()
        search_play._W.update(fast=self.net, lean_rollouts=True, batch_waits=True, rollout_sim_skip=0)
        self.addCleanup(search_play._W.clear)
        until = self.obs["tick"] + 1200
        expected = search_play._python_rollout_reference(self.env, self.obs, first, until, root, 4242)
        actual = search_play._rollout_policy(self.env, self.obs, first, until, root, 4242)
        self.assertEqual(_frame(actual), _frame(expected))


if __name__ == "__main__":
    unittest.main()
