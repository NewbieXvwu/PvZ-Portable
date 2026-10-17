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
        for pool, with_context, multiplier, version in product(("mean", "meanmax"), (False, True), (1.3, 1.5), ("v1", "v2lite")):
            with self.subTest(pool=pool, context=with_context, multiplier=multiplier):
                self.net = RolloutNet(width=13, pool=pool, context=with_context, obs=version).eval()
                weights = dict(self.net.state_dict())
                weights["value.weight"] = torch.zeros(1, self.net.plant.in_features)
                checkpoint = Path(self.directory.name) / "fast.pt"
                torch.save({"state_dict": weights, "pool": pool, "context": with_context, "obs": version}, checkpoint)
                export_student(load_student(checkpoint), self.path)
                self.env.load_rollout_net(self.path)
                self.obs, _ = self.env.reset(task=TaskSpec(level=7, seed=30000, zombie_count_multiplier=multiplier))
                task_context = observation_context(self.obs) if with_context else None
                previous = None
                previous_action = None
                history = StudentHistory() if version == "v2lite" else None
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
                        if native["cell_packet"] >= 0:
                            cells = self.net.cell_logits(hidden, context, torch.tensor([native["cell_packet"]]))[0]
                            torch.testing.assert_close(cells, torch.tensor(native["cell"]), rtol=0, atol=1e-5)
                    action = act(self.net, frame, previous, self.obs["legal_actions"]["plants"], context=task_context, history=history)
                    self.assertEqual(action, native["action"])
                    previous, previous_action = frame, action
                    self.obs, *_ = self.env.step(action)

    def test_v2lite_continuation_features_and_root_history(self):
        from scripted_baseline import choose
        model = ROOT / "artifacts/v2lite_cpp/hf/transfer/d17v2lw2s1_model.pt"
        net = load_student(model) if model.is_file() else RolloutNet(width=13, obs="v2lite").eval()
        export_student(net, self.path)
        self.env.load_rollout_net(self.path)
        history = StudentHistory()
        while self.obs["tick"] < 6000:
            history.extra(_frame(self.obs))
            self.obs, *_ = self.env.step(choose(self.obs))
        history.extra(_frame(self.obs))
        self.assertTrue(history.plants)
        context = observation_context(self.obs) if net.context else None
        root = self.env.snapshot()
        self.addCleanup(self.env.release_snapshot, root)
        search_play._W.clear()
        search_play._W.update(fast=net, context=context, student_history=history, batch_waits=True, rollout_sim_skip=0)
        self.addCleanup(search_play._W.clear)
        compared = 0
        for seed in (12345, 98765):
            first = {"type": "wait", "ticks": 60, "until": "timeout"}
            trace = Path(self.directory.name) / "features.jsonl"
            actual = self.env.rollout(first, self.obs["tick"] + 3000, snapshot_id=root, seed=seed,
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
                compared += 1
            expected = search_play._rollout_policy(self.env, self.obs, first, self.obs["tick"] + 3000, root, seed)
            self.assertEqual(_frame(actual), _frame(expected))
        self.assertGreater(compared, 50)

    def test_rollout_leaf_income_restore_and_rng(self):
        self.obs, *_ = self.env.step({"type": "wait", "ticks": 6000})
        root = self.env.snapshot()
        self.addCleanup(self.env.release_snapshot, root)
        search_play._W.clear()
        search_play._W.update(fast=self.net, lean_rollouts=True, batch_waits=True, rollout_sim_skip=21)
        self.addCleanup(search_play._W.clear)
        first = {"type": "wait", "ticks": 60, "until": "timeout"}
        expected = search_play._rollout_policy(self.env, self.obs, first, self.obs["tick"] + 3000, root, 12345)
        expected_frame = _frame(expected)
        expected_history = self.env._sun_production_history.copy()
        search_play._W["cpp_rollouts"] = True
        actual = search_play._rollout_policy(self.env, self.obs, first, self.obs["tick"] + 3000, root, 12345)
        self.assertEqual(_frame(actual), expected_frame)
        self.assertEqual(self.env._sun_production_history, expected_history)
        restored = self.env.restore(root)
        self.assertEqual(_frame(restored), _frame(self.obs))


if __name__ == "__main__":
    unittest.main()
