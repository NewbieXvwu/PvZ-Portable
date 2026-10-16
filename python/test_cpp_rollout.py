import os
import tempfile
import unittest
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import RolloutNet, act, export_student, features, load_student
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
        weights = dict(self.net.state_dict())
        weights["value.weight"] = torch.zeros(1, 26)
        checkpoint = Path(self.directory.name) / "fast.pt"
        torch.save({"state_dict": weights}, checkpoint)
        export_student(load_student(checkpoint), self.path)
        self.env.load_rollout_net(self.path)
        previous = None
        previous_action = None
        for _ in range(8):
            frame = _frame(self.obs)
            frame["action"] = previous_action
            native = self.env.rollout_predict(previous["tick"] if previous else None,
                                               bool(previous_action and previous_action["type"] == "plant"))
            board, scalars = features(frame, previous)
            with torch.inference_mode():
                hidden, context = self.net.encode(board[None], scalars[None])
                torch.testing.assert_close(self.net.plant(context).flatten(), torch.tensor([native["plant"]]), rtol=0, atol=1e-5)
                torch.testing.assert_close(self.net.packet(context)[0], torch.tensor(native["packet"]), rtol=0, atol=1e-5)
                if native["cell_packet"] >= 0:
                    cells = self.net.cell_logits(hidden, context, torch.tensor([native["cell_packet"]]))[0]
                    torch.testing.assert_close(cells, torch.tensor(native["cell"]), rtol=0, atol=1e-5)
            action = act(self.net, frame, previous, self.obs["legal_actions"]["plants"])
            self.assertEqual(action, native["action"])
            previous, previous_action = frame, action
            self.obs, *_ = self.env.step(action)

    def test_rollout_leaf_income_restore_and_rng(self):
        self.obs, *_ = self.env.step({"type": "wait", "ticks": 6000})
        root = self.env.snapshot()
        self.addCleanup(self.env.release_snapshot, root)
        search_play._W.clear()
        search_play._W.update(fast=self.net, lean_rollouts=True, batch_waits=True, rollout_sim_skip=21)
        self.addCleanup(search_play._W.clear)
        first = {"type": "wait", "ticks": 60, "until": "timeout"}
        expected = search_play._rollout_policy(self.env, self.obs, {}, first, self.obs["tick"] + 3000, root, 12345)
        expected_frame = _frame(expected)
        expected_history = self.env._sun_production_history.copy()
        search_play._W["cpp_rollouts"] = True
        actual = search_play._rollout_policy(self.env, self.obs, {}, first, self.obs["tick"] + 3000, root, 12345)
        self.assertEqual(_frame(actual), expected_frame)
        self.assertEqual(self.env._sun_production_history, expected_history)
        restored = self.env.restore(root)
        self.assertEqual(_frame(restored), _frame(self.obs))


if __name__ == "__main__":
    unittest.main()
