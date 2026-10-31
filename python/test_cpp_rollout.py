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
from pvz_tasks import profile_for_deck
from pvz_event_env import EventWaitEnv
from pvz_rollout_policy import RolloutNet, act, export_student, features, load_student
from pvz_value_net import observation_context
from pvz_student_observation import StudentHistory
from render_episode import _frame
import search_play



RESOURCE = Path(os.environ.get("PVZ_RESOURCE_DIR", str(Path.home() / ".cache/pvz-research-resources")))
BINARY = Path(os.environ.get("PVZ_EXECUTABLE", ROOT / "build/pvz-portable"))


def _build_up(obs: dict) -> dict:
    """A deterministic action stream that puts plants on the board: the first legal placement, else wait."""
    legal = obs["legal_actions"]["plants"]
    return {"type": "plant", **legal[0]} if legal else {"type": "wait", "ticks": 60}

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

    def test_v2timer_continuation_features(self):
        # task147: observation 9 = v2sleep + visible timers and the seed belt.
        self._continuation_features("v2timer")

    def _timer_channels(self, trace):
        """Native board vs Python timer channels for the current public frame."""
        self.env.rollout_predict(-1, False, trace_path=trace)
        record = json.loads(Path(trace).read_text())
        native = np.array(record["board"], np.float32).reshape(-1, 6, 10)
        board, _ = StudentHistory("v2timer").features(_frame(record["observation"]), None, None)
        return board, native

    def test_v2timer_visible_timers_and_belt_on_a_constructed_board(self):
        """Doom shroom / coffee bean / chomper countdowns and the seed belt, Python == C++."""
        net = RolloutNet(width=8, obs="v2timer").eval()
        export_student(net, self.path)
        self.env.load_rollout_net(self.path)
        deck = (0, 4, 6, 8, 15, 17, 35)
        task = TaskSpec(level=1, seed=424242, profile=profile_for_deck(deck),
                        preplanted=((6, 2, 6), (17, 3, 6), (8, 0, 5)))
        self.obs, _ = self.env.reset(deck=deck, task=task)
        trace = Path(self.directory.name) / "timer_features.jsonl"
        seen = {"special": set(), "state": 0, "belt": 0}
        planted = set()
        # Level 1: the first zombie reaches the preplanted chomper around tick 5500, so the
        # window has to cover a full bite (the chew alone is 4000 ticks).
        for _ in range(400):
            legal = self.obs["legal_actions"]["plants"]
            for action in legal:
                packet, row, col = action["packet"], action["row"], action["col"]
                kind = deck[packet] if packet < len(deck) else None
                if (kind == 15 and "doom" not in planted) or (kind == 35 and "coffee" not in planted
                                                               and row == 0 and col == 5):
                    planted.add("doom" if kind == 15 else "coffee")
                    self.obs, *_ = self.env.step({"type": "plant", **action})
                    break
            board, native = self._timer_channels(trace)
            np.testing.assert_array_equal(board[-4:].numpy().view(np.uint32), native[-4:].view(np.uint32))
            seen["special"].update(float(value) for value in board[-4].flatten() if float(value) > 0)
            seen["state"] = max(seen["state"], int((board[-3] > 0).sum()))
            seen["belt"] = max(seen["belt"], int((board[-2] > 0).sum()))
            if len(seen["special"]) >= 3 and seen["state"] and len(planted) == 2:
                break
            self.obs, *_ = self.env.step({"type": "wait", "ticks": 20})
        self.assertGreaterEqual(len(seen["special"]), 3, "explosion countdown never reached the board")
        self.assertGreater(seen["state"], 0, "no chomper/squash countdown frame was captured")
        self.assertGreater(seen["belt"], 0)

    def test_v2timer_cannon_and_fog_unknown(self):
        export_student(RolloutNet(width=8, obs="v2timer").eval(), self.path)
        self.env.load_rollout_net(self.path)
        deck = (0, 2, 6, 8, 16, 47)
        self.obs, _ = self.env.reset(deck=deck, task=TaskSpec(level=31, seed=424242,
            profile=profile_for_deck(deck), fog_observation="human_v2",
            preplanted=((34, 0, 0), (34, 0, 1), (47, 0, 0), (6, 1, 8))))
        trace = Path(self.directory.name) / "fog_timer.json"
        self.obs, *_ = self.env.step({"type": "wait", "ticks": 20})
        board, native = self._timer_channels(trace)
        np.testing.assert_array_equal(board[-4:].numpy().view(np.uint32), native[-4:].view(np.uint32))
        self.assertGreater(float(board[-3, 0, 0]), 0, "cannon setup timer was not sampled")
        self.obs, *_ = self.env.step({"type": "wait", "ticks": 1500})
        hidden = next(p for p in _frame(self.obs)['v2']['plants'] if p[:3] == [1, 8, 6])
        self.assertEqual(hidden[3:6], [-1, -1, -1])
        board, native = self._timer_channels(trace)
        np.testing.assert_array_equal(board[-4:].numpy().view(np.uint32), native[-4:].view(np.uint32))
        self.assertEqual(float(board[-4:-2, 1, 8].abs().sum()), 0)

    def test_v2timer_original_conveyor_shift(self):
        export_student(RolloutNet(width=8, obs="v2timer", packet_head="identity").eval(), self.path)
        self.env.load_rollout_net(self.path)
        self.obs, _ = self.env.reset(deck=(), task=TaskSpec(level=10, seed=30000,
            conveyor_start="original", profile=profile_for_deck(())))
        trace = Path(self.directory.name) / "conveyor_timer.json"
        for _ in range(100):
            packets = [p for p in _frame(self.obs)['packets'] if p[1] >= 0]
            board, native = self._timer_channels(trace)
            np.testing.assert_array_equal(board[-4:].numpy().view(np.uint32), native[-4:].view(np.uint32))
            legal = self.obs['legal_actions']['plants']
            if len(packets) >= 2 and any(p['packet'] == 0 for p in legal):
                action = next(p for p in legal if p['packet'] == 0)
                next_kind = packets[1][1]
                self.obs, *_ = self.env.step({'type': 'plant', **action})
                board, native = self._timer_channels(trace)
                np.testing.assert_array_equal(board[-4:].numpy().view(np.uint32), native[-4:].view(np.uint32))
                self.assertEqual(float(board[-2, 0, 0]), float(np.float32((next_kind + 1) / 48)))
                return
            self.obs, *_ = self.env.step({'type': 'wait', 'ticks': 60})
        self.fail('no legal conveyor shift was sampled')

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
        self.assertGreater(compared, 50)
        print(json.dumps({"obs": version, "trajectory_decisions": compared, "equal_float32": bits}), flush=True)



    def test_shovel_first_action_removes_the_target_plant(self):
        """A shovel branch removes its target plant."""
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



if __name__ == "__main__":
    unittest.main()
