import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pvz_env import TaskSpec
from pvz_event_env import EventWaitEnv
from render_episode import _frame
from train_rollout_policy import CandidateStore, CompactBoards, ValueStore, load
from pvz_frame_store import cache_episode, frame_paths, read_frames, write_frames
from value_ranking_check import _save_dump

RESOURCE = Path(os.environ.get("PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))
BINARY = Path(os.environ.get("PVZ_EXECUTABLE", ROOT / "build/pvz-portable"))


@unittest.skipUnless(BINARY.is_file() and (RESOURCE / "main.pak").is_file(), "requires simulator and game resources")
class PipelinePerfTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.env = EventWaitEnv(RESOURCE, executable=BINARY)
        self.addCleanup(self.env.close)
        self.obs, _ = self.env.reset(task=TaskSpec(level=7, seed=30000))

    def test_observe_only_branches(self):
        self.obs, *_ = self.env.step({"type": "wait", "ticks": 6000})
        root = self.env.snapshot()
        actions = [{"type": "plant", **action} for action in self.obs["legal_actions"]["plants"][:12]]
        actions.append({"type": "wait", "ticks": 60})
        full = self.env.branch_snapshot(root, actions)
        for record in full:
            if record.get("snapshot_id"):
                self.env.release_snapshot(record["snapshot_id"])
        self.env.restore(root)
        history = self.env._sun_production_history.copy()
        compact = self.env.branch_snapshot(root, actions, retain_snapshots=False)
        for expected, actual in zip(full, compact):
            self.assertEqual(expected["ok"], actual["ok"])
            self.assertEqual(expected["events"], actual["events"])
            self.assertEqual(_frame(expected["observation"]), _frame(actual["observation"]))
            self.assertNotIn("snapshot_id", actual)
        self.assertEqual(history, self.env._sun_production_history)
        self.assertEqual(_frame(self.env.restore(root)), _frame(self.obs))
        self.env.release_snapshot(root)

    def assert_rows_equal(self, expected, actual):
        for left, right in zip(expected, actual):
            if isinstance(left, (CandidateStore, ValueStore)):
                left = left[torch.arange(len(left.counts))]
            if isinstance(right, (CandidateStore, ValueStore)):
                right = right[torch.arange(len(right.counts))]
            if isinstance(left, CompactBoards):
                left = left.batch(torch.arange(left.shape[0]), "cpu").to(right.dtype if isinstance(right, torch.Tensor) else torch.half)
            if isinstance(right, CompactBoards):
                right = right.batch(torch.arange(right.shape[0]), "cpu").to(left.dtype if isinstance(left, torch.Tensor) else torch.half)
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
            else:
                self.assertEqual(left, right)

    def test_incremental_cache_and_dump(self):
        frames = [_frame(self.obs)]
        for decision in range(80):
            legal = self.obs["legal_actions"]["plants"]
            action = ({"type": "plant", **legal[0]} if legal and decision % 3 == 0
                      else {"type": "wait", "ticks": 60})
            self.obs, *_rest, info = self.env.step(action)
            frames.append(dict(_frame(self.obs), action=action, action_rejected=not info["ok"]))
            if decision % 7 == 0:
                frames[-1]["label"] = {"type": "wait", "ticks": 60}
        source = self.directory / "data/seed_30000/teacher/frames.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text("".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames))
        cache = self.directory / "cache"
        for fraction in (.1, .5, 1.0):
            expected = load(self.directory / "data", {"teacher"}, fraction, {"teacher": 2.5})
            for _ in range(2):
                actual = load(self.directory / "data", {"teacher"}, fraction, {"teacher": 2.5}, cache)
                self.assert_rows_equal(expected, actual)
        cached = next(cache.glob("*.npz"))
        modified = cached.stat().st_mtime_ns
        extra = self.directory / "data/seed_29999/teacher/frames.jsonl"
        extra.parent.mkdir(parents=True)
        extra.write_bytes(source.read_bytes())
        self.assert_rows_equal(load(self.directory / "data", {"teacher"}, .3),
                               load(self.directory / "data", {"teacher"}, .3, cache_dir=cache))
        self.assertEqual(modified, cached.stat().st_mtime_ns)
        frames[0]["sun"] += 25
        source.write_text("".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames))
        self.assert_rows_equal(load(self.directory / "data", {"teacher"}),
                               load(self.directory / "data", {"teacher"}, cache_dir=cache))
        self.assertNotEqual(modified, cached.stat().st_mtime_ns)
        frames[-2]["targets"] = [[-1, -1, -1, .25], [0, 0, 0, .5], [0, 1, 1, .4]]
        source.write_text("".join(json.dumps(frame, separators=(",", ":")) + "\n" for frame in frames))
        expected = load(self.directory / "data", {"teacher"}, .3)
        binary = source.with_suffix(".npz")
        write_frames(binary, frames)
        self.assertEqual(read_frames(source), read_frames(binary))
        self.assertEqual(2, len(frame_paths(self.directory / "data")))
        self.assertEqual(binary, cache_episode(binary, cache))
        self.assert_rows_equal(expected, load(self.directory / "data", {"teacher"}, .3, cache_dir=cache))
        self.assert_rows_equal(expected, load(self.directory / "data", {"teacher"}, .3))
        import numpy as np
        with np.load(binary, allow_pickle=False) as stored:
            arrays = dict(stored)
        arrays["encoder"] = "old encoder"
        np.savez_compressed(binary, **arrays)
        self.assertNotEqual(binary, cache_episode(binary, cache))
        self.assert_rows_equal(expected, load(self.directory / "data", {"teacher"}, .3, cache_dir=cache))
        boards, scalars = actual[:2]
        boards = boards.batch(torch.arange(boards.shape[0]), "cpu").half()
        scalars = scalars.batch(torch.arange(scalars.shape[0]), "cpu")
        rows = [(board.numpy(), scalar.numpy(), float(index % 2), 30000, index // 4, index % 4, 0, index % 2)
                for index, (board, scalar) in enumerate(zip(boards, scalars))]
        shard = self.directory / "dump/seed_30000.pt"
        _save_dump(rows, shard, {})
        stored = torch.load(shard, weights_only=False, mmap=True)
        self.assertTrue(torch.equal(boards, stored["boards"]))
        self.assertTrue(torch.equal(scalars, stored["scalars"]))
        self.assertEqual([row[2] for row in rows], stored["won"].tolist())
        write_frames(source, frames[:3])
        self.assertFalse(binary.exists())
        self.assertEqual(json.loads(json.dumps(frames[:3])), read_frames(source))


if __name__ == "__main__":
    unittest.main()
