"""The compact row store must hand the trainer bit-identical batches (task32).

Everything here is synthetic: two tiny games whose lanes are constant (so the per-archive plane has
something to hoist), a zombie walking across the lawn, cooldown progress that is not representable in
float16 (500/750), and search targets on some frames. The default path, the compact path and the
file-backed variants must produce the same dense boards, scalars, labels, values, candidates and packet
block -- that is the gate for calling any of them a memory-only change.
"""
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pvz_frame_store import write_frames  # noqa: E402
from train_rollout_policy import CompactBoards, PacketStore, load  # noqa: E402


def game_frames(count: int) -> list[dict]:
    frames = []
    for index in range(count):
        frames.append({
            "tick": 100 + 60 * index, "wave": 1 + index // 20, "sun": 50 + 7 * index,
            "scene": {"wave_count": 30},
            "plants": [[row, col, 4, 300 - 7 * index, 300]
                       for row in range(2) for col in range(min(3, 1 + index // 5))],
            "zombies": [[row, 3, 60.0 + 25.0 * ((index + row) % 12), .35, 200 - 5 * index, True,
                         index % 3 == 0, index % 5 == 0] for row in range(3)],
            "mowers": [[0, 1], [1, 1], [2, 2]],
            "lanes": [0, 1, 2, 3, 4],
            "packets": [[slot, slot, slot < 4, 0 if slot < 4 else 500, 750, 100] for slot in range(7)],
            "v2": {"wave_timer": 2250, "huge_wave_countdown": 0, "wave_health": 100 - index,
                   "wave_start_health": 200, "sun_countdown": 612, "sun_pending": 50,
                   "plants": [[0, 0, 4, 0, 750, 0, 0, 0]], "zombies": [[1, 0, 0], [0, 20, 0], [0, 0, 0]],
                   "projectiles": [[0, 80.0, 20 + index]]},
        })
    for index, frame in enumerate(frames):
        frame["action"] = ({"type": "plant", "packet": index % 7, "row": index % 6, "col": index % 5}
                           if index % 3 == 0 else {"type": "wait", "ticks": 60})
        frame["action_rejected"] = False
        if index % 4 == 1:
            frame["targets"] = [[index % 7, index % 6, index % 5, 0.5],
                                [(index + 2) % 7, (index + 1) % 6, (index + 3) % 5, 0.25],
                                [-1, -1, -1, 0.75]]
    return frames


class CompactLoaderTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.data = self.directory / "data"
        self.cache = self.directory / "cache"
        for seed in (100, 101):
            path = self.data / f"seed_{seed}" / "fast" / "frames.jsonl"
            path.parent.mkdir(parents=True)
            write_frames(path, game_frames(60))

    def variants(self):
        common = dict(cache_dir=self.cache, cells=True, packet_head="identity", obs="v2", wait_keep=.5)
        yield "compact", load(self.data, {"fast"}, compact=True, **common)
        yield "memmap", load(self.data, {"fast"}, memmap_dir=self.directory / "mapped", **common)
        yield "compact+memmap", load(self.data, {"fast"}, compact=True,
                                     memmap_dir=self.directory / "mapped_compact", **common)

    @staticmethod
    def dense(store, rows):
        return store.batch(torch.arange(rows), "cpu")

    @staticmethod
    def materialize(value, rows):
        return value[torch.arange(rows)] if isinstance(value, PacketStore) else value

    def test_every_variant_matches_the_default(self):
        base = load(self.data, {"fast"}, cache_dir=self.cache, cells=True, packet_head="identity",
                    obs="v2", wait_keep=.5)
        rows = len(base[2])
        self.assertGreater(rows, 40)
        expected_boards = self.dense(base[0], rows)
        expected_scalars = self.dense(base[1], rows)
        for name, other in self.variants():
            with self.subTest(variant=name):
                self.assertEqual(len(other[2]), rows)
                # int32 views also catch -0.0 versus 0.0, which assert_close would let through
                self.assertTrue(torch.equal(self.dense(other[0], rows).view(torch.int32),
                                            expected_boards.view(torch.int32)))
                self.assertTrue(torch.equal(self.dense(other[1], rows).view(torch.int32),
                                            expected_scalars.view(torch.int32)))
                for index in (2, 3, 4, 5, 7):
                    left, right = base[index], other[index]
                    torch.testing.assert_close(left.float(), right.float(), rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(base[8].float(), other[8].float(), rtol=0, atol=0, equal_nan=True)
                torch.testing.assert_close(self.materialize(base[9], rows).float(),
                                            self.materialize(other[9], rows).float(), rtol=0, atol=0, equal_nan=True)
                self.assertEqual(base[6], other[6])

    def test_compact_store_hoists_the_constant_plane(self):
        base = load(self.data, {"fast"}, cache_dir=self.cache, cells=True, packet_head="identity",
                    obs="v2", wait_keep=.5)
        compact = load(self.data, {"fast"}, cache_dir=self.cache, cells=True, packet_head="identity",
                       obs="v2", wait_keep=.5, compact=True)
        self.assertGreater(len(compact[0].planes), 0)
        self.assertLess(compact[0].nnz, base[0].nnz)
        self.assertLess(compact[0].nbytes, base[0].nbytes)
        self.assertIsInstance(compact[9], PacketStore)

    def test_compact_requires_the_cache(self):
        with self.assertRaises(ValueError):
            load(self.data, {"fast"}, compact=True, cells=True)


if __name__ == "__main__":
    unittest.main()
