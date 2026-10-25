"""The compact row store must hand the trainer bit-identical batches (task32).

The store is the only representation now, so what has to be pinned down is the invariant it replaced:
the rows a batch sees are the rows the frames said. Two tiny synthetic games -- constant lanes so the
per-archive plane has something to hoist, a zombie walking across the lawn, cooldown progress that is
not representable in float16 (500/750), -0.0 in a cell -- and the same rows loaded with and without the
feature cache must agree bit for bit.
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
from pack_training_rows import packed_load  # noqa: E402
from train_rollout_policy import (CandidateStore, CompactBoards, PacketStore, ValueStore,
                                  constant_plane, load)  # noqa: E402


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


class CompactBoardsTest(unittest.TestCase):
    def test_round_trip_is_bit_exact(self):
        rng = np.random.default_rng(0)
        boards = rng.normal(size=(5, 4, 3)).astype(np.float16)
        boards[boards > 0.5] = 0                      # mostly zeros, like a real board
        boards[0, 0, 0] = np.float16(-0.0)            # a stored zero whose sign matters
        boards[1, 2, 1] = np.float16(np.nan)
        store = CompactBoards(boards.shape[1:])
        store.append(boards)
        store.finish()
        out = store.batch(np.arange(5), "cpu")
        self.assertTrue(torch.equal(out.to(torch.float16).view(torch.int16),
                                    torch.from_numpy(boards).view(torch.int16)))

    def test_constant_plane_is_hoisted_and_written_back(self):
        rng = np.random.default_rng(1)
        boards = np.zeros((6, 10, 10), dtype=np.float16)
        boards[:, 0, :] = np.float16(0.75)            # the "lane" channel: constant inside the archive
        boards[:, 1:, :] = rng.normal(size=(6, 9, 10)).astype(np.float16)
        plane, covered = constant_plane(boards)
        self.assertIsNotNone(plane)
        plain = CompactBoards(boards.shape[1:])
        plain.append(boards)
        plain.finish()
        hoisted = CompactBoards(boards.shape[1:])
        hoisted.append(boards, covered=covered, plane=plane)
        hoisted.finish()
        self.assertEqual(len(hoisted.planes), 1)
        self.assertLess(hoisted.nnz, plain.nnz)
        self.assertLess(hoisted.nbytes, plain.nbytes)
        self.assertTrue(torch.equal(hoisted.batch(np.arange(6), "cpu").to(torch.float16).view(torch.int16),
                                    torch.from_numpy(boards).view(torch.int16)))

    def test_packet_columns_widen_back_exactly(self):
        store = PacketStore(3, 4)
        block = np.array([[[1, 1, 500 / 750, 0]] * 4,
                          [[2, 0, 0.25, 1]] * 4,
                          [[-1, 0, 0.0, 1]] * 4], dtype=np.float32)
        store.set(0, 3, block)
        store.finish(3)
        self.assertTrue(torch.equal(store[torch.arange(3)], torch.from_numpy(block)))
        self.assertEqual(store.kinds(torch.tensor([0, 2]), torch.tensor([0, 3])).tolist(), [1, -1])


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

    def load(self, **extra):
        return packed_load(self.data, {"fast"}, cache_dir=self.cache, cells=True, packet_head="identity",
                           obs="v2", wait_keep=.5, rebuild=True, **extra)

    @staticmethod
    def dense(value, rows):
        return value.batch(torch.arange(rows), "cpu") if hasattr(value, "batch") else value

    def test_pack_reproduces_the_archives(self):
        """Every array of the pack equals a fresh build from the archives (the converter's own check)."""
        from pack_training_rows import archive_fingerprint, pack_options, pack_path, verify
        loaded = self.load()
        self.assertGreater(len(loaded[2]), 40)
        options = pack_options({"fast"}, .5, "v2", True, 0, "identity", False, None,
                               sources=archive_fingerprint(self.data, {"fast"}))
        verify(self.data, options, pack_path(self.data, options))

    def test_demo_start_archives_pack_under_their_future_seed(self):
        """Demo-start games live in seed_<demo>_f<future>; packing them used to crash on int('f44300000')."""
        path = self.data / "seed_20261032_f44300000" / "fast" / "frames.jsonl"
        path.parent.mkdir(parents=True)
        write_frames(path, game_frames(60))
        loaded = self.load()
        self.assertIn(44300000, set(loaded[6].tolist()))

    def test_source_row_counts_take_demo_start_directories(self):
        """Round 13 training then crashed in SourceRowCounts on the same directory names."""
        from pvz_training_rows import SourceRowCounts
        counts = SourceRowCounts()
        counts.observe(Path("data/seed_20261032_f44300000/fast/frames.npz"), [0, 1, 2])
        counts.observe(Path("data/seed_7/fast/frames.npz"), [0])
        self.assertEqual(counts.sources["fast"]["seeds"], {44300000: 3, 7: 1})

    def test_the_store_hoists_the_constant_plane(self):
        rows = self.load()
        self.assertGreater(len(rows[0].planes), 0)
        self.assertIsInstance(rows[9], PacketStore)
        # a dense board would be 92*6*10 float16 a row; the CSR has to be a small multiple of the nonzeros
        self.assertLess(rows[0].nbytes / len(rows[2]), 600)


if __name__ == "__main__":
    unittest.main()
