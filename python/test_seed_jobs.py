from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pvz_seed_jobs import run_seed_jobs


def _initialize_worker() -> None:
    pass


def _collect_seed(seed: int) -> dict[str, object]:
    return {"seed": seed, "pid": os.getpid(), "tokens": np.arange(12, dtype=np.int16).reshape(3, 4),
            "tuple_value": (seed, seed + 1)}


def _unexpected_worker(seed: int) -> dict[str, int]:
    raise AssertionError(f"completed seed {seed} was collected again")


class SeedJobTests(unittest.TestCase):
    def test_spawn_jobs_resume_and_recollect_corrupt_shards(self) -> None:
        metadata = {"task": "test", "version": 1, "tuple_value": (3, 5)}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            collected = run_seed_jobs(
                [4, 2], directory, metadata, _collect_seed,
                workers=2, initializer=_initialize_worker, initargs=(), label="test",
            )
            self.assertEqual([row["seed"] for row in collected], [4, 2])
            self.assertTrue(all(row["pid"] != os.getpid() for row in collected))

            cached = run_seed_jobs(
                [4, 2], directory, metadata, _unexpected_worker,
                workers=2, initializer=_initialize_worker, initargs=(), label="test",
            )
            self.assertEqual([row["seed"] for row in cached], [row["seed"] for row in collected])
            self.assertEqual([row["pid"] for row in cached], [row["pid"] for row in collected])
            for before, after in zip(collected, cached):
                np.testing.assert_array_equal(before["tokens"], after["tokens"])

            restored = run_seed_jobs(
                [4, 2], directory, metadata, _unexpected_worker,
                workers=2, initializer=_initialize_worker, initargs=(), label="test",
            )
            self.assertEqual(restored[0]["tuple_value"], (4, 5))
            self.assertEqual(restored[0]["tokens"].dtype, np.int16)
            np.testing.assert_array_equal(restored[0]["tokens"], np.arange(12, dtype=np.int16).reshape(3, 4))

            (directory / "seed_4.npz").write_bytes(b"broken archive")
            repaired = run_seed_jobs(
                [4, 2], directory, metadata, _collect_seed,
                workers=2, initializer=_initialize_worker, initargs=(), label="test",
            )
            self.assertEqual([row["seed"] for row in repaired], [4, 2])
            self.assertNotEqual(repaired[0]["pid"], collected[0]["pid"])
            self.assertEqual(repaired[1]["pid"], collected[1]["pid"])
            np.testing.assert_array_equal(repaired[1]["tokens"], collected[1]["tokens"])


if __name__ == "__main__":
    unittest.main()
