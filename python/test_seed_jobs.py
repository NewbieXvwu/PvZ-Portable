from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from pvz_seed_jobs import run_seed_jobs


def _initialize_worker() -> None:
    pass


def _initialize_dying_worker() -> None:
    """Die the way an OOM kill or a segfault looks: no exception, no traceback.

    A raising initializer would print a traceback from every respawned worker and
    drown the test output; ``os._exit`` reproduces the same silent death quietly.
    """
    os._exit(1)


def _collect_seed(seed: int) -> dict[str, object]:
    return {"seed": seed, "pid": os.getpid(), "tokens": np.arange(12, dtype=np.int16).reshape(3, 4),
            "tuple_value": (seed, seed + 1)}


def _collect_with_int_key(seed: int) -> dict[object, object]:
    # ``_archive_encode`` rewrites dict keys with ``str(key)``, so an int key only
    # survives when the parent takes the result straight from the worker.
    return {"seed": seed, 7: "int-key"}


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

    def test_fresh_results_come_from_the_worker_not_the_shard(self) -> None:
        """The parent must not pay a shard round trip for data it just produced."""
        metadata = {"task": "int-key", "version": 1}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            fresh = run_seed_jobs(
                [5], directory, metadata, _collect_with_int_key,
                workers=1, initializer=_initialize_worker, initargs=(), label="test",
            )
            self.assertIn(7, fresh[0], "int key was stringified, so the shard was re-read")
            self.assertTrue((directory / "seed_5.npz").exists(),
                            "the shard must still be written for resumability")

            cached = run_seed_jobs(
                [5], directory, metadata, _unexpected_worker,
                workers=1, initializer=_initialize_worker, initargs=(), label="test",
            )
            self.assertIn("7", cached[0], "the shard path is expected to stringify the key")

    def test_a_dying_initializer_raises_instead_of_hanging(self) -> None:
        """A worker that dies must not leave the parent waiting forever."""
        with tempfile.TemporaryDirectory() as temporary:
            start = time.perf_counter()
            with self.assertRaises(RuntimeError) as raised:
                run_seed_jobs(
                    [1, 2, 3], Path(temporary), {"task": "dying"}, _collect_seed,
                    workers=2, initializer=_initialize_dying_worker, initargs=(),
                    label="dying", stall_timeout=3.0,
                )
            elapsed = time.perf_counter() - start
            self.assertIn("stalled", str(raised.exception))
            self.assertIn("dying", str(raised.exception))
            self.assertLess(elapsed, 30.0, "the watchdog must fire on its own timeout")


if __name__ == "__main__":
    unittest.main()
