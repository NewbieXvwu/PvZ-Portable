from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from pvz_seed_jobs import run_seed_jobs


def _initialize_worker() -> None:
    pass


def _collect_seed(seed: int) -> dict[str, int]:
    return {"seed": seed, "pid": os.getpid()}


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
            self.assertEqual(cached, collected)

            (directory / "seed_4.json.gz").write_bytes(b"broken gzip")
            repaired = run_seed_jobs(
                [4, 2], directory, metadata, _collect_seed,
                workers=2, initializer=_initialize_worker, initargs=(), label="test",
            )
            self.assertEqual([row["seed"] for row in repaired], [4, 2])
            self.assertNotEqual(repaired[0]["pid"], collected[0]["pid"])
            self.assertEqual(repaired[1], collected[1])


if __name__ == "__main__":
    unittest.main()
