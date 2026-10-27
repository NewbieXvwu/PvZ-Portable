"""task138: the collection pool must be able to widen after the group started.

Uses simulated MemAvailable readings and trivial jobs, so no engine and no real
memory pressure is involved.  The pool is real (fork), the jobs are real processes.
"""
import multiprocessing as mp
import os
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts')]

from growable_pool import GIB, GrowablePool, target_workers  # noqa: E402


def readings(values):
    """Return the next reading, repeating the last one forever."""
    queue = list(values)

    def probe():
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return probe


def play(job):
    time.sleep(job['sleep'])
    return dict(seed=job['seed'], pid=os.getpid())


def failing(job):
    raise ValueError(f'boom {job["seed"]}')


def dying(job):
    """A hard crash (what an OOM kill looks like): no exception, no result."""
    if job['die']:
        os._exit(1)
    time.sleep(job['sleep'])
    return dict(seed=job['seed'], pid=os.getpid())


class TargetTest(unittest.TestCase):
    def test_admission_formula(self):
        self.assertEqual(target_workers(40 * GIB, 2 * GIB, 16), 16)
        self.assertEqual(target_workers(40 * GIB, 2 * GIB, 4), 4)
        self.assertEqual(target_workers(10 * GIB, 2 * GIB, 16), 3)     # (10 - 3) // 2
        self.assertEqual(target_workers(2 * GIB, 2 * GIB, 16), 1)      # floor is one worker
        self.assertEqual(target_workers(40 * GIB, 2 * GIB, 16, background=8 * GIB), 14)

    def test_unknown_memory_or_cost_never_grows(self):
        self.assertIsNone(target_workers(None, 2 * GIB, 16))
        self.assertIsNone(target_workers(40 * GIB, 0, 16))


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.context = mp.get_context('fork')

    def test_group_starts_at_one_worker_and_widens_when_memory_returns(self):
        pool = GrowablePool(self.context, play, initial=1, upper=4, interval=0.05,
                            available=readings([4 * GIB, 40 * GIB]),
                            per_worker=2 * GIB)
        jobs = [dict(seed=index, sleep=0.15) for index in range(8)]
        results = list(pool.run(jobs))
        self.assertEqual(sorted(result['seed'] for result in results), list(range(8)))
        self.assertEqual(pool.max_workers, 4)
        self.assertGreater(len({result['pid'] for result in results}), 1)

    def test_grown_pool_never_shrinks_when_memory_falls_again(self):
        pool = GrowablePool(self.context, play, initial=1, upper=4, interval=0.05,
                            available=readings([40 * GIB, GIB, GIB]),
                            per_worker=2 * GIB)
        jobs = [dict(seed=index, sleep=0.2) for index in range(8)]
        widths = []
        for _result in pool.run(jobs):
            widths.append(len(pool.processes))
        self.assertEqual(pool.max_workers, 4)
        self.assertTrue(all(width == 4 for width in widths), widths)

    def test_unmeasured_cost_keeps_the_initial_width(self):
        pool = GrowablePool(self.context, play, initial=2, upper=8, interval=0.02,
                            available=readings([64 * GIB]), per_worker=0)
        jobs = [dict(seed=index, sleep=0.02) for index in range(6)]
        list(pool.run(jobs))
        self.assertEqual(pool.max_workers, 2)

    def test_background_headroom_limits_the_growth(self):
        pool = GrowablePool(self.context, play, initial=1, upper=8, interval=0.05,
                            available=readings([40 * GIB]), per_worker=2 * GIB,
                            background=30 * GIB)
        jobs = [dict(seed=index, sleep=0.15) for index in range(6)]
        list(pool.run(jobs))
        self.assertEqual(pool.max_workers, 3)      # (40 - 3 - 30) // 2

    def test_a_worker_that_dies_mid_game_does_not_hang_the_parent(self):
        pool = GrowablePool(self.context, dying, initial=2, upper=2, interval=0.05,
                            available=readings([40 * GIB]), per_worker=2 * GIB)
        jobs = [dict(seed=1, die=True, sleep=0), dict(seed=2, die=False, sleep=0.3)]
        with self.assertRaises(RuntimeError):
            list(pool.run(jobs))

    def test_worker_failure_reaches_the_parent(self):
        pool = GrowablePool(self.context, failing, initial=1, upper=2, interval=0.05,
                            available=readings([40 * GIB]), per_worker=2 * GIB)
        with self.assertRaises(RuntimeError):
            list(pool.run([dict(seed=1, sleep=0)]))


if __name__ == '__main__':
    unittest.main()
