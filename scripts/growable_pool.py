"""A multiprocessing pool that can grow while a group is running (task138).

The controller fixes a group's width from MemAvailable before the group starts.
That number used to be frozen for the whole group, so a group that started while
packing or a gate evaluation held memory ran with one to five workers for up to an
hour.  This pool keeps the same per-worker initialiser, the same one-result-per-job
contract and the same "no game is killed" rule, but re-reads available memory every
interval and starts more workers when there is headroom.  It never shrinks.

The growth target is the controller's own admission formula (worker_count in
curriculum_queue.py):

    clamp((MemAvailable - reserve - background) // per_worker, 1, upper)

so a grown worker is admitted by exactly the same rule as the first one.  With an
unmeasured per-worker RSS (or no /proc, e.g. macOS) the pool stays at its initial
width: growing on an unknown cost per worker is what causes OOM.

Every worker owns a private queue and the parent assigns jobs to it, so the parent
always knows which worker holds which game.  That is what makes an OOM-killed worker
(no exception, no result, message never flushed) a clean failure instead of a hang:
a worker that is gone while holding a job means that job is lost.

Results, seeds and the output contract do not change: the parent still consumes one
result per job and writes it to games.jsonl in completion order.
"""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
import traceback
from collections import deque

GIB = 1024 ** 3
MEMORY_RESERVE_BYTES = 3 * GIB
GROW_INTERVAL = 30.0


def mem_available_bytes() -> int | None:
    """Linux MemAvailable in bytes; None where /proc does not exist (macOS)."""
    try:
        with open('/proc/meminfo') as handle:
            for line in handle:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def target_workers(available: int | None, per_worker: int, upper: int,
                   background: int = 0, reserve: int = MEMORY_RESERVE_BYTES) -> int | None:
    """Worker count the admission formula allows now, or None when it cannot be computed."""
    if available is None or per_worker <= 0 or upper < 1:
        return None
    return max(1, min(upper, (available - reserve - background) // per_worker))


def _worker_main(tasks, results, play, init, initargs, worker_id: int) -> None:
    if init is not None:
        init(*initargs)
    while True:
        job = tasks.get()
        if job is None:
            return
        try:
            results.put(('ok', worker_id, play(job)))
        except BaseException:                       # noqa: BLE001 - the parent re-raises
            results.put(('error', worker_id, traceback.format_exc()))
            return


class GrowablePool:
    """A pool of worker processes that can be widened while jobs are still queued.

    play runs one job and returns its result; init/initargs are called once per
    worker before it takes a job (the same initialiser multiprocessing.Pool used).
    upper is the hard cap (--workers), initial the width the controller admitted
    at group start.
    """

    def __init__(self, context, play, init=None, initargs=(), initial=1, upper=None,
                 interval: float = GROW_INTERVAL, available=mem_available_bytes,
                 per_worker: int = 0, background: int = 0,
                 reserve: int = MEMORY_RESERVE_BYTES, clock=time.monotonic):
        self.context = context
        self.play = play
        self.init = init
        self.initargs = tuple(initargs)
        self.initial = max(1, int(initial))
        self.upper = max(self.initial, int(initial if upper is None else upper))
        self.interval = float(interval)
        self.available = available
        self.per_worker = int(per_worker)
        self.background = int(background)
        self.reserve = int(reserve)
        self.clock = clock
        self.processes = []
        self.max_workers = 0        # peak width, for the log line and the tests
        self._results = context.Queue()
        self._tasks = {}            # worker id -> its own SimpleQueue (no feeder thread)
        self._held = {}             # worker id -> the job it was given, or None
        self._process_of = {}       # worker id -> its process
        self._pending = deque()
        self._next_id = 0

    def _spawn(self, count: int) -> None:
        for _ in range(count):
            worker_id = self._next_id
            self._next_id += 1
            tasks = self.context.SimpleQueue()
            process = self.context.Process(
                target=_worker_main,
                args=(tasks, self._results, self.play, self.init, self.initargs, worker_id),
                daemon=True)
            process.start()
            self.processes.append(process)
            self._process_of[worker_id] = process
            self._tasks[worker_id] = tasks
            self._held[worker_id] = None
        self.max_workers = max(self.max_workers, len(self.processes))

    def _assign(self) -> int:
        """Hand pending jobs to idle workers; returns how many were assigned."""
        assigned = 0
        for worker_id in self._tasks:
            if not self._pending:
                break
            if self._held[worker_id] is not None:
                continue
            job = self._pending.popleft()
            self._held[worker_id] = job
            self._tasks[worker_id].put(job)
            assigned += 1
        return assigned

    def _grow(self) -> int:
        """Start workers while the admission formula and the pending jobs allow it."""
        if not self._pending:
            return 0
        allowed = target_workers(self.available(), self.per_worker, self.upper,
                                 self.background, self.reserve)
        if allowed is None:
            return 0
        room = min(allowed, len(self.processes) + len(self._pending)) - len(self.processes)
        if room <= 0:
            return 0
        self._spawn(room)
        return room

    def run(self, jobs):
        """Yield results in completion order; grows the pool every interval."""
        jobs = list(jobs)
        total = len(jobs)
        self._pending.extend(jobs)
        received = 0
        next_check = self.clock() + self.interval
        try:
            self._spawn(min(self.initial, max(1, total)))
            self._assign()
            while received < total:
                timeout = max(0.0, next_check - self.clock())
                try:
                    message = self._results.get(timeout=timeout)
                except queue.Empty:
                    message = None
                if message is not None:
                    status, worker_id, payload = message
                    self._held[worker_id] = None
                    if status == 'ok':
                        received += 1
                        yield payload
                    else:
                        raise RuntimeError('worker failed:\n' + payload)
                    self._assign()
                else:
                    # Nothing came back and nothing is pending for the idle workers: a
                    # worker that is gone was holding a game and took it with it. Without
                    # this the survivors just block on their queues and the parent waits
                    # forever (an OOM kill sends no message at all).
                    for worker_id, job in self._held.items():
                        if job is not None and not self._process_of[worker_id].is_alive():
                            raise RuntimeError(f'worker {worker_id} exited while holding a game '
                                               f'({total - received} games left)')
                now = self.clock()
                if now >= next_check:
                    self._grow()
                    self._assign()
                    next_check = now + self.interval
        finally:
            self.close()

    def close(self, budget: float = 30.0) -> None:
        """Sentinel every worker, then wait at most budget seconds in total.

        The normal path is instant: all jobs are done, every worker is blocked on its
        queue and takes the sentinel at once, so finalizers reap the simulator children
        and RUSAGE_CHILDREN counts them exactly once.  An interrupted run waits at most
        budget seconds and then terminates, like Pool.__exit__.
        """
        for tasks in self._tasks.values():
            tasks.put(None)
        deadline = time.monotonic() + budget
        for process in self.processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        self.processes = []
        self._tasks = {}
        self._held = {}
        self._process_of = {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False
