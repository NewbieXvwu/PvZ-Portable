"""Measure repeated RESET memory, optionally compare every public response to a reference binary."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_env import PvZEnv, TaskSpec
from pvz_seed_jobs import atomic_json


def rss(pid: int) -> int:
    for line in (Path("/proc") / str(pid) / "status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise ValueError("process RSS unavailable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--reference-executable", type=Path)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.executable = args.executable.resolve()
    if args.reference_executable:
        args.reference_executable = args.reference_executable.resolve()
    if not 1 <= args.episodes <= 1024:
        raise ValueError("bounded audit requires 1..1024 resets")
    started, records, comparisons = time.monotonic(), [], 0
    points = {1, 8, 32, 64, 128, 256, 512, 1024, args.episodes}
    for ticks in (0, 300):
        with ExitStack() as stack:
            env = stack.enter_context(PvZEnv(resource_dir=args.resource_dir, executable=args.executable))
            reference = (stack.enter_context(PvZEnv(resource_dir=args.resource_dir,
                                                   executable=args.reference_executable))
                         if args.reference_executable else None)
            for index in range(args.episodes):
                kwargs = {"deck": [0, 1, 2, 3, 4, 5],
                          "task": TaskSpec(level=1, seed=60000, wave_cap=1)}
                actual = env.reset(**kwargs)
                if reference:
                    expected = reference.reset(**kwargs)
                    if actual != expected:
                        raise AssertionError(f"reset response differs at {ticks}/{index}")
                    comparisons += 1
                if ticks:
                    action = {"type": "wait", "ticks": ticks}
                    actual = env.step(action)
                    if reference:
                        expected = reference.step(action)
                        if actual != expected:
                            raise AssertionError(f"wait response differs at {ticks}/{index}")
                        comparisons += 1
                if index + 1 in points:
                    row = {"resets": index + 1, "wait_ticks": ticks,
                           "python_rss_bytes": rss(os.getpid()),
                           "simulator_rss_bytes": rss(env._process.pid)}
                    if reference:
                        row["reference_simulator_rss_bytes"] = rss(reference._process.pid)
                    records.append(row)
                    atomic_json(args.output.with_suffix(".progress.json"), records)
                    print(row, flush=True)
    atomic_json(args.output, {"scope": "paired repeated same-task RESETs; with/without one300tick advance",
                             "episodes_per_process": args.episodes, "records": records,
                             "simulator_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                             "reference_sha256": hashlib.sha256(args.reference_executable.read_bytes()).hexdigest()
                                if args.reference_executable else None,
                             "exact_public_response_comparisons": comparisons,
                             "seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
