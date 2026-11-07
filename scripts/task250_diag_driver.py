"""Run task250 first-divergence diagnostics in parallel, resuming completed roots.

Each job is one (level, seed) pair.  A job whose diagnostic json already exists is skipped,
so a long diagnosis list can be stopped and restarted without repeating work.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_one(job):
    base = Path(job["data"])
    candidate = job.get("candidate", "finetune-t3")
    folder = "diag" if job.get("no_terminal", True) else "diagT"
    out = base / f"{folder}/l{job['level']}-{job['seed']}-{candidate}.json"
    if out.exists():
        return dict(level=job["level"], seed=job["seed"], skipped=True)
    command = [PY, str(ROOT / "scripts/task250_diagnose.py"),
               "--level", str(job["level"]), "--seed", str(job["seed"]),
               "--candidate", candidate,
               "--futures", str(job.get("futures", 4)),
               "--max-candidates", str(job.get("max_candidates", 5)),
               "--data", str(base)]
    if job.get("no_terminal", True):
        command.append("--no-terminal")
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    log = base / f"{folder}/l{job['level']}-{job['seed']}-{candidate}.log"
    log.write_text(result.stdout[-4000:] + "\n--- stderr ---\n" + result.stderr[-4000:])
    if result.returncode:
        raise RuntimeError(f"L{job['level']} seed {job['seed']}: {result.stderr[-500:]}")
    return dict(level=job["level"], seed=job["seed"], skipped=False,
                seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task250")
    parser.add_argument("--jobs", nargs="+", required=True, help="level:seed pairs")
    parser.add_argument("--futures", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--terminal", action="store_true",
                        help="continue each candidate to the real terminal (expensive)")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--candidate", default="finetune-t3")
    args = parser.parse_args()
    base = args.data.resolve()
    jobs = []
    for item in args.jobs:
        level, seed = item.split(":")
        jobs.append(dict(level=int(level), seed=int(seed), futures=args.futures,
                         max_candidates=args.max_candidates,
                         no_terminal=not args.terminal, data=str(base), candidate=args.candidate))
    os.environ.update(PVZ_NO_POPUP="1", TMPDIR=str(base / "work"),
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        queue = iter(jobs)
        exhausted = False
        while futures or not exhausted:
            while not exhausted and len(futures) < args.workers:
                job = next(queue, None)
                if job is None:
                    exhausted = True
                else:
                    futures[pool.submit(run_one, job)] = job
            done, _ = wait(list(futures), timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    print(json.dumps(dict(error=str(error), level=job["level"], seed=job["seed"])), flush=True)
                    continue
                results.append(result)
                print(json.dumps(result), flush=True)
    (base / "diag-plan-report.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
