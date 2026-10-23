"""Where a collection worker's resident memory goes (task32).

One worker of search_play.py is a Python process (torch, the student export, the value net, the
observation cache) plus an engine process. This loads the same pieces in the same order as
_init_worker and reports the resident set after each step, then opens an engine, resets a level and
reports the engine's own resident set. It plays no game: the search's own scratch is per decision, and
the number that matters for planning is what an idle worker costs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

RESOURCE = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")


def rss(pid: int | None = None) -> int:
    """Resident set in bytes; pid=None is this process."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid or os.getpid())],
                         capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student", type=Path, default=Path("/Users/newbiexvwu/PvZAgent/artifacts/student/v25/model.pt"))
    p.add_argument("--value", type=Path, default=Path("/Users/newbiexvwu/PvZAgent/artifacts/hard_mac/value_v4/model.pt"))
    p.add_argument("--level", type=int, default=7)
    p.add_argument("--seed", type=int, default=30000)
    args = p.parse_args()

    steps = {"interpreter": rss()}
    import search_play                                    # noqa: F401  (the worker's own import set)
    steps["search_play imported"] = rss()
    search_play._load_nets(str(args.value), str(args.student))
    steps["nets loaded"] = rss()

    from pvz_env import training_task
    from pvz_event_env import EventWaitEnv
    env = EventWaitEnv(RESOURCE, executable=ROOT / "build/pvz-portable", headless=True)
    try:
        observation, _ = env.reset(task=training_task(args.seed, args.level))
        steps["engine reset"] = rss()
        engine = rss(env._process.pid)
    finally:
        env.close()
    steps["engine resident"] = engine

    report = {"steps_bytes": steps,
              "steps_mib": {k: round(v / 2**20, 1) for k, v in steps.items()},
              "worker_without_engine_mib": round(steps["nets loaded"] / 2**20, 1),
              "worker_plus_engine_mib": round((steps["engine reset"] + engine) / 2**20, 1)}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
