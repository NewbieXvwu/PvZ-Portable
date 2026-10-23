"""Is ``OBS_LEAN`` usable for the episode loop's step? (task33 §17 item 7)

The search already asks for the compact observation inside rollouts (``lean_rollouts``), but the
episode loop's ``step`` still gets the full one.  This runs the same seed and the same action
stream twice -- ``OBS_LEAN 0`` then ``OBS_LEAN 1`` -- and reports, per level: whether every
``render_episode._frame`` is identical, the payload ratio and the per-step time.

  python3 scripts/hotpath_lean.py --levels 7,21 --steps 400
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pvz_env import TaskSpec  # noqa: E402
from pvz_event_env import EventWaitEnv  # noqa: E402
from render_episode import _frame  # noqa: E402

DEFAULT_RESOURCES = Path(os.environ.get("PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))


def play(executable: Path, resources: Path, level: int, seed: int, steps: int, lean: bool):
    """One episode with a fixed action stream; returns (frames, payload bytes, seconds)."""
    env = EventWaitEnv(resources, executable=executable, headless=True)
    try:
        obs, _ = env.reset(task=TaskSpec(level=level, seed=seed))
        if lean:
            env.set_lean_observation(True)      # after reset: the task context comes from the first observation
        frames, payload = [], 0
        started = time.perf_counter()
        for index in range(steps):
            legal = (obs.get("legal_actions") or {}).get("plants") or []
            action = {"type": "plant", **legal[0]} if (legal and index % 7 == 0) else {"type": "wait", "ticks": 60}
            obs, _r, done, _t, _info = env.step(action)
            payload += len(json.dumps(obs, separators=(",", ":")))
            frames.append(_frame(obs))
            if done:
                break
        return frames, payload, time.perf_counter() - started
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--levels", default="7,21")
    p.add_argument("--seeds", default="1740000")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--executable", type=Path, default=ROOT / "build/pvz-portable")
    p.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCES)
    args = p.parse_args()
    levels = [int(v) for v in args.levels.split(",")]
    seeds = [int(v) for v in args.seeds.split(",")]
    for level, seed in zip(levels, seeds * len(levels)):
        full, full_bytes, full_seconds = play(args.executable, args.resource_dir, level, seed, args.steps, False)
        lean, lean_bytes, lean_seconds = play(args.executable, args.resource_dir, level, seed, args.steps, True)
        print(json.dumps({
            "level": level, "seed": seed, "steps": len(full),
            "frames_identical": len(full) == len(lean) and all(a == b for a, b in zip(full, lean)),
            "bytes_full": full_bytes, "bytes_lean": lean_bytes,
            "bytes_ratio": round(lean_bytes / max(full_bytes, 1), 3),
            "per_step_ms_full": round(1000 * full_seconds / max(len(full), 1), 3),
            "per_step_ms_lean": round(1000 * lean_seconds / max(len(lean), 1), 3)}), flush=True)


if __name__ == "__main__":
    main()
