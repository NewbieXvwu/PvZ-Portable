"""Where one static-candidate branch's time goes (task33 §16.0).

The audit's §13 item 6 assumed pushing the static candidates into the engine would save 8 s a game
(7%).  This measures the ingredients instead of assuming them: one BRANCH_OBSERVE round trip, the
Python json.loads / _frame / featurize that follow it, the torch value-net forward, and the engine's
own restore (BENCH_SNAPSHOT).  It also plays one game to find the largest lean observation, so the
parse cost is measured on a mid-game payload rather than on a nearly empty board.

  python3 scripts/hotpath_branch.py            # needs build/pvz-portable and the game resources
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

DEFAULT_RESOURCES = Path(os.environ.get("PVZ_RESOURCE_DIR", "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"))


def largest_lean_observation(env, level: int, seed: int, multiplier: float):
    """Play one game and keep the largest lean observation: that is the parse cost to beat."""
    from pvz_env import TaskSpec
    obs, _ = env.reset(task=TaskSpec(level=level, seed=seed, zombie_count_multiplier=multiplier))
    env.set_lean_observation(True)
    biggest, biggest_text = obs, ""
    while not obs["terminal"] and obs["tick"] < 40000:
        legal = (obs.get("legal_actions") or {}).get("plants") or []
        obs, *_ = env.step({"type": "plant", **legal[0]} if legal else {"type": "wait", "ticks": 60})
        text = json.dumps(obs, separators=(",", ":"))
        if len(text) > len(biggest_text):
            biggest, biggest_text = obs, text
    env.set_lean_observation(False)
    return biggest, biggest_text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--executable", type=Path, default=ROOT / "build/pvz-portable")
    p.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCES)
    p.add_argument("--level", type=int, default=7)
    p.add_argument("--seed", type=int, default=1740000)
    p.add_argument("--multiplier", type=float, default=3.0)
    p.add_argument("--reps", type=int, default=2000)
    args = p.parse_args()

    from pvz_env import TaskSpec
    from pvz_event_env import EventWaitEnv
    from pvz_value_net import featurize_numpy
    from render_episode import _frame

    env = EventWaitEnv(args.resource_dir, executable=args.executable, headless=True)
    try:
        probe, text = largest_lean_observation(env, args.level, args.seed, args.multiplier)
        obs, _ = env.reset(task=TaskSpec(level=args.level, seed=args.seed, zombie_count_multiplier=args.multiplier))
        root = env.snapshot()
        plants = (obs.get("legal_actions") or {}).get("plants") or []
        specs = [{"type": "plant", **p} for p in plants[:8]]
        env.set_lean_observation(True)
        round_trip = []
        for _ in range(20):
            started = time.perf_counter()
            records = env.branch_snapshot(root, specs, retain_snapshots=False)
            round_trip.append((time.perf_counter() - started) / len(specs))
        env.set_lean_observation(False)
        bench = env._command("BENCH_SNAPSHOT 400")
    finally:
        env.close()

    parses, frames, features = [], [], []
    for _ in range(args.reps):
        started = time.perf_counter(); loaded = json.loads(text); parses.append(time.perf_counter() - started)
        started = time.perf_counter(); frame = _frame(loaded); frames.append(time.perf_counter() - started)
        frame["sun_income_rate"] = 5.0
        started = time.perf_counter(); featurize_numpy(frame); features.append(time.perf_counter() - started)

    import torch
    from pvz_value_net import ValueNet
    net = ValueNet(width=64, context=True).eval()
    board, scalars = torch.zeros(18, 92, 6, 10), torch.zeros(18, 113 + 35)
    with torch.inference_mode():
        for _ in range(20):
            net(board, scalars)
        started = time.perf_counter()
        for _ in range(50):
            net(board, scalars)
        forward = (time.perf_counter() - started) / 50 / 18

    median = statistics.median
    print(json.dumps({
        "worst_lean_bytes": len(text.encode()), "zombies": len(probe.get("zombies") or []),
        "branches_probed": len(specs),
        "branch_round_trip_us": round(1e6 * median(round_trip), 1),
        "python_parse_us": round(1e6 * median(parses), 1),
        "python_frame_us": round(1e6 * median(frames), 1),
        "python_featurize_us": round(1e6 * median(features), 1),
        "torch_forward_us_per_frame": round(1e6 * forward, 1),
        "engine_restore_us": bench.get("restore", {}).get("median_us"),
        "engine_observation_us": bench.get("observation", {}).get("median_us"),
    }), flush=True)


if __name__ == "__main__":
    main()
