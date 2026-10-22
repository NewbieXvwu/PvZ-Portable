"""Replay one deterministic action sequence through two engines and compare every frame (task32).

The engine gate is not "the same weights": it is that the same actions produce the same state, frame by
frame. This drives `pvz-portable -env` through a fixed sequence, records the compact frame of every
decision, and prints a digest plus the peak resident set and the startup time of the engine process, so a
resource-loading change can be checked against the binary it replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pvz_env import training_task  # noqa: E402
from pvz_event_env import EventWaitEnv  # noqa: E402
from render_episode import _frame  # noqa: E402

RESOURCE = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")


def replay(executable: Path, level: int, seed: int, decisions: int) -> dict:
    started = time.time()
    env = EventWaitEnv(RESOURCE, executable=executable)
    frames = []
    try:
        observation, _ = env.reset(task=training_task(seed, level))
        startup = time.time() - started
        frames.append(_frame(observation))
        for decision in range(decisions):
            legal = observation.get("legal_actions", {}).get("plants") or []
            action = ({"type": "plant", **legal[(decision * 7) % len(legal)]} if legal and decision % 3 == 0
                      else {"type": "wait", "ticks": 60})
            observation, *_rest, info = env.step(action)
            frames.append(_frame(observation))
            if info.get("terminal"):
                break
    finally:
        env.close()
    payload = json.dumps(frames, sort_keys=True, separators=(",", ":")).encode()
    return {"executable": str(executable), "frames": len(frames),
            "digest": hashlib.sha256(payload).hexdigest(),
            "final_tick": frames[-1].get("tick"), "final_wave": frames[-1].get("wave"),
            "startup_seconds": round(startup, 3), "replay_seconds": round(time.time() - started, 2),
            "peak_child_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            "frame_bytes": len(payload)}


def _run_child(executable, level, seed, decisions, connection) -> None:
    try:
        connection.send(replay(executable, level, seed, decisions))
    except BaseException as exc:                     # report, do not hang the parent on the pipe
        connection.send({"executable": str(executable), "error": repr(exc)})
    finally:
        connection.close()


def replay_isolated(executable: Path, level: int, seed: int, decisions: int) -> dict:
    """One replay in a fresh interpreter: ru_maxrss of a process covers every child it ever reaped, so
    measuring two engines from the same parent would report the first one's peak for both."""
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_run_child, args=(executable, level, seed, decisions, sender))
    process.start()
    sender.close()
    result = receiver.recv()
    process.join()
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--executable", type=Path, action="append", required=True,
                   help="one or more engines; every one after the first is compared with the first")
    p.add_argument("--level", type=int, default=7)
    p.add_argument("--seed", type=int, default=30000)
    p.add_argument("--decisions", type=int, default=120)
    p.add_argument("--json", type=Path)
    args = p.parse_args()
    results = [replay_isolated(executable, args.level, args.seed, args.decisions) for executable in args.executable]
    for result in results:
        print(json.dumps(result), flush=True)
    reference = results[0]
    report = {"level": args.level, "seed": args.seed, "decisions": args.decisions, "runs": results,
              "identical": all(run["digest"] == reference["digest"] for run in results)}
    print(json.dumps({"identical": report["identical"]}), flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
