"""Micro-benchmark for the snapshot round trips the search pays for.

``BRANCH_SNAPSHOT_FAST`` is where the search spends most of its wall clock, and the cost
is per branch: restore the parent, run the action, serialise an observation, save a child
snapshot and hash it.  This script drives those commands directly and reports the cost of
each ingredient, so an optimisation can be attributed instead of guessed at.

It talks to the protocol with ``_command`` rather than through ``PvZEnv``'s public
methods on purpose: the search uses the ``*_FAST`` variants, which return a minimal
response, and measuring the full-observation variants would overstate the cost.  It also
bypasses ``PvZEnv``'s snapshot bookkeeping, which the search does too.

Every run reports a fingerprint of the measured board, so two runs are only comparable
when the fingerprint matches.

    python scripts/branch_benchmark.py --resource-dir <PvZ 1.2.0.1073 dir>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from pvz_common import canonical_digest  # noqa: E402
from pvz_env import PvZEnv, branch_action_token, training_task  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5)
BRANCH_COUNTS = (1, 2, 4, 8, 16, 32)
PLANT_LIMIT = 12


def time_command(env: PvZEnv, command: Callable[[], Any], reps: int, warmup: int = 3) -> dict[str, float]:
    """Median and mean wall time of ``command`` in microseconds."""
    for _ in range(warmup):
        command()
    samples = []
    for _ in range(reps):
        started = time.perf_counter()
        command()
        samples.append((time.perf_counter() - started) * 1e6)
    return {"median_us": statistics.median(samples), "mean_us": statistics.mean(samples),
            "min_us": min(samples), "reps": reps}


def advance_to_busy_board(env: PvZEnv, task: Any, ticks: int) -> dict[str, Any]:
    """Plant what is affordable, then wait, until the board has plants and zombies."""
    observation = env.reset(deck=DECK, task=task)[0]
    planted = 0
    for _ in range(ticks):
        if observation["terminal"]:
            break
        placements = observation["legal_actions"]["plants"]
        if placements and planted < PLANT_LIMIT:
            observation = env.step({"type": "plant", **placements[0]})[0]
            planted += 1
        else:
            observation = env.step({"type": "wait", "ticks": 1})[0]
    return observation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--level", type=int, default=8)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--warmup-ticks", type=int, default=900)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--branch-reps", type=int, default=15)
    parser.add_argument("--json", action="store_true", help="emit the raw result as JSON")
    args = parser.parse_args()

    task = training_task(args.seed, args.level)
    report: dict[str, Any] = {"level": args.level, "seed": args.seed,
                              "executable": str(args.executable) if args.executable else "default"}

    with PvZEnv(args.resource_dir, args.executable, headless=True) as env:
        observation = advance_to_busy_board(env, task, args.warmup_ticks)
        privileged = env.privileged_state()
        report["board"] = {
            "tick": observation["tick"],
            "plants": len(observation["plants"]),
            "zombies": len(observation["zombies"]),
            "projectiles": len(observation["projectiles"]),
            "reanimations": len(privileged.get("hidden", {}).get("reanimations", [])),
            "digest": canonical_digest(privileged),
        }

        root = env.snapshot()
        # Every branch restores the parent before acting, so the batch does not compound:
        # each entry costs one full restore + action + observation + save + hash, which is
        # exactly the cost being measured.  Legal placements come first because they are
        # what the search actually branches on; the batch is then padded with one-tick
        # waits, which exercise the same path, so the count can reach the protocol limit
        # even on a board where nothing is affordable.
        actions = [{"type": "plant", **placement}
                   for placement in observation["legal_actions"]["plants"][:32]]
        actions += [{"type": "wait", "ticks": 1}] * (max(BRANCH_COUNTS) - len(actions))
        report["batch"] = {"legal_placements": len(observation["legal_actions"]["plants"]),
                           "actions": len(actions)}

        # The ingredients of one branch, measured separately.
        report["observation"] = time_command(env, lambda: env._command("OBS"), args.reps)
        report["privileged"] = time_command(env, lambda: env._command("PRIV"), args.reps)

        def save_drop() -> None:
            response = env._command("SNAPSHOT_FAST")
            env._command(f"DROP_SNAPSHOT_FAST {response['snapshot_id']}")

        report["snapshot_save_plus_drop"] = time_command(env, save_drop, args.reps)
        report["restore"] = time_command(env, lambda: env._command(f"RESTORE_FAST {root}"), args.reps)

        branches: dict[str, Any] = {}
        for count in BRANCH_COUNTS:
            batch = actions[:count]
            if len(batch) != count:
                continue
            tokens = " ".join(branch_action_token(action) for action in batch)

            def branch(batch_tokens: str = tokens, count: int = count) -> None:
                response = env._command(f"BRANCH_SNAPSHOT_FAST {root} {count} {batch_tokens}")
                for item in response["branches"]:
                    if item.get("snapshot_id") is not None:
                        env._command(f"DROP_SNAPSHOT_FAST {item['snapshot_id']}")

            branches[str(count)] = time_command(env, branch, args.branch_reps)
            # Leave the board at the root state for the next measurement.
            env._command(f"RESTORE_FAST {root}")

        report["branch"] = branches
        marginal = {}
        ordered = [str(count) for count in BRANCH_COUNTS if str(count) in branches]
        for previous, current in zip(ordered, ordered[1:]):
            delta_branches = int(current) - int(previous)
            delta_us = branches[current]["median_us"] - branches[previous]["median_us"]
            marginal[f"{previous}->{current}"] = delta_us / delta_branches
        report["marginal_us_per_branch"] = marginal

        env._command(f"DROP_SNAPSHOT_FAST {root}")

    if args.json:
        print(json.dumps(report, indent=2))
        return

    board = report["board"]
    print(f"level {report['level']} seed {report['seed']}  tick {board['tick']}  "
          f"plants {board['plants']}  zombies {board['zombies']}  "
          f"projectiles {board['projectiles']}  reanimations {board['reanimations']}")
    print(f"board digest {board['digest']}")
    print()
    for name in ("observation", "privileged", "snapshot_save_plus_drop", "restore"):
        stats = report[name]
        print(f"  {name:26s} {stats['median_us']:9.1f} us  (min {stats['min_us']:8.1f})")
    print()
    for count in ordered:
        stats = branches[count]
        print(f"  branch x{count:<3s}                 {stats['median_us']:9.1f} us  "
              f"({stats['median_us'] / int(count):7.1f} us/branch)")
    print()
    for key, value in marginal.items():
        print(f"  marginal {key:10s}          {value:9.1f} us/branch")


if __name__ == "__main__":
    main()
