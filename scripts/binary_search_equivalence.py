"""Prove two ``pvz-portable`` binaries drive the search to *identical* decisions.

Every change to ``BRANCH_SNAPSHOT_FAST`` has to be justified twice: it must be faster, and
it must not change what the teacher does.  Timing is easy.  Behaviour is not, because the
search is a long, branching, budget-limited procedure whose result depends on the exact
order in which states are compared -- so "the unit tests still pass" is nowhere near
enough.  This script runs ``SearchTeacher.advice`` on the same real game states against
two executables and requires the two to agree on everything the caller can observe: the
chosen action, every scored candidate, the distilled policy, the best/second margin, the
terminal outcome and the simulation count.

It deliberately does **not** compare ``state_hash``.  That value is an opaque transposition
key, so rewriting the hash function changes the digits while leaving every dedup decision
alone; comparing it would fail on a legitimate optimisation.

    python scripts/binary_search_equivalence.py --resource-dir <dir> \
        --other-binary /tmp/pvz-portable.baseline

Exit status is 1 if any seed disagrees.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "python"))

import torch  # noqa: E402

from pvz_env import PvZEnv, training_task  # noqa: E402
from pvz_search import SearchTeacher  # noqa: E402
from pvz_search_value import SearchValueModel  # noqa: E402


def summarise(advice) -> str:
    """Everything a caller can observe, except the opaque ``state_hash``."""
    return json.dumps({
        "action": advice.action,
        "candidates": [[action, round(score, 12)] for action, score in advice.candidates],
        "policy": [round(p, 12) for p in advice.search_policy],
        "margin": None if advice.best_second_margin is None else round(advice.best_second_margin, 12),
        "terminal_outcome": advice.terminal_outcome,
        "search_elapsed_ticks": advice.search_elapsed_ticks,
        "simulation_count": advice.simulation_count,
        "screening_simulations": getattr(advice, "screening_simulations", None),
        "depth_simulations": getattr(advice, "depth_simulations", None),
    }, sort_keys=True)


def run(env: PvZEnv, observation: dict, model: SearchValueModel, repeats: int) -> tuple[str, float]:
    """Return (summary, best seconds) over *repeats* searches on the same state."""
    summaries = []
    best = float("inf")
    for _ in range(repeats):
        teacher = SearchTeacher(env, value_model=model, simulation_budget=BUDGET,
                                candidate_limit=CANDIDATE_LIMIT, horizon_ticks=HORIZON)
        started = time.perf_counter()
        advice = teacher.advice(observation)
        best = min(best, time.perf_counter() - started)
        summaries.append(summarise(advice))
    if len(set(summaries)) != 1:
        raise SystemExit(f"one binary is not self-consistent across repeats:\n"
                         + "\n".join(sorted(set(summaries))))
    return summaries[0], best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resource-dir", type=Path,
                        default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--other-binary", type=Path, required=True,
                        help="the executable to compare against the built one")
    parser.add_argument("--levels", type=int, nargs="+", default=(8,))
    parser.add_argument("--seeds", type=int, nargs="+", default=(30000, 30001))
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--candidate-limit", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=900)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not args.resource_dir:
        raise SystemExit("pass --resource-dir or set PVZ_RESOURCE_DIR")

    global BUDGET, CANDIDATE_LIMIT, HORIZON
    BUDGET, CANDIDATE_LIMIT, HORIZON = args.budget, args.candidate_limit, args.horizon
    BUDGET, CANDIDATE_LIMIT, HORIZON = args.budget, args.candidate_limit, args.horizon

    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = SearchValueModel().eval()

    built = PvZEnv(resource_dir=args.resource_dir)
    other = PvZEnv(resource_dir=args.resource_dir, executable=args.other_binary)
    failures = 0
    try:
        for level in args.levels:
            for seed in args.seeds:
                task = training_task(seed, level)
                built_observation, _ = built.reset(level=level, task=task)
                other_observation, _ = other.reset(level=level, task=task)
                if built_observation.get("terminal") or other_observation.get("terminal"):
                    print(f"level {level} seed {seed}: level already over, skipping")
                    continue

                # Warm up: the first search after reset pays page-fault and allocator costs.
                run(built, built_observation, model, 1)
                run(other, other_observation, model, 1)

                built_summary, built_seconds = run(built, built_observation, model, args.repeats)
                other_summary, other_seconds = run(other, other_observation, model, args.repeats)
                identical = built_summary == other_summary
                ratio = other_seconds / max(built_seconds, 1e-9)
                print(f"level {level} seed {seed}: identical={identical}  "
                      f"built={built_seconds * 1e3:8.2f} ms  "
                      f"other={other_seconds * 1e3:8.2f} ms  speedup={ratio:5.2f}x")
                if not identical:
                    failures += 1
                    for label, summary in (("built", built_summary), ("other", other_summary)):
                        print(f"    {label}: {summary}")
    finally:
        built.close()
        other.close()

    total = len(args.levels) * len(args.seeds)
    print(f"\n{total - failures}/{total} (level, seed) pairs produced identical search decisions")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
