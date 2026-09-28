"""End-to-end equivalence proof on the *real* simulator.

The unit tests in ``test_search_value_features.py`` prove the vectorised feature
builder is bit-exact.  This script closes the loop at the level the user cares
about: run the actual C++ simulator, run ``SearchTeacher.advice`` twice on the
same real game states -- once with the vectorised builder, once with the frozen
list-based reference -- and check the search returns *identical* decisions.

Usage:
    python3 real_env_equivalence.py [resource_dir]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "python"))

import torch  # noqa: E402

import pvz_search_value as value_module  # noqa: E402
from pvz_env import PvZEnv, training_task  # noqa: E402
from pvz_search import SearchTeacher  # noqa: E402
from pvz_search_value import SearchValueModel  # noqa: E402
from test_search_value_features import _reference_search_value_features  # noqa: E402

def _resource_dir(argument: str | None) -> Path:
    """Resolve the game resource directory from the argument or ``$PVZ_RESOURCE_DIR``.

    The path is deliberately not baked in -- this repository is public, and a
    personal download directory is nobody else's business.
    """
    candidate = argument or os.environ.get("PVZ_RESOURCE_DIR")
    if not candidate:
        raise SystemExit("pass the resource directory as an argument or set PVZ_RESOURCE_DIR")
    return Path(candidate)
LEVEL = 7
SEEDS = (30000, 30001, 30002, 30003)
BUDGET = 256
CANDIDATE_LIMIT = 8
HORIZON = 900
REPEATS = 3


def summarise(advice) -> str:
    return json.dumps({
        "action": advice.action,
        "candidates": [[action, round(score, 12)] for action, score in advice.candidates],
        "policy": [round(p, 12) for p in advice.search_policy],
        "margin": None if advice.best_second_margin is None else round(advice.best_second_margin, 12),
        "terminal_outcome": advice.terminal_outcome,
        "search_elapsed_ticks": advice.search_elapsed_ticks,
        "simulation_count": advice.simulation_count,
    }, sort_keys=True)


def run(env, observation, model, builder, sink) -> tuple[str, float]:
    """Run one search with *builder* installed and return (summary, seconds)."""
    original = value_module.search_value_features
    value_module.search_value_features = builder
    try:
        teacher = SearchTeacher(env, value_model=model, simulation_budget=BUDGET,
                                candidate_limit=CANDIDATE_LIMIT, horizon_ticks=HORIZON)
        started = time.perf_counter()
        advice = teacher.advice(observation)
        elapsed = time.perf_counter() - started
    finally:
        value_module.search_value_features = original
    sink.append(advice)
    return summarise(advice), elapsed


def main() -> int:
    resource_dir = _resource_dir(sys.argv[1] if len(sys.argv) > 1 else None)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = SearchValueModel().eval()

    print(f"resource dir : {resource_dir}")
    print(f"torch threads: {torch.get_num_threads()}")
    print(f"search config: budget={BUDGET} candidate_limit={CANDIDATE_LIMIT} "
          f"horizon={HORIZON} repeats={REPEATS}\n")

    env = PvZEnv(resource_dir=resource_dir)
    failures = 0
    try:
        for seed in SEEDS:
            observation, _ = env.reset(level=LEVEL, task=training_task(seed, LEVEL))
            if observation.get("terminal"):
                print(f"seed {seed}: level already over, skipping")
                continue

            # Warm up: the first search after reset pays page-fault and allocator costs.
            run(env, observation, model, value_module.search_value_features, [])

            fast_times: list[float] = []
            slow_times: list[float] = []
            summaries: list[str] = []
            for repeat in range(REPEATS):
                order = (value_module.search_value_features, _reference_search_value_features)
                if repeat % 2:
                    order = order[::-1]
                for builder in order:
                    summary, seconds = run(env, observation, model, builder, [])
                    summaries.append(summary)
                    (fast_times if builder is value_module.search_value_features
                     else slow_times).append(seconds)

            identical = len(set(summaries)) == 1
            fast_best, slow_best = min(fast_times), min(slow_times)
            ratio = slow_best / max(fast_best, 1e-9)
            print(f"seed {seed}: identical={identical}  "
                  f"vectorised={fast_best * 1e3:8.2f} ms  "
                  f"reference={slow_best * 1e3:8.2f} ms  speedup={ratio:5.2f}x  "
                  f"action={json.loads(summaries[0])['action']}")
            if not identical:
                failures += 1
                for summary in sorted(set(summaries)):
                    print("   ", summary)
    finally:
        env.close()

    total = len(SEEDS)
    print(f"\n{total - failures}/{total} seeds produced bit-identical search decisions "
          f"across {REPEATS * 2} runs each")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

