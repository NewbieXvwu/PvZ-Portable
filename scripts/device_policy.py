"""Decide the right device policy for Apple Silicon.

``resolve_device("auto")`` picks MPS on any Apple Silicon Mac, and
``train_pvz_agent.py`` binds *both* the search value model and the student model
to that single device.  The value model is evaluated once per search leaf
(``SearchTeacher._leaf_value`` -> ``SearchValueModel.predict``), which is a
batch-of-1 forward pass -- the shape where MPS is known to be slow.

This script measures the *net* effect on the real simulator: same seed, same
search config, value model on CPU vs on MPS.  It reports

  * total ``advice()`` wall time,
  * how much of that time is the leaf evaluator,
  * whether both devices produce the same decision.

Usage:
    python3 device_policy.py [resource_dir] [--seeds 30000,30001]
    python3 device_policy.py [resource_dir] --episode [--seeds 30000]
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

from pvz_env import PvZEnv, training_task  # noqa: E402
from pvz_search import SearchTeacher  # noqa: E402
from pvz_search_value import SearchValueModel  # noqa: E402

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
DEFAULT_SEEDS = (30000, 30001)
BUDGET = 256
CANDIDATE_LIMIT = 8
HORIZON = 900
DECK = (0, 1, 2, 3, 4, 5)  # mirrors train_pvz_agent.DECK
MAX_DECISIONS = 64
MAX_ACTIONS = 2000

# Accumulators filled by the instrumented leaf evaluator.
_LEAF_SECONDS = 0.0
_LEAF_CALLS = 0
_LEAF_ORIGINAL = SearchTeacher._leaf_value


def _timed_leaf_value(self, observation):  # type: ignore[no-untyped-def]
    global _LEAF_SECONDS, _LEAF_CALLS
    started = time.perf_counter()
    try:
        return _LEAF_ORIGINAL(self, observation)
    finally:
        _LEAF_SECONDS += time.perf_counter() - started
        _LEAF_CALLS += 1


def summarise(advice) -> str:
    return json.dumps({
        "action": advice.action,
        "candidates": [[action, round(score, 12)] for action, score in advice.candidates],
        "policy": [round(p, 12) for p in advice.search_policy],
        "margin": None if advice.best_second_margin is None else round(advice.best_second_margin, 12),
        "terminal_outcome": advice.terminal_outcome,
        "simulation_count": advice.simulation_count,
    }, sort_keys=True)


def measure(env, observation, model) -> dict:
    global _LEAF_SECONDS, _LEAF_CALLS
    _LEAF_SECONDS = 0.0
    _LEAF_CALLS = 0
    teacher = SearchTeacher(env, value_model=model, simulation_budget=BUDGET,
                            candidate_limit=CANDIDATE_LIMIT, horizon_ticks=HORIZON)
    started = time.perf_counter()
    advice = teacher.advice(observation)
    total = time.perf_counter() - started
    return {
        "total_ms": total * 1e3,
        "leaf_ms": _LEAF_SECONDS * 1e3,
        "leaf_calls": _LEAF_CALLS,
        "leaf_us_each": (_LEAF_SECONDS / max(_LEAF_CALLS, 1)) * 1e6,
        "summary": summarise(advice),
    }


def measure_episode(env, model, seed: int) -> dict:
    """Time one whole rollout collection -- the unit that actually dominates a run."""
    import tempfile

    from train_pvz_agent import collect_search_episode

    with tempfile.TemporaryDirectory() as replay_dir:
        started = time.perf_counter()
        episode = collect_search_episode(
            env, seed, Path(replay_dir), model, MAX_ACTIONS, LEVEL, DECK, 1.0,
            3, CANDIDATE_LIMIT, HORIZON, BUDGET, MAX_DECISIONS, "device_policy",
        )
        total = time.perf_counter() - started
    return {
        "total_s": total,
        "decisions": len(episode["steps"]),
        "seconds_per_decision": total / max(len(episode["steps"]), 1),
        "won": bool(episode.get("won")),
        "tick": int(episode.get("tick", 0)),
        "actions": [step["action"] for step in episode["steps"]],
    }


def main() -> int:
    args = [item for item in sys.argv[1:] if not item.startswith("--")]
    resource_dir = _resource_dir(args[0] if args else None)
    episode_mode = "--episode" in sys.argv
    seeds = DEFAULT_SEEDS
    for item in sys.argv[1:]:
        if item.startswith("--seeds"):
            seeds = tuple(int(part) for part in item.split("=", 1)[1].split(","))

    if not torch.backends.mps.is_available():
        print("MPS is not available on this machine; nothing to compare.")
        return 1

    torch.set_num_threads(1)
    torch.manual_seed(0)
    # Build the weights once on CPU, then replicate to MPS, so both devices are
    # provably evaluating the *same* parameters.
    cpu_model = SearchValueModel().eval()
    mps_model = SearchValueModel().eval()
    mps_model.load_state_dict({k: v.clone() for k, v in cpu_model.state_dict().items()})
    mps_model = mps_model.to(torch.device("mps"))

    print(f"resource dir : {resource_dir}")
    print(f"torch threads: {torch.get_num_threads()}")
    print(f"search config: budget={BUDGET} candidate_limit={CANDIDATE_LIMIT} horizon={HORIZON}")
    print(f"cpu model dev: {cpu_model.device()}   mps model dev: {mps_model.device()}\n")

    env = PvZEnv(resource_dir=resource_dir)
    failures = 0
    totals: dict[str, list[float]] = {"cpu": [], "mps": []}
    try:
        if episode_mode:
            for seed in seeds:
                # Warm up on a throwaway seed so page faults / kernel compilation
                # do not land on the measured episode.
                measure_episode(env, cpu_model, seed)
                measure_episode(env, mps_model, seed)
                results = {}
                for name, model in (("cpu", cpu_model), ("mps", mps_model)):
                    results[name] = measure_episode(env, model, seed)
                    totals[name].append(results[name]["total_s"])
                cpu, mps = results["cpu"], results["mps"]
                same_actions = cpu["actions"] == mps["actions"]
                if not same_actions:
                    failures += 1
                print(f"seed {seed}: decisions cpu={cpu['decisions']} mps={mps['decisions']}  "
                      f"same_actions={same_actions}  won cpu={cpu['won']} mps={mps['won']}")
                print(f"    cpu  {cpu['total_s']:8.2f} s   {cpu['seconds_per_decision'] * 1e3:8.1f} ms/decision")
                print(f"    mps  {mps['total_s']:8.2f} s   {mps['seconds_per_decision'] * 1e3:8.1f} ms/decision")
                print(f"    episode is {mps['total_s'] / max(cpu['total_s'], 1e-9):5.2f}x slower on mps")
                if not same_actions:
                    for index, (left, right) in enumerate(zip(cpu["actions"], mps["actions"])):
                        if left != right:
                            print(f"      first divergence at decision {index}: {left} vs {right}")
                            break
        else:
            SearchTeacher._leaf_value = _timed_leaf_value
            for seed in seeds:
                observation, _ = env.reset(level=LEVEL, task=training_task(seed, LEVEL))
                if observation.get("terminal"):
                    print(f"seed {seed}: level already over, skipping")
                    continue

                # Warm up both paths so page faults and lazy MPS kernel compilation
                # do not land on the measured run.
                measure(env, observation, cpu_model)
                measure(env, observation, mps_model)

                results = {}
                for name, model in (("cpu", cpu_model), ("mps", mps_model)):
                    runs = [measure(env, observation, model) for _ in range(3)]
                    best = min(runs, key=lambda item: item["total_ms"])
                    results[name] = best
                    totals[name].append(best["total_ms"])

                identical = results["cpu"]["summary"] == results["mps"]["summary"]
                if not identical:
                    failures += 1
                cpu, mps = results["cpu"], results["mps"]
                print(f"seed {seed}: identical={identical}")
                print(f"    cpu  total {cpu['total_ms']:9.2f} ms   leaf {cpu['leaf_ms']:9.2f} ms "
                      f"({cpu['leaf_ms'] / cpu['total_ms'] * 100:5.1f}%)  {cpu['leaf_calls']} calls "
                      f"x {cpu['leaf_us_each']:7.2f} us")
                print(f"    mps  total {mps['total_ms']:9.2f} ms   leaf {mps['leaf_ms']:9.2f} ms "
                      f"({mps['leaf_ms'] / mps['total_ms'] * 100:5.1f}%)  {mps['leaf_calls']} calls "
                      f"x {mps['leaf_us_each']:7.2f} us")
                print(f"    leaf is {mps['leaf_us_each'] / max(cpu['leaf_us_each'], 1e-9):5.2f}x slower on mps; "
                      f"end-to-end {mps['total_ms'] / max(cpu['total_ms'], 1e-9):5.2f}x")
                if not identical:
                    print("     cpu:", cpu["summary"])
                    print("     mps:", mps["summary"])
    finally:
        SearchTeacher._leaf_value = _LEAF_ORIGINAL
        env.close()

    if totals["cpu"]:
        cpu_mean = sum(totals["cpu"]) / len(totals["cpu"])
        mps_mean = sum(totals["mps"]) / len(totals["mps"])
        unit = " s" if episode_mode else " ms"
        print(f"\nmean advice(): cpu {cpu_mean:.2f}{unit}   mps {mps_mean:.2f}{unit}   "
              f"mps/cpu = {mps_mean / cpu_mean:.2f}x")
    print(f"{len(seeds) - failures}/{len(seeds)} seeds agreed between cpu and mps")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
