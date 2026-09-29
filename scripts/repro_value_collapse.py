"""Reproduce the SearchValue bootstrapping loop locally, at small scale.

The real pipeline (train_pvz_agent.py) does:

    bootstrap episodes (cold-start teacher) -> train SearchValueModel
    -> refinement episodes (teacher + value model) -> train again -> checkpoint

Every episode in that loop is a *loss* (the teacher has never won), so the only
signal the value model can learn is "how long until I die", discounted with
gamma=0.99 per 300 ticks.  This script repeats the loop with 3 episodes instead
of 32 and measures three things the TODO numbers do not show:

1. how much the trained model's prediction actually varies between sibling
   actions at one decision (the quantity the search maximises), versus
   the model's own training error (RMSE);
2. whether the model is monotonic in elapsed time, i.e. whether it pays the
   search to sit on zero-tick actions instead of advancing the clock;
3. the terminal wave reached by the teacher *with* the trained value model,
   against the same teacher with the cold-start evaluator.

    python scripts/repro_value_collapse.py --resource-dir <PvZ 1.2.0.1073 dir>
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import torch  # noqa: E402

from pvz_env import PvZEnv, training_task  # noqa: E402
from pvz_search import SearchTeacher  # noqa: E402
from pvz_search_value import SearchValueModel, train_search_value  # noqa: E402
from pvz_value import DISCOUNT_REFERENCE_TICKS, VALUE_GAMMA, discounted_terminal_value  # noqa: E402

DECK = (0, 1, 2, 3, 4, 5)
LEVEL = 7


def play_episode(env: PvZEnv, teacher: SearchTeacher, seed: int, max_actions: int,
                 trace: bool = False) -> dict:
    observation, _ = env.reset(deck=DECK, task=training_task(seed, LEVEL))
    steps: list[dict] = []
    for index in range(max_actions):
        advice = teacher.advice(observation)
        steps.append({"observation": observation, "tick": observation["tick"],
                      "effective_depth_budget": advice.effective_depth_budget,
                      "simulations": advice.simulation_count})
        if trace and index % 20 == 0:
            print(f"   #{index} t={observation['tick']} w={observation['wave']} "
                  f"sun={observation['sun']} -> {advice.action}")
        observation, _, done, _, info = env.step(advice.action)
        if not info.get("ok"):
            raise RuntimeError(f"illegal action on seed {seed}: {advice.action}")
        if done:
            break
    return {
        "seed": seed,
        "won": bool(observation.get("result") == 1),
        "tick": int(observation["tick"]),
        "wave": int(observation["wave"]),
        "wave_count": int(observation["wave_count"]),
        "actions": len(steps),
        "steps": steps,
    }


def label_scale(episode: dict) -> dict:
    """How much the *training target itself* moves over a search horizon."""
    rows = []
    for step in episode["steps"]:
        remaining = max(0, episode["tick"] - int(step["observation"]["tick"]))
        rows.append(discounted_terminal_value(episode["won"], remaining))
    early = [value for value, step in zip(rows, episode["steps"])
             if step["observation"]["tick"] < 20000]
    return {
        "min": min(rows), "max": max(rows),
        "range": max(rows) - min(rows),
        "per_300_ticks": abs(rows[len(rows) // 2]) * (1 - VALUE_GAMMA ** (300 / DISCOUNT_REFERENCE_TICKS)),
        "early_mean": statistics.fmean(early) if early else None,
    }


def sibling_spread(env: PvZEnv, teacher: SearchTeacher, observation: dict) -> dict:
    """Predictions for the one-step children of the root candidates."""
    from pvz_search_candidates import CandidateGenerator
    actions = CandidateGenerator().actions(observation, teacher.root_request_limit, True,
                                           teacher.horizon_ticks)
    children = teacher.one_step_children(observation, actions)
    values = [teacher._leaf_value(child) for _, child in children if child is not None]
    if not values:
        return {}
    return {
        "actions": len(children),
        "min": min(values), "max": max(values), "spread": max(values) - min(values),
        "stdev": statistics.pstdev(values),
    }


def monotonic_in_time(env: PvZEnv, teacher: SearchTeacher, observation: dict) -> dict:
    """Predicted value of 'do nothing for k ticks' -- the waiting penalty."""
    with teacher._speculative_fast() as snapshot:
        teacher._root_snapshot = snapshot
        teacher._snapshots = set()
        try:
            rows = []
            for ticks in (0, 60, 150, 300, 600, 900):
                branches = teacher._branch_snapshot_fast(snapshot, [{"type": "wait", "ticks": ticks or 1}])
                if not branches or branches[0].get("observation") is None:
                    continue
                rows.append((ticks, teacher._leaf_value(branches[0]["observation"])))
        finally:
            for snapshot_id in tuple(teacher._snapshots):
                teacher._release(snapshot_id)
    return {"wait_curve": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20000)
    parser.add_argument("--eval-seed", type=int, default=30000)
    parser.add_argument("--max-actions", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    env = PvZEnv(args.resource_dir)
    cold = SearchTeacher(env)
    episodes = []
    for index in range(args.episodes):
        seed = args.seed_start + index
        started = time.perf_counter()
        episode = play_episode(env, cold, seed, args.max_actions)
        episodes.append(episode)
        print(f"bootstrap seed {seed}: wave {episode['wave']}/{episode['wave_count']} "
              f"won={episode['won']} actions={episode['actions']} "
              f"tick={episode['tick']} ({time.perf_counter() - started:.0f}s)")

    rows = sum(len(episode["steps"]) for episode in episodes)
    params = sum(tensor.numel() for tensor in SearchValueModel().parameters())
    print(f"\ntraining rows={rows}  model params={params}  rows/param={rows / params:.4f}")
    for episode in episodes:
        scale = label_scale(episode)
        print(f"  seed {episode['seed']}: target range {scale['range']:.4f} "
              f"[{scale['min']:.3f}, {scale['max']:.3f}], "
              f"change per 300 ticks ~{scale['per_300_ticks']:.5f}")

    model = SearchValueModel()
    losses = train_search_value(model, episodes, args.epochs, torch.device("cpu"))
    print(f"loss history: {[round(x, 6) for x in losses]}")
    print(f"final train RMSE ~ {losses[-1] ** 0.5:.4f}")

    warm = SearchTeacher(env, value_model=model)
    observation, _ = env.reset(deck=DECK, task=training_task(args.eval_seed, LEVEL))
    print("\nsibling spread (cold-start evaluator):", sibling_spread(env, cold, observation))
    print("sibling spread (trained value model):", sibling_spread(env, warm, observation))
    print("wait curve (cold):", monotonic_in_time(env, cold, observation))
    print("wait curve (warm):", monotonic_in_time(env, warm, observation))

    started = time.perf_counter()
    with_model = play_episode(env, warm, args.eval_seed, args.max_actions, trace=True)
    print(f"\nteacher + trained value model on seed {args.eval_seed}: "
          f"wave {with_model['wave']}/{with_model['wave_count']} "
          f"won={with_model['won']} actions={with_model['actions']} "
          f"({time.perf_counter() - started:.0f}s)")
    env.close()


if __name__ == "__main__":
    main()
