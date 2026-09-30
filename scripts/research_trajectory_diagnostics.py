"""Read completed rollout manifests without loading observation arrays or changing training.

The state snapshot fixes the set of completed updates, so this can inspect a live
experiment. Raw per-episode results and summaries retain failures and truncations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_seed_jobs import atomic_json


def distribution(values: list[float] | np.ndarray) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"count": 0}
    if not np.isfinite(x).all():
        raise ValueError("nonfinite trajectory statistic")
    return {"count": int(len(x)), "mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "p05": float(np.quantile(x, .05)),
            "median": float(np.median(x)), "p95": float(np.quantile(x, .95)),
            "max": float(x.max()), "positive_fraction": float((x > 0).mean())}


def episode_signals(episode: dict, gamma: float, lam: float, shaping_weight: float,
                    deck: list[int] | None = None) -> dict:
    steps = episode["transitions"]
    if not steps:
        raise ValueError("empty completed trajectory")
    durations = np.array([s["action_duration_ticks"] for s in steps], dtype=np.float64)
    discounts = gamma ** (durations / 300)
    if not np.allclose(discounts, [s["discount"] for s in steps], rtol=0, atol=1e-12):
        raise ValueError("stored discounts do not match frozen reward config")
    rewards = np.array([s["reward"] for s in steps], dtype=np.float64)
    values = np.array([s["value"] for s in steps], dtype=np.float64)
    shaping = np.array([s["shaping_reward"] for s in steps], dtype=np.float64)
    terminal = np.array([s["terminal_outcome"] for s in steps], dtype=np.float64)
    if not np.allclose(rewards, shaping + terminal, rtol=0, atol=1e-12):
        raise ValueError("reward decomposition mismatch")
    prefix = np.concatenate(([1.0], np.cumprod(discounts)[:-1]))
    bootstrap = float(episode.get("bootstrap_value", 0))
    mc, adv, delta = np.empty(len(steps)), np.empty(len(steps)), np.empty(len(steps))
    tail_return, tail_adv = bootstrap, 0.0
    for i in range(len(steps) - 1, -1, -1):
        next_value = values[i + 1] if i + 1 < len(steps) else bootstrap
        delta[i] = rewards[i] + discounts[i] * next_value - values[i]
        tail_adv = delta[i] + discounts[i] * lam ** (durations[i] / 300) * tail_adv
        tail_return = rewards[i] + discounts[i] * tail_return
        adv[i], mc[i] = tail_adv, tail_return
    first_phi = steps[0]["potential"]
    if episode["terminated"]:
        last_phi = 0.0
    elif shaping_weight:
        last_phi = (shaping[-1] / shaping_weight + steps[-1]["potential"]) / discounts[-1]
    else:
        last_phi = None
    telescoping_error = (float(np.dot(prefix, shaping) - shaping_weight *
                               (discounts.prod() * last_phi - first_phi))
                         if last_phi is not None else None)
    immediate = 0
    for previous, current in zip(steps, steps[1:]):
        a, b = previous["action"], current["action"]
        immediate += (a["type"] == "plant" and b["type"] == "shovel" and
                      (a["row"], a["col"]) == (b["row"], b["col"]))
    counts = Counter(s["action"]["type"] for s in steps)
    plant_cards = [(str(deck[s["action"]["packet"]]) if deck else str(s["action"]["packet"]))
                   if s["action"]["type"] == "plant" else "none" for s in steps]
    return {"advantages": adv, "deltas": delta, "mc": mc, "values": values,
            "kinds": [s["action"]["type"] for s in steps], "durations": durations,
            "plant_cards": plant_cards,
            "shaping": shaping,
            "row": {"task_id": episode["task_id"], "job_id": episode["seed"],
                    "environment_seed": episode["task_seed"], "won": episode["won"],
                    "terminated": episode["terminated"], "truncated": episode["truncated"],
                    "wave": episode["wave"], "wave_count": episode["wave_count"],
                    "tick": episode["tick"], "actions": len(steps), "action_counts": dict(counts),
                    "plant_card_counts": dict(Counter(card for card in plant_cards if card != "none")),
                    "wait_choice_counts": dict(Counter(str(s["action"]["ticks"]) for s in steps
                                                       if s["action"]["type"] == "wait")),
                    "zero_tick_actions": int((durations == 0).sum()),
                    "immediate_plant_shovels": int(immediate), "first_mc_return": float(mc[0]),
                    "discounted_terminal_return": float(np.dot(prefix, terminal)),
                    "discounted_shaping_return": float(np.dot(prefix, shaping)),
                    "shaping_telescoping_error": telescoping_error,
                    "first_advantage": float(adv[0]), "last_advantage": float(adv[-1]),
                    "first_to_terminal_td_trace_weight": float(
                        (gamma * lam) ** (durations[:-1].sum() / 300)),
                    "simulated_ticks": int(durations.sum())}}


def summarize(signals: list[dict], normalized: list[np.ndarray]) -> dict:
    rows = [x["row"] for x in signals]
    def concatenate(key: str) -> np.ndarray:
        return np.concatenate([x[key] for x in signals])
    values, mc = concatenate("values"), concatenate("mc")
    advantages = concatenate("advantages")
    norm = np.concatenate(normalized)
    kinds = np.array([kind for x in signals for kind in x["kinds"]])
    plant_cards = np.array([card for x in signals for card in x["plant_cards"]])
    action_counts = Counter(kinds.tolist())
    plants = action_counts["plant"]
    immediate = sum(r["immediate_plant_shovels"] for r in rows)
    error = mc - values
    mc_variance = float(mc.var())
    return {"episodes": len(rows), "decisions": len(mc), "won": sum(r["won"] for r in rows),
            "truncated": sum(r["truncated"] for r in rows), "action_counts": dict(action_counts),
            "zero_tick_fraction": float((concatenate("durations") == 0).mean()),
            "immediate_plant_shovels": immediate,
            "immediate_shovel_fraction_of_plants": immediate / plants if plants else None,
            "mc_value_mse": float(np.mean(error ** 2)),
            "mc_value_explained_variance": 1 - float(error.var()) / mc_variance
                if mc_variance > 1e-12 else None,
            "mc_return": distribution(mc), "raw_advantage": distribution(advantages),
            "normalized_advantage": distribution(norm),
            "td_error": distribution(concatenate("deltas")),
            "absolute_shaping_reward": distribution(abs(concatenate("shaping"))),
            "discounted_terminal_return": distribution([r["discounted_terminal_return"] for r in rows]),
            "discounted_shaping_return": distribution([r["discounted_shaping_return"] for r in rows]),
            "first_to_terminal_td_trace_weight": distribution(
                [r["first_to_terminal_td_trace_weight"] for r in rows]),
            "max_abs_shaping_telescoping_error": max(
                (abs(r["shaping_telescoping_error"]) for r in rows
                 if r["shaping_telescoping_error"] is not None), default=None),
            "by_action": {kind: {"count": int((kinds == kind).sum()),
                                  "raw_advantage": distribution(advantages[kinds == kind]),
                                  "normalized_advantage": distribution(norm[kinds == kind])}
                          for kind in sorted(action_counts)},
            "by_plant_card": {card: {"count": int((plant_cards == card).sum()),
                                     "normalized_advantage": distribution(norm[plant_cards == card])}
                              for card in sorted(set(plant_cards) - {"none"})},
            "wait_choice_counts": dict(sum((Counter(r["wait_choice_counts"]) for r in rows), Counter()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    state_bytes = (args.experiment_dir / "training_state.json").read_bytes()
    state = json.loads(state_bytes)
    config = json.loads((args.experiment_dir / "experiment_config.json").read_text())
    provenance = json.loads((args.experiment_dir / "provenance.json").read_text())
    task_decks = {task["task_id"]: task["deck"] for task in
                  json.loads((ROOT / config["sampling"]["manifest"]).read_text())["tasks"]}
    all_signals, all_normalized, updates, raw = [], [], [], []
    by_task, by_task_normalized = defaultdict(list), defaultdict(list)
    for update in state["update_history"]:
        directory = args.experiment_dir / update["shard_directory"]
        files = sorted((p for p in directory.glob("seed_*.npz") if ".invalid_" not in p.name),
                       key=lambda p: int(p.stem.split("_")[1]))
        signals = []
        manifest_digest = hashlib.sha256()
        for path in files:
            with np.load(path, allow_pickle=False) as archive:
                manifest_bytes = archive["__manifest__"].tobytes()
            manifest_digest.update(path.name.encode() + b"\0" + manifest_bytes)
            stored = json.loads(manifest_bytes)["value"]
            episode = stored["result"]
            signals.append(episode_signals(episode, config["reward"]["gamma"],
                                           config["ppo"]["gae_lambda"],
                                           config["reward"]["shaping_weight"], task_decks[episode["task_id"]]))
        expected = sum(update["trajectory_stats"]["task_counts"].values())
        if len(signals) != expected:
            raise ValueError(f"update {update['update']}: expected {expected} shards, got {len(signals)}")
        adv = np.concatenate([x["advantages"] for x in signals])
        mean, std = float(adv.mean()), max(float(adv.std()), 1e-6)
        normalized = [(x["advantages"] - mean) / std for x in signals]
        summary = summarize(signals, normalized)
        updates.append({"update": update["update"], "counters": update["counters"],
                        "manifest_sha256": manifest_digest.hexdigest(), "summary": summary})
        for signal, norm in zip(signals, normalized):
            row = {"update": update["update"], **signal["row"]}
            raw.append(row)
            by_task[row["task_id"]].append(signal)
            by_task_normalized[row["task_id"]].append(norm)
        all_signals.extend(signals)
        all_normalized.extend(normalized)
        print(f"diagnosed update {update['update']}: {len(signals)} episodes / {len(adv)} decisions", flush=True)
    if not all_signals:
        raise ValueError("no completed updates to diagnose")
    raw_path = args.output.with_suffix(".episodes.json.gz")
    atomic_json(raw_path, raw, compressed=True)
    atomic_json(args.output, {"schema_version": 1, "experiment_id": config["experiment_id"],
                              "scope": "every episode in every completed update of the state snapshot",
                              "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
                              "source_provenance": provenance, "config": config,
                              "counters": state["counters"], "snapshot_phase": state["phase"],
                              "normalization": "per completed update, population std; float64 offline recomputation",
                              "mc_semantics": "duration-discounted rewards plus stored truncation bootstrap",
                              "trace_weight_semantics": "terminal TD error coefficient in first GAE, not a memory gradient",
                              "plant_card_semantics": "actual seed_type from the frozen task deck, not packet position",
                              "summary": summarize(all_signals, all_normalized), "updates": updates,
                              "per_task": {k: summarize(v, by_task_normalized[k]) for k, v in by_task.items()},
                              "raw_episode_results": str(raw_path),
                              "seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
