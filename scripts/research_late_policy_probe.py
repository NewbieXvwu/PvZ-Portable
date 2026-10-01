"""Read-only fixed-history comparison of two already frozen evaluation nodes."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import torch
from pvz_agent_model import GameplayModelV1, configure_torch_threads, select_action
from pvz_common import sha256_file
from pvz_seed_jobs import atomic_json, read_numpy, read_episode


def changes(before, after, prefix=""):
    old_square, delta_square = 0.0, 0.0
    for name in before:
        if name.startswith(prefix):
            old, new = before[name].double(), after[name].double()
            old_square += float(old.square().sum())
            delta_square += float((new - old).square().sum())
    return {"l2_change": math.sqrt(delta_square), "relative_l2_change": math.sqrt(delta_square / old_square) if old_square else None}


def summarize(rows):
    count = len(rows)
    return {"decisions": count, "legal_plant_decisions": sum(row["legal_plant"] for row in rows),
            "mean_old_wait_probability": sum(row["type_probability_before"][2] for row in rows) / count,
            "mean_new_wait_probability": sum(row["type_probability_after"][2] for row in rows) / count,
            "mean_type_kl_before_after": sum(row["type_kl_before_after"] for row in rows) / count,
            "mean_type_js": sum(row["type_js"] for row in rows) / count,
            "greedy_action_changes": sum(row["greedy_before"] != row["greedy_after"] for row in rows),
            "greedy_type_changes": sum(row["greedy_before"]["type"] != row["greedy_after"]["type"] for row in rows),
            "greedy_type_transitions": dict(Counter(row["greedy_before"]["type"] + "->" + row["greedy_after"]["type"] for row in rows)),
            "old_type_probability_margin_below": {str(limit): sum(row["type_margin_before"] <= limit for row in rows) for limit in (.01, .05, .10, .25)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.experiment_dir.resolve(), args.output.resolve()
    output.relative_to(ROOT)
    raw_output = output.with_suffix(".rows.json.gz")
    if output.exists() or raw_output.exists():
        raise ValueError("retain existing diagnostics; use a fresh path")
    configure_torch_threads(1)
    started = time.monotonic()
    state = json.loads((source / "training_state.json").read_text())
    before_point = next(p for p in state["learning_curve"] if 375000 <= p["counters"]["decisions"] < 500000)
    after_point = next(p for p in state["learning_curve"] if p["counters"]["decisions"] >= 500000)
    checkpoints = []
    models = []
    for point in (before_point, after_point):
        path = sorted((source / "runs/run_1").glob(f"update_{point['updates']:06d}_evaluated_*.pt"))[0]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["training_state"]["counters"] != point["counters"]:
            raise ValueError("checkpoint/node mismatch")
        model = GameplayModelV1(saved["config"]).eval()
        model.load_state_dict(saved["state_dict"])
        models.append(model)
        checkpoints.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "counters": point["counters"]})
        del saved
    update = next(item for item in state["update_history"] if item["update"] == before_point["updates"] + 1)
    shard_dir = source / update["shard_directory"]
    first_shard = sorted(shard_dir.glob("seed_*.npz"))[0]
    assignments = read_numpy(first_shard)["metadata"]["assignments"]
    selected, seen = [], Counter()
    for job in sorted(assignments, key=int):
        terrain = assignments[job]["task"]["terrain"]
        if seen[terrain] < 2:
            selected.append((int(job), terrain, assignments[job]["task"]["task_id"]))
            seen[terrain] += 1
    if len(seen) != 5 or any(n != 2 for n in seen.values()):
        raise ValueError("frozen diagnostic requires the first two original jobs of every terrain")
    rows, shards = [], []
    maximum_replay_error = 0.0
    for job, terrain, task_id in selected:
        path = shard_dir / f"seed_{job}.npz"
        episode = read_episode(path)
        if episode["task_id"] != task_id or episode["seed"] != job:
            raise ValueError("assignment/shard mismatch")
        shards.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "job": job,
                       "terrain": terrain, "task_id": task_id, "decisions": len(episode["transitions"])})
        hiddens = [None, None]
        with torch.no_grad():
            for offset in range(0, len(episode["transitions"]), 64):
                transitions = episode["transitions"][offset:offset + 64]
                paired_outputs = []
                for index, model in enumerate(models):
                    outputs, hidden = model.forward_sequences([transitions], [hiddens[index]])
                    paired_outputs.append(outputs)
                    hiddens[index] = hidden[:, 0, :]
                for index, (step, old, new) in enumerate(zip(transitions, *paired_outputs, strict=True)):
                    legal = step["legal"]
                    probabilities, greedy, logprobs, entropies, margins = [], [], [], [], []
                    for model, result in zip(models, (old, new), strict=True):
                        mask = torch.tensor([bool(legal["packets"]), bool(legal["shovel_mask"]), bool(legal["wait"])])
                        probability = torch.softmax(result["type_logits"].masked_fill(~mask, torch.finfo(torch.float32).min), -1)
                        probabilities.append(probability)
                        ranked = probability.sort(descending=True).values
                        margins.append(float(ranked[0] - ranked[1]))
                        action, _, _ = select_action(model, result, legal, deterministic=True)
                        greedy.append(action)
                        _, log_prob, entropy = select_action(model, result, legal, action=step["action"])
                        logprobs.append(float(log_prob)); entropies.append(float(entropy))
                    p, q = probabilities
                    m = (p + q) / 2
                    kl = float((p * (p.clamp_min(1e-30).log() - q.clamp_min(1e-30).log())).sum())
                    js = float(((p * (p.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum()
                              + (q * (q.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum()) / 2)
                    maximum_replay_error = max(maximum_replay_error, abs(logprobs[0] - step["log_prob"]))
                    rows.append({"job": job, "terrain": terrain, "task_id": task_id, "step": offset + index,
                        "legal_plant": bool(legal["packets"]), "type_probability_before": p.tolist(),
                        "type_probability_after": q.tolist(), "type_margin_before": margins[0], "type_margin_after": margins[1],
                        "type_kl_before_after": kl, "type_js": js, "greedy_before": greedy[0], "greedy_after": greedy[1],
                        "recorded_action_log_prob": step["log_prob"], "replayed_log_prob_before": logprobs[0],
                        "replayed_log_prob_after": logprobs[1], "recorded_action_branch_entropy_before": entropies[0],
                        "recorded_action_branch_entropy_after": entropies[1],
                        "wait_branch_entropy_before": float(torch.distributions.Categorical(logits=old["wait_logits"]).entropy()),
                        "wait_branch_entropy_after": float(torch.distributions.Categorical(logits=new["wait_logits"]).entropy())})
        print(f"probed job={job} terrain={terrain} decisions={len(episode['transitions'])}", flush=True)
    if maximum_replay_error > 3e-6:
        raise RuntimeError(f"old checkpoint replay differs from original collection: {maximum_replay_error}")
    tail = []
    previous_weights = None
    for item in state["update_history"][-8:]:
        path = sorted((source / "runs/run_1").glob(f"update_{item['update']:06d}_trained_*.pt"))[0]
        saved = torch.load(path, map_location="cpu", weights_only=False)
        weights = saved["state_dict"]
        if previous_weights is not None:
            previous = state["update_history"][item["update"] - 2]["counters"]
            tail.append({"update": item["update"], "checkpoint": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                "batch_episodes": item["counters"]["episodes"] - previous["episodes"], "losses": item["losses"],
                "parameter_change": changes(previous_weights, weights), "action_type_change": changes(previous_weights, weights, "action_type."),
                "wait_duration_change": changes(previous_weights, weights, "wait_duration.")})
        previous_weights = weights
        del saved
    atomic_json(raw_output, {"checkpoints": checkpoints, "shards": shards, "rows": rows}, compressed=True)
    atomic_json(output, {"schema_version": 1, "scope": "off-policy fixed-public-history distribution diagnostic, not deployment win rate or causal attribution",
        "experiment_id": state["experiment_id"], "checkpoints": checkpoints, "shards": shards,
        "old_policy_collection_replay_max_error": maximum_replay_error, "overall": summarize(rows),
        "per_terrain": {terrain: summarize([row for row in rows if row["terrain"] == terrain]) for terrain in sorted(seen)},
        "legal_plant_states": summarize([row for row in rows if row["legal_plant"]]), "last_seven_updates": tail,
        "raw_rows": str(raw_output.relative_to(ROOT)), "raw_sha256": sha256_file(raw_output),
        "seconds": time.monotonic() - started, "coexecution": "CPU diagnostic overlaps the original reward matrix"})
    print("diagnostic complete", summarize(rows), flush=True)


if __name__ == "__main__":
    main()
