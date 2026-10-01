"""Compare two existing evaluation nodes on every frozen task/seed, without rerunning policies."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_seed_jobs import atomic_json
from research_comparison_summary import outcomes, summarize_evaluation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_node(directory: Path, state: dict, decisions: int) -> tuple[dict, dict, Path]:
    points = [p for p in state["learning_curve"] if p["counters"]["decisions"] == decisions]
    if len(points) != 1:
        raise ValueError("require one completed node at the exact requested decision count")
    point = points[0]
    raw = directory / point["raw_seed_results_path"]
    with gzip.open(raw, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload["counters"] != point["counters"] or payload["experiment_id"] != state["experiment_id"]:
        raise ValueError("node identity/counters disagree with frozen state")
    return point, payload, raw


def paired_summary(pairs: list[dict]) -> dict:
    before = [p["before"] for p in pairs]
    after = [p["after"] for p in pairs]
    transitions = Counter((p["before"]["won"], p["after"]["won"]) for p in pairs)
    def behavior(rows: list[dict]) -> dict:
        counts = sum((Counter(r["action_counts"]) for r in rows), Counter())
        actions = sum(counts.values())
        return {"mean_peak_offense": sum(r["peak_offense"] for r in rows) / len(rows),
                "mean_actions": sum(r["actions"] for r in rows) / len(rows),
                "action_counts": dict(counts),
                "wait_fraction": counts["wait"] / actions if actions else None,
                "immediate_plant_shovels": sum(r["immediate_plant_shovels"] for r in rows)}
    return {"before": outcomes(before), "after": outcomes(after),
            "lost_previous_wins": transitions[True, False],
            "gained_new_wins": transitions[False, True],
            "retained_wins": transitions[True, True],
            "retained_failures": transitions[False, False],
            "before_behavior": behavior(before), "after_behavior": behavior(after)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--before-decisions", type=int, required=True)
    parser.add_argument("--after-decisions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.experiment_dir = args.experiment_dir.resolve()
    args.output = args.output.resolve()
    # Validate the evidence location before creating either report file.
    args.output.relative_to(ROOT)
    started = time.monotonic()
    raw_output = args.output.with_suffix(".pairs.json.gz")
    if args.output.exists() or raw_output.exists():
        raise ValueError("keep existing reports; use a fresh output path")
    state_path = args.experiment_dir / "training_state.json"
    state = json.loads(state_path.read_text())
    config = json.loads((args.experiment_dir / "experiment_config.json").read_text())
    manifest = ROOT / config["evaluation"]["manifest"]
    tasks = json.loads(manifest.read_text())["tasks"]
    before_point, before, before_path = read_node(args.experiment_dir, state, args.before_decisions)
    after_point, after, after_path = read_node(args.experiment_dir, state, args.after_decisions)
    if args.before_decisions >= args.after_decisions or before["model_config"] != after["model_config"]:
        raise ValueError("nodes must be ordered with the same model configuration")
    for payload in (before, after):
        summarize_evaluation(payload, tasks, config["evaluation"]["modes"])
    rows, modes = [], {}
    for mode in config["evaluation"]["modes"]:
        by_task, by_cap = {}, {}
        for task in tasks:
            task_id = task["task_id"]
            left = {r["seed"]: r for r in before["seed_results"][mode][task_id]}
            right = {r["seed"]: r for r in after["seed_results"][mode][task_id]}
            pairs = [{"mode": mode, "task_id": task_id, "seed": seed,
                      "terrain": task["terrain"], "wave_cap": task["wave_cap"],
                      "zombie_count_multiplier": task["zombie_count_multiplier"],
                      "evaluation_role": task["evaluation_role"],
                      "before": left[seed], "after": right[seed]} for seed in sorted(left)]
            rows.extend(pairs)
            by_task[task_id] = paired_summary(pairs)
            if task["evaluation_role"] != "training_probe":
                by_cap.setdefault(str(task["wave_cap"]), []).extend(pairs)
        modes[mode] = {"per_task": by_task,
                       "validation_by_cap": {cap: paired_summary(pairs) for cap, pairs in by_cap.items()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw_output, "xt", encoding="utf-8") as stream:
        json.dump(rows, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    atomic_json(args.output, {"schema_version": 1, "experiment_id": state["experiment_id"],
                             "scope": "all paired seeds at two completed preregistered nodes; descriptive, not causal",
                             "manifest": str(manifest.relative_to(ROOT)), "manifest_sha256": digest(manifest),
                             "state_sha256": digest(state_path),
                             "before": {"counters": before_point["counters"], "raw_sha256": digest(before_path)},
                             "after": {"counters": after_point["counters"], "raw_sha256": digest(after_path)},
                             "seed_pairs": len(rows), "summary": modes,
                             "raw_pairs": str(raw_output.relative_to(ROOT)), "raw_pairs_sha256": digest(raw_output),
                             "seconds": time.monotonic() - started})
    print(f"paired {len(rows)} task/seed/mode results; all failures and truncations retained", flush=True)


if __name__ == "__main__":
    main()
