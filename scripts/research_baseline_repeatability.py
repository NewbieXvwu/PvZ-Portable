"""Repeat the preregistered anomalous baseline job without changing training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch
from pvz_agent_model import configure_torch_threads
from pvz_seed_jobs import atomic_json
import train_pvz_ppo_task_family as family


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--resource-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh diagnostic output required")
    protocol = json.loads(args.protocol.read_text())
    configure_torch_threads(protocol["worker_threads"])
    checkpoint_path = ROOT / protocol["reference_checkpoint"]
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if protocol.get("reference_checkpoint_sha256", checkpoint_sha) != checkpoint_sha:
        raise ValueError("reference checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["experiment_config"]
    manifest = json.loads((ROOT / config["evaluation"]["manifest"]).read_text())
    task = next(t for t in manifest["tasks"] if t["task_id"] == protocol["task_id"])
    if protocol["environment_seed"] not in task["seeds"] or protocol["modes"] != ["sampled"]:
        raise ValueError("diagnostic must preserve the original sampled evaluation job")
    jobs = {i: {"task": task, "seed": protocol["environment_seed"],
                "action_seed": protocol["action_seed"], "deterministic": False,
                "max_actions": protocol["max_actions"], "allow_truncation": True}
            for i in range(protocol["repeats_per_configuration"])}
    started = time.monotonic()
    rows = {}
    for workers in protocol["workers"]:
        rows[str(workers)] = family._run_evaluation_jobs(
            jobs, checkpoint["state_dict"], args.resource_dir, workers,
            protocol["worker_threads"], "cpu", checkpoint["config"])
        atomic_json(args.output, {"status": "partial", "records": rows})
    all_rows = [row for group in rows.values() for row in group]
    same = all(row == all_rows[0] for row in all_rows)
    label_keys = ("won", "result", "terminated", "truncated", "terminal_wave", "wave_count")
    outcome_keys = (*label_keys, "terminal_tick")
    if any(any(key not in row for key in outcome_keys) for row in all_rows):
        raise ValueError("returned record lacks required outcome fields")
    same_outcomes = all(all(row[k] == all_rows[0][k] for k in outcome_keys)
                        for row in all_rows)
    atomic_json(args.output, {
        "schema_version": 2, "status": "complete", "scope": protocol["scope"],
        "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha,
        "records": rows, "all_returned_fields_exact": same,
        "all_outcome_fields_exact": same_outcomes,
        "outcome_keys": outcome_keys,
        "all_win_terminal_labels_exact": all(all(row[k] == all_rows[0][k] for k in label_keys)
                                             for row in all_rows),
        "seconds": time.monotonic() - started,
        "interpretation": "Finite repeatability diagnostic; does not retroactively replace original rows."})
    print("completed", len(all_rows), "exact", same, "outcomes_exact", same_outcomes, flush=True)


if __name__ == "__main__":
    main()
