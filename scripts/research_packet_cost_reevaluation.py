"""Reevaluate every frozen reward node on the guarded native binary.

Original training, checkpoints, curves and evaluation rows are read-only. A
completed matrix is required; waiting never imports Torch or starts workers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import gzip
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
from pvz_common import canonical_digest, git_metadata, sha256_file
from pvz_seed_jobs import atomic_json, run_seed_jobs
from research_comparison_summary import comparable_fingerprints, storage_revision, summarize_evaluation


def validate_curve(state: dict, config: dict) -> list[dict]:
    points = state["learning_curve"]
    nodes = [0, *config["evaluation"]["decision_nodes"]]
    if (state["status"] != "budget_complete" or len(points) != len(nodes)
            or state["counters"]["decisions"] < config["budget"]["decisions"]
            or points[-1]["counters"] != state["counters"]
            or points[-1]["updates"] != state["updates"]):
        raise ValueError("require every original node and completed budget")
    for i, (node, point) in enumerate(zip(nodes, points, strict=True)):
        actual = point["counters"]["decisions"]
        if (actual < node or (i == 0 and (actual != 0 or point["updates"] != 0))
                or (i + 1 < len(nodes) and actual >= nodes[i + 1])
                or (i and point["updates"] <= points[i - 1]["updates"])):
            raise ValueError("evaluation nodes missing, duplicated or reordered")
    return points


def paired_changes(before: dict, after: dict, tasks: list[dict], modes: list[str],
                   allowed_task_ids: list[str]) -> tuple[dict, list[dict]]:
    # Also rejects missing/repeated/substituted seeds and inconsistent win labels.
    summarize_evaluation(before, tasks, modes)
    summarize_evaluation(after, tasks, modes)
    details, counts = [], Counter()
    for mode in modes:
        for task in tasks:
            task_id = task["task_id"]
            left = {r["seed"]: r for r in before["seed_results"][mode][task_id]}
            right = {r["seed"]: r for r in after["seed_results"][mode][task_id]}
            for seed in task["seeds"]:
                a, b = left[seed], right[seed]
                fields = [k for k in sorted(set(a) | set(b))
                          if k not in a or k not in b or a[k] != b[k]]
                if fields:
                    allowed = task_id in allowed_task_ids
                    details.append({"mode": mode, "task_id": task_id, "seed": seed,
                                    "allowed_conveyor": allowed, "changed_fields": fields,
                                    "before": a, "after": b})
                    counts["changed_rows"] += 1
                    counts["conveyor_changed_rows" if allowed else "unexpected_changed_rows"] += 1
                    counts["win_label_changes"] += a["won"] != b["won"]
    return {k: counts[k] for k in ("changed_rows", "conveyor_changed_rows",
                                  "unexpected_changed_rows", "win_label_changes")}, details


def initialize(resource_dir: str, weights: dict, jobs: dict, threads: int,
               model_config: dict, executable: str) -> None:
    import train_pvz_ppo_task_family as family
    original = family.PvZEnv
    try:
        family.PvZEnv = lambda resource_dir: original(resource_dir, executable=executable)
        family._init_evaluation_worker(resource_dir, weights, jobs, threads, "cpu", model_config)
    finally:
        family.PvZEnv = original


def worker(job_id: int) -> dict:
    import train_pvz_ppo_task_family as family
    returned, record = family._evaluation_worker(job_id)
    if returned != job_id:
        raise ValueError("evaluation returned a different job")
    return {"seed": job_id, "record": record}


def matrix_ready(queue: dict) -> bool:
    for entry in queue["order"]:
        path = ROOT / entry["output_dir"] / "training_state.json"
        if not path.exists():
            return False
        state = json.loads(path.read_text())
        if state["status"] != "budget_complete":
            return False
        validate_curve(state, json.loads((ROOT / entry["config"]).read_text()))
    return True


def checked_protocol(path: Path, resource_dir: Path) -> tuple[dict, dict, list[dict]]:
    protocol = json.loads(path.read_text())
    if sha256_file(Path(__file__)) != protocol["helper_sha256"]:
        raise ValueError("helper changed after preregistration")
    for name, expected in protocol["required_fingerprints"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"preregistered fingerprint changed: {name}")
    for name, expected in protocol["resource_fingerprints"].items():
        if sha256_file(resource_dir / name) != expected:
            raise ValueError(f"resource fingerprint changed: {name}")
    for name in protocol["required_gates"]:
        if json.loads((ROOT / name).read_text()).get("result") != "pass":
            raise ValueError(f"native prerequisite not passed: {name}")
    queue = json.loads((ROOT / protocol["queue"]).read_text())
    storage_revision(queue)
    if len(queue["order"]) != 12 or queue["required_initializations"] != [0, 1, 2]:
        raise ValueError("the entire four-reward three-initialization matrix is required")
    tasks = json.loads((ROOT / protocol["evaluation_manifest"]).read_text())["tasks"]
    if (len(tasks) != 35 or any(len(t["seeds"]) != 64 for t in tasks)
            or [t["task_id"] for t in tasks] != protocol["task_ids"]):
        raise ValueError("all35 original tasks and64 seeds per task are required")
    if protocol["allowed_changed_task_ids"] != [t["task_id"] for t in tasks if t["level"] in (10, 20, 30, 40)]:
        raise ValueError("only the four diagnosed conveyor levels permit changed rows")
    return protocol, queue, tasks


def make_inventory(queue: dict, protocol: dict, tasks: list[dict]) -> dict:
    nodes, states, paired = [], {}, {}
    audit = storage_revision(queue)
    original_fingerprints = None
    for entry in queue["order"]:
        directory = ROOT / entry["output_dir"]
        state_path = directory / "training_state.json"
        state = json.loads(state_path.read_text())
        config = json.loads((ROOT / entry["config"]).read_text())
        saved = json.loads((directory / "experiment_config.json").read_text())
        provenance = json.loads((directory / "provenance.json").read_text())
        fingerprints = comparable_fingerprints(provenance["fingerprints"], audit)
        if original_fingerprints is not None and fingerprints != original_fingerprints:
            raise ValueError("original matrix provenance drifted beyond approved retention change")
        original_fingerprints = fingerprints
        if fingerprints != protocol["original_comparable_fingerprints"]:
            raise ValueError("original core, simulator, resources or manifests changed")
        if config != saved or config["evaluation"]["modes"] != protocol["modes"]:
            raise ValueError("saved configuration or evaluation modes changed")
        if config["evaluation"]["manifest"] != protocol["evaluation_manifest"]:
            raise ValueError("candidate uses a different evaluation manifest")
        for key in ("workers", "worker_threads", "max_actions"):
            if config["runtime"][key] != protocol["runtime"][key]:
                raise ValueError(f"original evaluation runtime differs: {key}")
        seed = config["initialization_seed"]
        initial = state["initial_state_sha256"]
        if seed in paired and paired[seed] != initial:
            raise ValueError("reward arms did not share their paired initialization")
        paired[seed] = initial
        states[str(state_path.relative_to(ROOT))] = sha256_file(state_path)
        for index, point in enumerate(validate_curve(state, config)):
            matches = list((directory / "runs/run_1").glob(
                f"update_{point['updates']:06d}_evaluated_*.pt"))
            if len(matches) != 1:
                raise ValueError("require exactly one original evaluated checkpoint per node")
            checkpoint = matches[0]
            raw = directory / point["raw_seed_results_path"]
            with gzip.open(raw, "rt") as stream:
                original = json.load(stream)
            if original["counters"] != point["counters"] or original["experiment_id"] != config["experiment_id"]:
                raise ValueError("original raw identity/counters mismatch")
            summarize_evaluation(original, tasks, protocol["modes"])
            nodes.append({"experiment_id": config["experiment_id"], "node_index": index,
                          "initialization_seed": seed, "reward": config["reward"]["name"],
                          "counters": point["counters"], "updates": point["updates"],
                          "checkpoint": str(checkpoint.relative_to(ROOT)),
                          "checkpoint_sha256": sha256_file(checkpoint),
                          "config": entry["config"], "original_raw": str(raw.relative_to(ROOT)),
                          "original_raw_sha256": sha256_file(raw),
                          "original_training_wall_seconds": None})
    if len(nodes) != 60:
        raise ValueError("all60 nodes are required")
    known = {n["checkpoint"]: n["checkpoint_sha256"] for n in nodes}
    for path, expected in protocol["already_existing_checkpoints"].items():
        if known.get(path) != expected:
            raise ValueError("an already existing node was replaced")
    return {"nodes": nodes, "original_states": states, "paired_initial_states": paired}


def compact(results: list[dict]) -> list[dict]:
    return [{**{k: v for k, v in r.items() if k != "summary"},
             "validation": {mode: {f"cap{cap}": r["summary"][mode]["cohorts"][f"validation/cap{cap}"]
                                    for cap in (1, 3, 5)} for mode in r["summary"]}}
            for r in results]


def publish(output: Path, summary_path: Path, state: dict, protocol: dict) -> None:
    atomic_json(output / "state.json", state)
    atomic_json(summary_path, {"schema_version": 1, "status": state["status"],
                              "protocol_sha256": state["protocol_sha256"],
                              "inventory_sha256": state.get("inventory_sha256"),
                              "completed_nodes": len(state["results"]), "required_nodes": 60,
                              "executable": protocol["executable"],
                              "simulator_sha256": protocol["required_fingerprints"][protocol["executable"]],
                              "expected_jobs_per_node": 4480,
                              "results": compact(state["results"]),
                              "raw_report": str((output / "state.json").relative_to(ROOT)),
                              "raw_report_sha256": sha256_file(output / "state.json"),
                              "delivery": "Raw shards, evaluations and logs local pending HF; no learning gate declared",
                              "timing": "Reevaluation wall is separate from original training wall; no equal-wall interpolation"})


def evaluate_node(node: dict, protocol: dict, tasks: list[dict], resource_dir: Path,
                  output: Path) -> dict:
    import torch
    import pvz_research
    import t4_capability_profile as profile
    checkpoint_path = ROOT / node["checkpoint"]
    if (sha256_file(checkpoint_path) != node["checkpoint_sha256"]
            or sha256_file(ROOT / node["original_raw"]) != node["original_raw_sha256"]):
        raise ValueError("original checkpoint/raw changed after inventory freeze")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = json.loads((ROOT / node["config"]).read_text())
    snapshot = checkpoint["training_state"]
    if (checkpoint["experiment_config"] != config or snapshot["counters"] != node["counters"]
            or snapshot["updates"] != node["updates"]):
        raise ValueError("checkpoint does not match its original node/config")
    weights = checkpoint["state_dict"]
    weight_hash = profile._state_sha256(weights)
    if node["node_index"] == 0 and weight_hash != snapshot["initial_state_sha256"]:
        raise ValueError("untrained node is not the actual original initialization")
    jobs, labels = {}, []
    for mode in protocol["modes"]:
        for task in tasks:
            for seed in task["seeds"]:
                labels.append((mode, task["task_id"]))
                jobs[len(jobs)] = {"task": task, "seed": seed, "action_seed": seed + 170_000,
                                   "deterministic": mode == "greedy", "allow_truncation": True,
                                   "max_actions": protocol["runtime"]["max_actions"]}
    identity = {"node": node, "model_state_sha256": weight_hash,
                "model_config": checkpoint["config"], "protocol": protocol, "jobs": jobs}
    job_identity = canonical_digest(identity)
    destination = output / node["experiment_id"] / f"node_{node['node_index']}"
    shard_dir = destination / ".seed_jobs" / job_identity
    monitor = pvz_research.ResourceMonitor()
    monitor.thread.start()
    started = time.monotonic()
    try:
        rows = run_seed_jobs(list(jobs), shard_dir, {"identity": job_identity}, worker,
                             workers=protocol["runtime"]["workers"], initializer=initialize,
                             initargs=(str(resource_dir), weights, jobs,
                                       protocol["runtime"]["worker_threads"], checkpoint["config"],
                                       protocol["executable"]), label=f"guarded {node['experiment_id']}/{node['node_index']}")
    finally:
        monitor.stop.set()
        monitor.thread.join()
    grouped = {mode: {t["task_id"]: [] for t in tasks} for mode in protocol["modes"]}
    for label, row in zip(labels, rows, strict=True):
        grouped[label[0]][label[1]].append(row["record"])
    payload = {"experiment_id": node["experiment_id"], "counters": node["counters"],
               "model_config": checkpoint["config"], "model_state_sha256": weight_hash,
               "checkpoint_sha256": node["checkpoint_sha256"], "seed_results": grouped,
               "simulator_sha256": protocol["required_fingerprints"][protocol["executable"]]}
    raw = destination / "evaluations.json.gz"
    atomic_json(raw, payload, compressed=True)
    summary = summarize_evaluation(payload, tasks, protocol["modes"])
    with gzip.open(ROOT / node["original_raw"], "rt") as stream:
        original = json.load(stream)
    changes, details = paired_changes(original, payload, tasks, protocol["modes"],
                                     protocol["allowed_changed_task_ids"])
    atomic_json(destination / "changed_rows.json.gz", details, compressed=True)
    resource = {"peak_process_tree_rss_bytes": monitor.peak_tree_rss_bytes,
                "min_system_available_bytes": monitor.min_available_bytes,
                "peak_system_swap_used_bytes": monitor.peak_swap_used_bytes}
    result = {"experiment_id": node["experiment_id"], "node_index": node["node_index"],
              "initialization_seed": node["initialization_seed"], "reward": node["reward"],
              "counters": node["counters"], "updates": node["updates"],
              "checkpoint_sha256": node["checkpoint_sha256"], "model_state_sha256": weight_hash,
              "original_training_wall_seconds": snapshot["wall_seconds"],
              "reevaluation_seconds": time.monotonic() - started, "resources": resource,
              "raw_path": str(raw.relative_to(ROOT)), "raw_sha256": sha256_file(raw),
              "changes": changes, "summary": summary,
              "jobs": len(rows), "shard_directory": str(shard_dir.relative_to(ROOT))}
    atomic_json(destination / "report.json", result)
    if profile._state_sha256(weights) != weight_hash or sha256_file(checkpoint_path) != node["checkpoint_sha256"]:
        raise ValueError("readonly reevaluation changed original weights/checkpoint")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--wait-for-matrix", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output, summary_path = args.output_dir.resolve(), args.summary.resolve()
    output.relative_to(ROOT)
    summary_path.relative_to(ROOT)
    if not args.resume and (output.exists() or summary_path.exists()):
        raise ValueError("fresh evidence paths required; use explicit --resume for the same protocol")
    protocol, queue, tasks = checked_protocol(args.protocol, args.resource_dir.resolve())
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".execution.lock").open("a") as lock, ExitStack() as original_locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = sha256_file(args.protocol)
        state = {"protocol_sha256": identity, "status": "waiting_for_matrix", "results": [],
                 "invocations": []}
        if args.resume:
            state = json.loads((output / "state.json").read_text())
            if state["protocol_sha256"] != identity or state["status"] == "complete":
                raise ValueError("resume requires the same unfinished protocol")
        state["invocations"].append({"commit": git_metadata(ROOT)[0], "started_ns": time.time_ns()})
        publish(output, summary_path, state, protocol)
        try:
            waiting_started = time.monotonic()
            last_print, dead_pid = 0., None
            while True:
                queue_state = json.loads((ROOT / protocol["queue_state"]).read_text())
                if queue_state.get("returncode", 0) != 0:
                    raise RuntimeError("reward supervisor failed; stop queued reevaluation and preserve scene")
                if queue_state.get("status") == "matrix_budget_complete":
                    if not matrix_ready(queue):
                        raise ValueError("supervisor completion disagrees with original runs")
                    break
                if not args.wait_for_matrix:
                    raise ValueError("reward matrix not complete; no workers started")
                pid = queue_state.get("pid")
                if queue_state.get("status") == "running" and pid:
                    try:
                        os.kill(pid, 0)
                        dead_pid = None
                    except ProcessLookupError:
                        if dead_pid == pid:
                            raise RuntimeError("reward process missing while queue still reports running")
                        dead_pid = pid
                if time.monotonic() - last_print >= 1800:
                    print("waiting for all12 frozen reward candidates; no Torch or workers started", flush=True)
                    last_print = time.monotonic()
                time.sleep(30)
            state["invocations"][-1]["waiting_seconds"] = time.monotonic() - waiting_started
            # Hold shared locks for the original runs: no training may resume while read-only nodes are being compared.
            for entry in queue["order"]:
                run_lock = original_locks.enter_context((ROOT / entry["output_dir"] / ".execution.lock").open("a"))
                fcntl.flock(run_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            checked_protocol(args.protocol, args.resource_dir.resolve())
            inventory = make_inventory(queue, protocol, tasks)
            inventory_path = output / "inventory.json"
            if inventory_path.exists():
                if json.loads(inventory_path.read_text()) != inventory:
                    raise ValueError("original inventory changed during interruption")
            else:
                atomic_json(inventory_path, inventory)
            state["inventory_sha256"] = sha256_file(inventory_path)
            state["status"] = "reevaluating"
            publish(output, summary_path, state, protocol)
            for i, node in enumerate(inventory["nodes"]):
                if i < len(state["results"]):
                    prior = state["results"][i]
                    if (prior["experiment_id"] != node["experiment_id"] or prior["node_index"] != node["node_index"]
                            or sha256_file(ROOT / prior["raw_path"]) != prior["raw_sha256"]
                            or prior["changes"]["unexpected_changed_rows"]):
                        raise ValueError("completed reevaluation evidence invalid; not skipping it")
                    continue
                result = evaluate_node(node, protocol, tasks, args.resource_dir.resolve(), output)
                state["results"].append(result)
                publish(output, summary_path, state, protocol)
                if result["changes"]["unexpected_changed_rows"]:
                    raise ValueError("non-conveyor rows changed; stop before capability acceptance")
                print(f"completed guarded node {i+1}/60 changes={result['changes']}", flush=True)
            for path, expected in inventory["original_states"].items():
                if sha256_file(ROOT / path) != expected:
                    raise ValueError("original training state changed during readonly reevaluation")
            state["status"] = "complete"
            publish(output, summary_path, state, protocol)
            atomic_json(output / "report.json", state)
        except BaseException:
            state["status"] = "failed"
            failure = {"error": traceback.format_exc(), "recorded_ns": time.time_ns(),
                       "completed_nodes": len(state["results"]), "protocol_sha256": identity}
            atomic_json(output / f"failure_{time.time_ns()}.json", failure)
            publish(output, summary_path, state, protocol)
            raise


if __name__ == "__main__":
    main()
