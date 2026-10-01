"""Paired native observations, allowing only a defined empty-slot cost correction."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
from pvz_env import PvZEnv, TaskSpec
from pvz_seed_jobs import atomic_json
import scripted_baseline


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def normalized(observation: dict) -> dict:
    corrected = copy.deepcopy(observation)
    for packet in corrected["packets"]:
        if packet["type"] == -1:
            packet["cost"] = 0
    return corrected


def task_spec(task: dict, seed: int) -> TaskSpec:
    return TaskSpec(level=task["level"], seed=seed, playthrough=task["playthrough"],
                    zombie_count_multiplier=task["zombie_count_multiplier"], wave_cap=task["wave_cap"],
                    preplanted=tuple(tuple(x) for x in task["preplanted"]))


def action_for(observation: dict, control: str, rng: random.Random) -> dict:
    if control == "wait":
        return {"type": "wait", "ticks": 150}
    if control == "scripted":
        return scripted_baseline.choose(observation)
    if control == "random":
        legal = observation["legal_actions"]
        actions = [{"type": "plant", **x} for x in legal["plants"]]
        actions += [{"type": "shovel", "col": col, "row": row} for col, row in legal["shovels"]]
        actions += [{"type": "wait", "ticks": ticks} for ticks in [60, 150, 300]]
        return rng.choice(actions)
    raise ValueError(control)


def paired(job: dict) -> dict:
    task, seed = job["task"], job["seed"]
    rng = random.Random(seed)
    rows, failure = [], None
    empty_states = 0
    with PvZEnv(job["resource_dir"], executable=job["old_binary"]) as old, PvZEnv(
            job["resource_dir"], executable=job["new_binary"]) as new:
        left, _ = old.reset(deck=task["deck"], task=task_spec(task, seed))
        right, _ = new.reset(deck=task["deck"], task=task_spec(task, seed))
        for index in range(job["max_actions"] + 1):
            expected = normalized(left)
            equal = expected == right
            empty = [p for p in right["packets"] if p["type"] == -1]
            empty_states += bool(empty)
            rows.append({"decision": index, "tick": right["tick"],
                         "old_sha256": digest(left), "normalized_old_sha256": digest(expected),
                         "new_sha256": digest(right), "empty_packets": len(empty)})
            if not equal or any(p["cost"] != 0 for p in empty):
                failure = {"decision": index, "old": left, "new": right}
                break
            if right["terminal"] or index == job["max_actions"]:
                break
            action = action_for(right, job["control"], rng)
            rows[-1]["action"] = action
            left, _, _, _, old_info = old.step(action)
            right, _, _, _, new_info = new.step(action)
            if old_info != new_info or not new_info["ok"]:
                failure = {"decision": index, "action": action, "old_info": old_info, "new_info": new_info}
                break
    result = {"job": job["job"], "task_id": task["task_id"], "seed": seed,
              "control": job["control"], "observations": len(rows), "empty_states": empty_states,
              "outcome": {k: right[k] for k in ["terminal", "result", "wave", "wave_count", "tick"]},
              "failure": failure}
    path = Path(job["output_dir"]) / f"paired_{job['job']:04d}.json.gz"
    atomic_json(path, {"result": result, "steps": rows}, compressed=True)
    return {**result, "raw_path": str(path), "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def replay_recorded(path: Path, task: dict, protocol: dict, resource_dir: Path, destination: Path) -> dict:
    stored = json.loads(gzip.decompress(path.read_bytes()))
    rows, failure, empty_states = [], None, 0
    with PvZEnv(resource_dir, executable=protocol["new_binary"]) as env:
        observation, _ = env.reset(deck=task["deck"], task=task_spec(task, protocol["trace_seed"]))
        for index, step in enumerate(stored["steps"]):
            empty_states += any(p["type"] == -1 for p in observation["packets"])
            rows.append({"decision": index, "tick": observation["tick"],
                         "normalized_recorded_sha256": digest(normalized(step["observation"])),
                         "new_sha256": digest(observation), "action": step["action"]})
            if normalized(step["observation"]) != observation:
                failure = {"decision": index, "recorded": step["observation"], "new": observation}
                break
            observation, _, _, _, info = env.step(step["action"])
            if not info["ok"]:
                failure = {"decision": index, "action": step["action"], "info": info}
                break
        expected = stored["outcome"]
        actual = {"won": observation["result"] == 1, "result": observation["result"],
                  "terminated": observation["terminal"], "terminal_tick": observation["tick"],
                  "terminal_wave": observation["wave"], "wave_count": observation["wave_count"]}
        if any(actual[k] != expected[k] for k in actual):
            failure = failure or {"outcome_difference": {"expected": expected, "actual": actual}}
    result = {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "decisions": len(rows), "empty_states": empty_states, "failure": failure, "outcome": actual}
    atomic_json(destination, {"result": result, "steps": rows}, compressed=True)
    return {**result, "raw_path": str(destination), "raw_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.summary.exists():
        raise ValueError("fresh outputs required")
    protocol = json.loads(args.protocol.read_text())
    for relative, sha in protocol["required_fingerprints"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != sha:
            raise ValueError(f"preregistered file changed: {relative}")
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    jobs = []
    for task in protocol["tasks"]:
        for seed in task["seeds"]:
            for control in protocol["controls"]:
                jobs.append({"job": len(jobs), "task": task, "seed": seed, "control": control,
                             "old_binary": protocol["old_binary"], "new_binary": protocol["new_binary"],
                             "resource_dir": str(args.resource_dir), "max_actions": protocol["max_actions"],
                             "output_dir": str(args.output_dir.resolve())})
    records = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(protocol["workers"]) as pool:
        iterator = pool.imap_unordered(paired, jobs, chunksize=1)
        for i in range(len(jobs)):
            records.append(iterator.next(timeout=900))
            if (i + 1) % 64 == 0 or i + 1 == len(jobs):
                atomic_json(args.output_dir / "partial.json", {"completed": i + 1, "total": len(jobs), "records": records})
                print(f"native paired {i + 1}/{len(jobs)}", flush=True)
    traces = []
    trace_root = ROOT / protocol["trace_root"]
    for i, path in enumerate(sorted(trace_root.glob("fusion*_workers*/job_*.json.gz"))):
        traces.append(replay_recorded(path, protocol["trace_task"], protocol, args.resource_dir,
                                      args.output_dir / f"recorded_{i:03d}.json.gz"))
    if len(traces) != protocol["trace_count"]:
        raise ValueError("recorded trace count differs from preregistration")
    report = {"schema_version": 1, "protocol_sha256": hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              "native_hashes": {k: hashlib.sha256(Path(protocol[k]).read_bytes()).hexdigest()
                                for k in ["old_binary", "new_binary"]},
              "paired_jobs": len(records), "paired_observations": sum(x["observations"] for x in records),
              "paired_empty_states": sum(x["empty_states"] for x in records),
              "paired_failures": sum(x["failure"] is not None for x in records),
              "recorded_traces": len(traces), "recorded_failures": sum(x["failure"] is not None for x in traces),
              "records": sorted(records, key=lambda x: x["job"]), "recorded_replays": traces,
              "seconds": time.monotonic() - started, "coexecution": protocol["coexecution"],
              "intentional_exception": "Only SEED_NONE cost is defined as0; all other public fields and action responses exact under identical actions.",
              "scope": "Native semantics and readonly replay; not learning success, not yet fixed-policy repeatability."}
    report["result"] = "pass" if not report["paired_failures"] and not report["recorded_failures"] else "fail"
    atomic_json(args.output_dir / "report.json", report)
    compact = {k: v for k, v in report.items() if k != "records"}
    compact["full_report"] = {"path": str(args.output_dir / "report.json"),
                              "sha256": hashlib.sha256((args.output_dir / "report.json").read_bytes()).hexdigest()}
    compact["failure_jobs"] = [x for x in records if x["failure"] is not None]
    atomic_json(args.summary, compact)
    if report["result"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
