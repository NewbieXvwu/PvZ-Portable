#!/usr/bin/env python3
"""Compare a real uninterrupted candidate with SIGKILL and full resumption.

This is an engineering probe with separately frozen configs. All subprocess
output, interrupted shards and checkpoints remain in the fresh evidence tree.
It cannot grant a learning pass or replace the running reward matrix.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

import numpy as np
import torch

from pvz_common import sha256_file
from pvz_research import load_config
from pvz_seed_jobs import atomic_json, read_numpy


def compare(left, right, label, differences):
    """Require exact tensors, RNG, losses and raw outcomes, not just weights."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        same = left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        if not same:
            detail = {"path": label, "kind": "tensor"}
            if left.shape == right.shape and left.is_floating_point() and right.is_floating_point():
                detail["max_abs_error"] = float((left - right).abs().max()) if left.numel() else 0.0
            differences.append(detail)
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype or not np.array_equal(left, right):
            differences.append({"path": label, "kind": "array"})
    elif isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            differences.append({"path": label, "kind": "keys"})
        for key in left.keys() & right.keys():
            compare(left[key], right[key], f"{label}.{key}", differences)
    elif isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            differences.append({"path": label, "kind": "length"})
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{label}[{index}]", differences)
    elif type(left) is not type(right) or left != right:
        differences.append({"path": label, "kind": "scalar", "left": str(left)[:160], "right": str(right)[:160]})


def gpu_free_mib():
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return int(result.stdout.splitlines()[0].strip())


def launch(config, output, resource, log_path, resume=False, minimum_free_mib=1024):
    if gpu_free_mib() < minimum_free_mib:
        raise RuntimeError(f"probe requires at least {minimum_free_mib} MiB free VRAM; no other candidate is interrupted")
    command = [sys.executable, str(ROOT / "python/train_pvz_ppo_task_family.py"),
               "--resource-dir", str(resource), "--experiment-config", str(config),
               "--output-dir", str(output)]
    if resume:
        command.append("--resume")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "python"), str(ROOT / "scripts")])
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("xb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    return process, command


def kill_group(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    return process.wait(timeout=30)


def wait_success(process, timeout):
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_group(process)
        raise RuntimeError("engineering probe exceeded frozen invocation timeout; evidence retained")
    if code != 0:
        raise RuntimeError(f"engineering subprocess failed with exit code {code}")


def checkpoint(output):
    pointer = json.loads((output / "resume.json").read_text())
    path = output / pointer["checkpoint"]
    if sha256_file(path) != pointer["sha256"]:
        raise ValueError("resume pointer hash mismatch")
    return pointer, torch.load(path, map_location="cpu", weights_only=False)


def read_protocol(path, configs, loaded):
    protocol = json.loads(path.read_text())
    if sha256_file(Path(__file__)) != protocol["helper_sha256"]:
        raise ValueError("interruption helper changed after preregistration")
    for name, expected in protocol["required_fingerprints"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"preregistered interruption source/config changed: {name}")
    if [str(p.relative_to(ROOT)) for p in configs] != protocol["configs"]:
        raise ValueError("configs differ from the preregistered pair")
    if any(c["model"] != protocol["model"] for c in loaded):
        raise ValueError("actual model differs from preregistered reference")
    return protocol


def wait_for_idle(protocol, output, enabled):
    started, last_print = time.monotonic(), 0.
    while True:
        states = {name: json.loads(Path(name).read_text()).get("status")
                  if Path(name).exists() else "missing" for name in protocol["idle_states"]}
        if all(states[name] == expected for name, expected in protocol["idle_states"].items()):
            return time.monotonic() - started
        failed = any(status == "failed" for status in states.values())
        for name, status in states.items():
            if status == "process_finished" and json.loads(Path(name).read_text()).get("returncode", 0) != 0:
                failed = True
        if failed:
            raise RuntimeError("preceding execution stopped; reference probe not started")
        if not enabled:
            raise RuntimeError("reference probe requires preceding matrix and reevaluation to finish")
        if time.monotonic() - last_print >= 1800:
            atomic_json(output / "waiting.json", {"states": states, "workers_started": False})
            print(f"waiting for reference-probe idle prerequisites: {states}", flush=True)
            last_print = time.monotonic()
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-config", type=Path, required=True)
    parser.add_argument("--interrupted-config", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--wait-for-idle", action="store_true")
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path("/home/newbiexvwu/.venvs/ml/bin/python").resolve():
        raise RuntimeError("use the project ML virtualenv")
    configs = [args.continuous_config.resolve(), args.interrupted_config.resolve()]
    loaded = [load_config(path)[0] for path in configs]
    common = [{k: v for k, v in config.items() if k != "experiment_id"} for config in loaded]
    if common[0] != common[1] or loaded[0]["experiment_id"] == loaded[1]["experiment_id"]:
        raise ValueError("paired engineering configs must differ only in independent output identity")
    protocol = read_protocol(args.protocol, configs, loaded) if args.protocol else None
    if args.wait_for_idle and protocol is None:
        raise ValueError("waiting requires an explicit preregistered protocol")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    execution_lock = (output / ".execution.lock").open("a")
    fcntl.flock(execution_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    continuous, interrupted = output / "continuous", output / "interrupted"
    timeout = protocol["invocation_timeout_seconds"] if protocol else 600
    minimum_free_mib = protocol["minimum_free_vram_mib"] if protocol else 1024
    started = time.monotonic()
    process = None
    report = {"schema_version": 1, "scope": "actual explicitly configured SIGKILL/full-resume engineering probe; no learning pass",
              "configs": [{"path": str(path), "sha256": sha256_file(path)} for path in configs],
              "model_config": loaded[0]["model"], "invocation_timeout_seconds": timeout,
              "exact_equality_required": True, "coexecution": "original reward matrix remains running; elapsed time is not dedicated-machine performance"}
    try:
        if protocol:
            report["protocol_sha256"] = sha256_file(args.protocol)
            report["waiting_seconds"] = wait_for_idle(protocol, output, args.wait_for_idle)
            read_protocol(args.protocol, configs, loaded)
            started = time.monotonic()
            available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                                 if line.startswith("MemAvailable:"))) * 1024
            if available < protocol["minimum_available_ram_bytes"]:
                raise RuntimeError("reference probe RAM precondition unmet; no other candidate is stopped")
            report["coexecution"] = "preceding frozen reward matrix and corrected reevaluation completed before reference probe"

        def remaining_timeout():
            remaining = (protocol["total_timeout_seconds"] - (time.monotonic() - started)) if protocol else timeout
            if remaining <= 0:
                raise RuntimeError("reference probe exceeded total preregistered timeout")
            return min(timeout, remaining)

        print("phase=continuous", flush=True)
        process, command = launch(configs[0], continuous, args.resource_dir.resolve(), output / "continuous.log",
                                  minimum_free_mib=minimum_free_mib)
        report["continuous_command"] = command
        wait_success(process, remaining_timeout())
        print("phase=interrupt_at_update2_partial_collection", flush=True)
        process, command = launch(configs[1], interrupted, args.resource_dir.resolve(), output / "interrupted.log",
                                  minimum_free_mib=minimum_free_mib)
        report["interrupted_command"] = command
        deadline = time.monotonic() + remaining_timeout()
        while True:
            if process.poll() is not None:
                raise RuntimeError("interrupted arm exited before the frozen partial-collection trigger")
            if time.monotonic() >= deadline:
                raise RuntimeError("partial-collection trigger timed out")
            state_path = interrupted / "training_state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            shards = sorted((interrupted / "runs/run_1/.seed_jobs/update_000002").glob("*/seed_*.npz"))
            if state.get("updates") == 1 and len(shards) >= 4:
                assigned_count = len(read_numpy(shards[0])["metadata"]["assignments"])
                if len(shards) >= assigned_count:
                    raise RuntimeError("second assigned collection already complete; no partial trigger")
                prior_pointer = json.loads((interrupted / "resume.json").read_text())
                code = kill_group(process)
                pointer, saved = checkpoint(interrupted)
                if code != -signal.SIGKILL or pointer != prior_pointer or saved["training_state"]["updates"] != 1:
                    raise RuntimeError("SIGKILL did not retain exactly the update-1 checkpoint")
                shards = sorted((interrupted / "runs/run_1/.seed_jobs/update_000002").glob("*/seed_*.npz"))
                retained = [{"path": str(path.relative_to(interrupted)), "sha256": sha256_file(path),
                             "mtime_ns": path.stat().st_mtime_ns} for path in shards]
                if len(retained) < 4 or len(retained) >= assigned_count:
                    raise RuntimeError("SIGKILL must occur during an incomplete second collection")
                report["intentional_interruption"] = {"exit_code": code, "last_complete_updates": 1,
                    "counters": saved["training_state"]["counters"], "resume_pointer": pointer,
                    "completed_partial_shards": retained,
                    "assigned_second_batch_size": assigned_count,
                    "optimizer_states": len(saved["optimizer_state_dict"]["state"]),
                    "rng_components": sorted(saved["rng_state"])}
                atomic_json(output / "intentional_interruption.json", report["intentional_interruption"])
                break
            time.sleep(0.1)
        print(f"phase=resume cached_partial_shards={len(retained)}", flush=True)
        process, command = launch(configs[1], interrupted, args.resource_dir.resolve(), output / "resumed.log", resume=True,
                                  minimum_free_mib=minimum_free_mib)
        report["resume_command"] = command
        wait_success(process, remaining_timeout())
        differences = []
        left_pointer, left = checkpoint(continuous)
        right_pointer, right = checkpoint(interrupted)
        for key in ("state_dict", "optimizer_state_dict", "rng_state"):
            compare(left[key], right[key], key, differences)
        state_keys = ("updates", "counters", "phase", "status", "evaluation_cursor", "recent_passes", "initial_state_sha256")
        if loaded[0]["sampling"]["method"] in ("terrain_balanced", "learning_progress"):
            state_keys += ("curriculum_state",)
        for key in state_keys:
            compare(left["training_state"][key], right["training_state"][key], f"training_state.{key}", differences)
        for index, (a, b) in enumerate(zip(left["training_state"]["update_history"], right["training_state"]["update_history"], strict=True)):
            for key in ("update", "counters", "losses", "trajectory_stats", "episode_digests"):
                compare(a[key], b[key], f"update_history[{index}].{key}", differences)
            if loaded[0]["sampling"]["method"] in ("terrain_balanced", "learning_progress"):
                compare(a["curriculum_sampling"], b["curriculum_sampling"],
                        f"update_history[{index}].curriculum_sampling", differences)
        for index, (a, b) in enumerate(zip(left["training_state"]["learning_curve"], right["training_state"]["learning_curve"], strict=True)):
            for key in ("updates", "counters", "summary"):
                compare(a[key], b[key], f"learning_curve[{index}].{key}", differences)
            with gzip.open(continuous / a["raw_seed_results_path"], "rt") as stream:
                a_raw = json.load(stream)
            with gzip.open(interrupted / b["raw_seed_results_path"], "rt") as stream:
                b_raw = json.load(stream)
            compare(a_raw["seed_results"], b_raw["seed_results"], f"raw_evaluation[{index}]", differences)
        preservation = []
        for shard in retained:
            path = interrupted / shard["path"]
            preservation.append({"path": shard["path"], "same_sha256": sha256_file(path) == shard["sha256"],
                                 "same_mtime_ns": path.stat().st_mtime_ns == shard["mtime_ns"]})
        resume_lines = [line for line in (output / "resumed.log").read_text().splitlines()
                        if "update 2 cached=" in line]
        if not resume_lines or not all(row["same_sha256"] and row["same_mtime_ns"] for row in preservation):
            differences.append({"path": "partial_shard_reuse", "kind": "preservation_or_log"})
        if left["training_state"]["status"] != "budget_complete" or right["training_state"]["status"] != "budget_complete":
            differences.append({"path": "final_status", "kind": "budget_incomplete"})
        remaining_timeout()
        report.update({"gate_result": "pass" if not differences else "fail", "differences": differences,
                       "final_counters": left["training_state"]["counters"], "final_updates": left["training_state"]["updates"],
                       "evaluated_nodes": len(left["training_state"]["learning_curve"]),
                       "partial_shard_preservation": preservation, "resume_log_lines": resume_lines,
                       "final_resume_pointers": [left_pointer, right_pointer],
                       "resources": [left["training_state"].get("resources"), right["training_state"].get("resources")],
                       "continuous_wall_seconds": left["training_state"]["wall_seconds"],
                       "interrupted_recorded_wall_seconds": right["training_state"]["wall_seconds"],
                       "seconds": time.monotonic() - started})
        atomic_json(output / "report.json", report)
        print(json.dumps({k: report[k] for k in ("gate_result", "final_counters", "final_updates", "seconds")}), flush=True)
        if differences:
            raise RuntimeError(f"exact interruption/resume comparison failed in {len(differences)} places")
    except BaseException as error:
        if process is not None and process.poll() is None:
            kill_group(process)
        report.update({"gate_result": "fail", "error_type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc(), "seconds": time.monotonic() - started})
        atomic_json(output / f"failure_{time.time_ns()}.json", report)
        raise


if __name__ == "__main__":
    main()
