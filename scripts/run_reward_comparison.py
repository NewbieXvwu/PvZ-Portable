"""Run the frozen twelve-candidate reward matrix, one candidate at a time."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_seed_jobs import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--resource-dir", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    if len(queue["order"]) != 12 or queue["required_initializations"] != [0, 1, 2]:
        raise ValueError("expected the preregistered four-reward, three-initialization matrix")
    matrix = queue["matrix"]
    if not matrix.replace("_", "").isalnum():
        raise ValueError("matrix must be a simple identifier")
    status_path = ROOT / "artifacts/research" / f"{matrix}_queue_state.json"
    for index, item in enumerate(queue["order"]):
        config_path = ROOT / item["config"]
        config = json.loads(config_path.read_text())
        output = ROOT / item["output_dir"]
        state_path = output / "training_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if (state.get("status") == "budget_complete"
                    and state["counters"]["decisions"] >= queue["common_decision_budget"]
                    and state["learning_curve"][-1]["updates"] == state["updates"]):
                print(f"already completed {config['experiment_id']}; preserving results", flush=True)
                continue
        command = [sys.executable, str(ROOT / "python/train_pvz_ppo_task_family.py"),
                   "--resource-dir", str(args.resource_dir.resolve()),
                   "--experiment-config", str(config_path), "--output-dir", str(output)]
        if (output / "resume.json").exists():
            command.append("--resume")
        log_path = ROOT / item["log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log_path.open("a", buffering=1) as log:
            log.write(f"\n{datetime.now(timezone.utc).isoformat()} command={command!r}\n")
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            atomic_json(status_path, {"candidate_index": index, "experiment_id": config["experiment_id"],
                                      "pid": process.pid, "status": "running", "log": str(log_path),
                                      "config": str(config_path)})
            print(f"running {index + 1}/12 {config['experiment_id']} pid={process.pid}", flush=True)
            last_heartbeat = time.monotonic()
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() - last_heartbeat >= 1800:
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    latest = (state.get("update_history") or [{}])[-1]
                    losses = latest.get("losses", {})
                    message = (f"heartbeat elapsed={time.monotonic() - started:.1f}s "
                               f"counters={state.get('counters')} "
                               f"policy_loss={losses.get('policy_loss')} value_loss={losses.get('value_loss')}")
                    log.write(message + "\n")
                    print(message, flush=True)
                    last_heartbeat = time.monotonic()
            atomic_json(status_path, {"candidate_index": index, "experiment_id": config["experiment_id"],
                                      "pid": process.pid, "returncode": process.returncode,
                                      "status": "process_finished", "seconds": time.monotonic() - started,
                                      "log": str(log_path)})
            if process.returncode:
                raise SystemExit(f"candidate failed with exit {process.returncode}; scene retained: {output}")
        state = json.loads(state_path.read_text())
        if state["status"] != "budget_complete" or state["counters"]["decisions"] < queue["common_decision_budget"]:
            raise SystemExit(f"candidate stopped before frozen budget: {output}")
    atomic_json(status_path, {"status": "matrix_budget_complete", "candidate_count": 12,
                              "note": "budgets completed; learning and full-level gates require separate review"})


if __name__ == "__main__":
    main()
