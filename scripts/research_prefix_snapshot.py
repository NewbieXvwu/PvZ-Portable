"""Verify native snapshot replay and batched branches with the capped-prefix fix."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_env import PvZEnv, TaskSpec, branch_action_token
from pvz_seed_jobs import atomic_json


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tasks = [t for t in json.loads((ROOT / "artifacts/task_family/train.json").read_text())["tasks"]
             if t["wave_cap"] == 1 and t["zombie_count_multiplier"] == 1]
    aids = {t["terrain"]: t["preplanted"] for t in json.loads(
        (ROOT / "experiments/t5/aid_feasibility_v1.json").read_text())["tasks"]
            if t["aid"] == "full_defence" and t["wave_cap"] == 5}
    started, records, failed = time.monotonic(), [], []
    with PvZEnv(args.resource_dir) as env:
        for task in tasks:
            for cap in (None, 1, 3, 5):
                for seed in task["seeds"][:4]:
                    for aid in ("none", "full_defence"):
                        for anchor in (1740, 1980, 3000):
                            obs, _ = env.reset(deck=task["deck"], task=TaskSpec(
                                level=task["level"], seed=seed, wave_cap=cap,
                                preplanted=tuple(tuple(p) for p in aids[task["terrain"]])
                                if aid != "none" else ()))
                            for _ in range(anchor // 60):
                                obs = env._command("WAIT 60")["observation"]
                                if obs["terminal"]:
                                    break
                            if obs["terminal"]:
                                raise RuntimeError("snapshot anchor unexpectedly terminal")
                            parent = env._command("SNAPSHOT")["snapshot_id"]
                            actions = [{"type": "wait", "ticks": 60}, {"type": "wait", "ticks": 120}]
                            if obs["legal_actions"]["plants"]:
                                actions.append({"type": "plant", **obs["legal_actions"]["plants"][0]})
                            if obs["legal_actions"]["shovels"]:
                                col, row = obs["legal_actions"]["shovels"][0]
                                actions.append({"type": "shovel", "col": col, "row": row})
                            tokens = [branch_action_token(action) for action in actions]
                            response = env._command(f"BRANCH_SNAPSHOT_FAST {parent} {len(tokens)} " + " ".join(tokens))
                            if not response["ok"]:
                                raise RuntimeError("native branch command rejected")
                            for action, branch in zip(actions, response["branches"], strict=True):
                                env._command(f"RESTORE {parent}")
                                if action["type"] == "wait":
                                    command = f"WAIT {action['ticks']}"
                                elif action["type"] == "plant":
                                    command = f"PLANT {action['packet']} {action['col']} {action['row']}"
                                else:
                                    command = f"SHOVEL {action['col']} {action['row']}"
                                direct = env._command(command)
                                first = {k: direct[k] for k in ("ok", "observation", "events")}
                                second = {k: branch[k] for k in ("ok", "observation", "events")}
                                env._command(f"RESTORE {parent}")
                                replay = env._command(command)
                                third = {k: replay[k] for k in ("ok", "observation", "events")}
                                record = {"task": task["task_id"], "cap": cap, "seed": seed,
                                          "aid": aid, "anchor": anchor, "action": action,
                                          "direct_sha256": digest(first), "branch_sha256": digest(second),
                                          "replay_sha256": digest(third), "match": first == second == third}
                                records.append(record)
                                if not record["match"]:
                                    failed.append({**record, "direct": first, "branch": second, "replay": third})
                                if branch.get("snapshot_id", 0):
                                    env._command(f"DROP_SNAPSHOT {branch['snapshot_id']}")
                            env._command(f"DROP_SNAPSHOT {parent}")
            print(task["task_id"], len(records), "responses", flush=True)
    report = {"gate_result": "fail" if failed else "pass", "responses": len(records),
              "records": records, "failed": failed, "seconds": time.monotonic() - started,
              "simulator_sha256": hashlib.sha256((ROOT / "build/pvz-portable").read_bytes()).hexdigest(),
              "scope": "five original terrains; full and caps1/3/5; four seeds; no/full aid; three anchors; WAIT/PLANT/SHOVEL"}
    atomic_json(args.output, report)
    print(report["gate_result"], len(records), "responses", len(failed), "failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
