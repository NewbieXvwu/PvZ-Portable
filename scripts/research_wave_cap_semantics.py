"""Gate dynamic capped-wave behaviour against the corresponding full-level prefix.

Fixed WAIT60 controls use the original five cap1 tasks and seeds. Compare battle
zombies/plants/projectiles/sun until the full run starts wave N+1 or the capped
run terminates. Goal counters/timers and final UI are not physical-state equality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_env import PvZEnv, TaskSpec
from pvz_seed_jobs import atomic_json
from research_aid_feasibility import choose


def battle(observation: dict) -> dict:
    return {"zombies": [z for z in observation["zombies"] if z["on_board"]],
            "plants": observation["plants"], "projectiles": observation["projectiles"],
            "sun": observation["sun"], "coins": observation["coins"],
            "grid_items": observation["grid_items"], "defenses": observation["defenses"],
            "packets": observation["packets"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--seeds-per-task", type=int, default=4)
    parser.add_argument("--caps", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--max-ticks", type=int, default=20000)
    parser.add_argument("--control", choices=["wait", "scripted"], default="wait")
    parser.add_argument("--aid", choices=["none", "full_defence"], default="none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.seeds_per_task <= 64 or args.max_ticks < 2400:
        raise ValueError("invalid bounded audit scope")
    tasks = json.loads((ROOT / "artifacts/task_family/train.json").read_text())["tasks"]
    tasks = [t for t in tasks if t["wave_cap"] == 1 and t["zombie_count_multiplier"] == 1]
    aid_manifest = ROOT / "experiments/t5/aid_feasibility_v1.json"
    aids = {t["terrain"]: t["preplanted"] for t in json.loads(aid_manifest.read_text())["tasks"]
            if t["aid"] == args.aid and t["wave_cap"] == 5}
    started, records, comparisons = time.monotonic(), [], 0
    with PvZEnv(resource_dir=args.resource_dir, executable=args.executable.resolve()) as full, \
         PvZEnv(resource_dir=args.resource_dir, executable=args.executable.resolve()) as capped:
        for task in tasks:
            for cap in args.caps:
                for seed in task["seeds"][:args.seeds_per_task]:
                    kwargs = {"level": task["level"], "seed": seed, "playthrough": task["playthrough"],
                              "zombie_count_multiplier": task["zombie_count_multiplier"],
                              "preplanted": tuple(tuple(p) for p in aids[task["terrain"]])}
                    f, _ = full.reset(deck=task["deck"], task=TaskSpec(**kwargs))
                    c, _ = capped.reset(deck=task["deck"], task=TaskSpec(**kwargs, wave_cap=cap))
                    cap_table = capped._command("PRIV")["observation"]["hidden"]["zombies_in_wave"]
                    full_table = full._command("PRIV")["observation"]["hidden"]["zombies_in_wave"]
                    table_match = cap_table == full_table[:cap]
                    first_mismatch = ({"tick": 0, "full": battle(f), "capped": battle(c)}
                                      if battle(f) != battle(c) else None)
                    first_count_mismatch, compared = None, 0
                    evidence = []
                    while f["tick"] < args.max_ticks and not f["terminal"] and not c["terminal"]:
                        action = {"type": "wait", "ticks": 60} if args.control == "wait" else choose(f)
                        f, _, _, _, _ = full.step(action)
                        c, _, _, _, _ = capped.step(action)
                        if f["wave"] > cap:
                            break  # the full run is now intentionally a different task
                        compared += 1
                        comparisons += 1
                        fb, cb = battle(f), battle(c)
                        # Terminal animations/result are intentionally distinct.
                        if not c["terminal"] and fb != cb and first_mismatch is None:
                            first_mismatch = {"tick": f["tick"], "full": fb, "capped": cb}
                        if len(fb["zombies"]) != len(cb["zombies"]) and first_count_mismatch is None:
                            first_count_mismatch = {"tick": f["tick"], "full": fb["zombies"], "capped": cb["zombies"]}
                        if first_mismatch and len(evidence) < 6:
                            evidence.append({"tick": f["tick"], "full_wave": f["wave"], "cap_wave": c["wave"],
                                             "full_count": len(fb["zombies"]), "cap_count": len(cb["zombies"])})
                    records.append({"task_id": task["task_id"], "terrain": task["terrain"], "seed": seed,
                                    "cap": cap, "full_wave_count": f["wave_count"],
                                    "wave_table_matches_prefix": table_match, "compared_responses": compared,
                                    "first_physical_mismatch": first_mismatch,
                                    "first_count_mismatch": first_count_mismatch, "mismatch_followup": evidence,
                                    "end_tick": c["tick"], "cap_result": c["result"], "full_result": f["result"],
                                    "full_wave_at_end": f["wave"], "cap_wave_at_end": c["wave"],
                                    "stop_reason": ("cap_terminal" if c["terminal"] else
                                                    "full_terminal" if f["terminal"] else
                                                    "next_full_wave" if f["wave"] > cap else "tick_horizon")})
            atomic_json(args.output.with_suffix(".progress.json"), {"task": task["task_id"],
                        "records": len(records), "comparisons": comparisons})
            print(task["task_id"], "cases", len(records), flush=True)
    failed = [r for r in records if not r["wave_table_matches_prefix"] or r["first_physical_mismatch"]]
    report = {"gate_result": "fail" if failed else "pass", "scope": "original five terrains; caps1/3/5; actual dynamic battle prefixes",
              "seed_count_per_task": args.seeds_per_task, "control": args.control, "aid": args.aid,
              "max_ticks": args.max_ticks,
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "aid_manifest_sha256": hashlib.sha256(aid_manifest.read_bytes()).hexdigest(), "cases": len(records), "failed_cases": len(failed),
              "simulator_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
              "compared_responses": comparisons, "records": records, "seconds": time.monotonic() - started,
              "limits": "identical controls from full public observation; stop at full waveN+1, cap terminal, real loss, or bounded tick horizon; not an RL learning gate"}
    atomic_json(args.output, report)
    print(report["gate_result"], len(records), "cases", len(failed), "failed", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
