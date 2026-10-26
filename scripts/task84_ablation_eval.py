#!/usr/bin/env python3
"""task84: evaluate one ablation variant on the heldout seeds and compare with r11.

Plays the variant student on the 192 day_l7_eval_d7p_holdout1 seeds of the round-13
gate block (deploy4 engine, value v5, arms=fast, dagger-beta=0 - the gate contract),
then pairs the outcomes with the incumbent r11 games from the same block (deterministic,
so reuse is exact). Statistics only.

Usage: python3 scripts/task84_ablation_eval.py {a|b} [--workers 2]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

R13_EVAL = Path("/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000013/evaluations/a41fdfa96075ae608a40")
TASK = "day_l7_eval_d7p_holdout1"
MODELS = {"a": ROOT / "artifacts/task84/variant_a/student/model.pt",
          "b": ROOT / "artifacts/task84/variant_b/student/model.pt",
          "c": ROOT / "artifacts/task84/variant_c/student/model.pt",
          "d2": ROOT / "artifacts/task84/variant_d2/student/model.pt",
          "d3": ROOT / "artifacts/task84/variant_d3/student/model.pt",
          "r13s2": ROOT / "artifacts/task84/variant_r13s2/student/model.pt",
          "r13s3": ROOT / "artifacts/task84/variant_r13s3/student/model.pt"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("a", "b", "c", "d2", "d3", "r13s2", "r13s3"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--control", action="store_true", help="standard-deck day_l7 control task instead")
    args = parser.parse_args()
    variant = args.variant
    model = MODELS[variant]
    if not model.is_file():
        raise SystemExit("variant model missing: %s" % model)
    manifest = ROOT / ("artifacts/task84/control_eval_manifest.json" if args.control
                       else "artifacts/task84/heldout_eval_manifest.json")
    document = json.loads(manifest.read_text())
    task = document["tasks"][0]
    task_id = task["task_id"]
    out = ROOT / ("artifacts/task84/variant_%s/eval_%s" % (variant, task_id))
    out.mkdir(parents=True, exist_ok=True)
    games = out / "games.jsonl"
    complete = games.exists() and len(games.read_text().splitlines()) == len(task["seeds"])
    if not complete:
        started = time.time()
        command = ["/home/newbiexvwu/.venvs/ml/bin/python3", str(ROOT / "scripts/search_play.py"),
                   "--rollout-policy", str(model),
                   "--value", "/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt",
                   "--manifest", str(manifest), "--task-id", task_id,
                   "--executable", "/home/newbiexvwu/PvZ-search/.claude/worktrees/cq/build4/pvz-portable",
                   "--resource-dir", "/home/newbiexvwu/.cache/pvz-research-resources",
                   "--seeds", str(len(task["seeds"])), "--workers", str(args.workers),
                   "--arms", "fast", "--dagger-beta", "0", "--out", str(out)]
        import os
        with (out / "run.log").open("a") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PVZ_NO_POPUP": "1", "CUDA_VISIBLE_DEVICES": "",
                                         "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
        if result.returncode:
            raise SystemExit("search_play exited %d; see %s" % (result.returncode, out / "run.log"))
        print("played %d games in %.0f s" % (len(task["seeds"]), time.time() - started), flush=True)
    candidate = {row["seed"]: row for row in map(json.loads, games.read_text().splitlines())}
    incumbent = {row["seed"]: row for row in
                 map(json.loads, (R13_EVAL / "incumbent" / task_id / "games.jsonl").read_text().splitlines())}
    seeds = sorted(incumbent)
    flips = [s for s in seeds if incumbent[s]["won"] and not candidate[s]["won"]]
    gains = [s for s in seeds if not incumbent[s]["won"] and candidate[s]["won"]]
    summary = dict(variant=variant, task=task_id, model=str(model),
                   variant_wins=sum(1 for s in seeds if candidate[s]["won"]),
                   incumbent_wins=sum(1 for s in seeds if incumbent[s]["won"]),
                   flips=len(flips), gains=len(gains), flip_seeds=flips, gain_seeds=gains)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("variant", "variant_wins", "incumbent_wins", "flips", "gains")}))


if __name__ == "__main__":
    main()
