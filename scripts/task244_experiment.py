"""Offline task244 matched games through the existing production runner and statistics."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
from task243_teacher import paired


def tempered_model(model, target, temperature):
    import torch
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    for key in ("head.2.weight", "head.2.bias"):
        checkpoint["state_dict"][key] /= temperature
    checkpoint["task244_temperature"] = temperature
    torch.save(checkpoint, target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--margin-relative", type=float, default=0)
    parser.add_argument("--phase", choices=("l22", "panel"), default="l22")
    parser.add_argument("--seed-start", type=int, default=244100000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task244")
    args = parser.parse_args()
    base = args.data.resolve()
    (base / "work").mkdir(exist_ok=True)
    models = base / "models"
    env = dict(os.environ, PVZ_NO_POPUP="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", TMPDIR=str(base / "work"))
    model = args.model.resolve()
    if args.temperature != 1:
        model = tempered_model(model, models / f"{args.label}.pt", args.temperature)
    panel = json.loads((base / "panel-original.json").read_text())
    if args.phase == "l22":
        panel["tasks"] = [t for t in panel["tasks"] if t["level"] == 22]
    else:
        for i, task in enumerate(panel["tasks"]):
            task["seeds"] = [args.seed_start + i * 1000 + k for k in range(32)]
    panel_path = base / f"panel-{args.phase}.json"
    panel_path.write_text(json.dumps(panel, indent=1))
    out = base / f"{args.phase}-{args.label}"
    out.mkdir(exist_ok=True)
    rows = []
    for task in panel["tasks"]:
        target = out / f"l{task['level']}"
        command = [sys.executable, str(ROOT / "scripts/search_play.py"),
            "--rollout-policy", str(models / "student.pt"), "--rollout-policy-gpt", str(models / "gptbc.pt"),
            "--value", str(model), "--value-roof", str(model if args.label != "production" else models / "roof.pt"),
            "--static-value", str(models / "static.pt"), "--manifest", str(panel_path),
            "--task-id", task["task_id"], "--seeds", "32", "--arms", "fast_search", "--workers", str(args.workers),
            "--verify-samples", "9", "--student-candidates", "2", "--native-games",
            "--margin-relative", str(args.margin_relative),
            "--executable", str(Path.home() / "pvz-mac-collect/build-vn6/pvz-portable"), "--out", str(target)]
        with (out / f"l{task['level']}.log").open("a") as log:
            subprocess.run(command, env=env, stdout=log, stderr=log, check=True)
        games = [json.loads(line) for line in (target / "games.jsonl").read_text().splitlines()]
        if len(games) != 32 or not all(g["terminal"] for g in games):
            raise RuntimeError(f"L{task['level']}: incomplete terminal pairs")
        rows.extend(dict(level=task["level"], seed=g["seed"], won=g["won"], terminal=g["terminal"],
                         overrides=g["overrides"], searched=g["searched"], planted=g.get("planted_types")) for g in games)
        print(args.label, task["level"], sum(g["won"] for g in games), "/", len(games), flush=True)
    (out / "outcomes.json").write_text(json.dumps(rows, indent=1) + "\n")
    if args.phase == "l22":
        old = json.loads((base / 'task243-pairs.json').read_text())
        reference = {(r['level'], r['seed']): r for r in old}
        pairs = [dict(candidate=r["won"], production=reference[r['level'], r['seed']]['production']) for r in rows]
        report = paired(pairs, "candidate", "production")
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report), flush=True)
    elif args.label != "production":
        control = json.loads((base / "panel-production/outcomes.json").read_text())
        reference = {(r["level"], r["seed"]): r["won"] for r in control}
        pairs = [dict(level=r["level"], seed=r["seed"], candidate=r["won"],
                      production=reference[r["level"], r["seed"]]) for r in rows]
        report = dict(per_level={str(t["level"]): paired(
            [r for r in pairs if r["level"] == t["level"]], "candidate", "production")
            for t in panel["tasks"]}, pooled=paired(pairs, "candidate", "production"))
        report["eligible"] = report["pooled"]["ci_noninferior_margin_2_of_32"] and all(
            r["point_noninferior_margin_2_of_32"] and r["mcnemar_worse_p"] >= .05
            for r in report["per_level"].values())
        (out / "pairs.json").write_text(json.dumps(pairs, indent=1) + "\n")
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
