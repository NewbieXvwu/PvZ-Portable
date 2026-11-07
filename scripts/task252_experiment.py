"""task252: resumable fog/roof confirmation through task250's paired-game driver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

from task250_experiment import ROOT, report, run_arm


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task252")
    parser.add_argument("--models", type=Path,
                        default=ROOT.parent / "vnpool/artifacts/task250/models")
    parser.add_argument("--fog-panel", type=Path,
                        default=ROOT.parent / "fogvalue/MEMORY/docs/task245-fog-value.json")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    base = args.data.resolve()
    base.mkdir(parents=True, exist_ok=True)
    (base / "work").mkdir(exist_ok=True)
    if not (base / "models").exists():
        (base / "models").symlink_to(args.models.resolve(), target_is_directory=True)
    panel_path = base / "panel.json"
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
    else:
        fog = json.loads(args.fog_panel.read_text())["panel"]["tasks"]
        gate = json.loads((ROOT / "configs/gate_v8_panel.json").read_text())
        initial = {41: .75, 42: .5, 43: .75, 47: .75, 48: .5}
        roof = [dict(t, zombie_count_multiplier=initial[t["level"]])
                for t in gate["tasks"] if t["level"] in initial]
        tasks = []
        for source in fog + roof:
            task = dict(source, task_id=f"task252_l{source['level']}", pairs=32)
            task["seeds"] = [252100000 + task["level"] * 1000 + k for k in range(32)]
            tasks.append(task)
        panel = dict(tasks=tasks)
        write(panel_path, panel)
    calibration_path = base / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {}
    outcomes = {"production": [], "candidate": []}

    def arm(task, label, seeds, target):
        target.mkdir(parents=True, exist_ok=True)
        cached = target / "outcomes.json"
        if cached.exists():
            return json.loads(cached.read_text())
        models = base / "models"
        model = dict(value=models / ("production.pt" if label == "production" else "finetune-t3.pt"),
                     roof=models / ("roof.pt" if label == "production" else "finetune-t3.pt"))
        settings = SimpleNamespace(workers=args.workers, margin_relative=0 if label == "production" else .02)
        print(f"L{task['level']} {label}: {len(seeds)} games, multiplier={task['zombie_count_multiplier']}",
              flush=True)
        return run_arm(sys.executable, base, {"tasks": [task]}, label, model, settings,
                       [task["level"]], {task["level"]: seeds}, {}, target)

    for task in panel["tasks"]:
        level = task["level"]
        if task["terrain"] == "roof":
            trials = calibration.setdefault(str(level), [])
            # Only production outcomes select the multiplier. Candidate games start after freeze.
            while not trials or not trials[-1].get("selected"):
                index = len(trials)
                multiplier = task["zombie_count_multiplier"] if not trials else trials[-1]["next_multiplier"]
                measured = [r for r in trials if not r.get("interrupted")]
                rates = [(r["multiplier"], r.get("wins", r["pilot_wins"]) /
                          len(r.get("seeds", r["pilot_seeds"]))) for r in measured]
                easy = [m for m, rate in rates if rate > .8]
                hard = [m for m, rate in rates if rate < .2]
                if easy and hard and max(easy) < min(hard):
                    multiplier = round((max(easy) + min(hard)) / 2, 6)
                trial = dict(task, zombie_count_multiplier=multiplier)
                seeds = [252000000 + level * 1000 + index * 100 + k for k in range(32)]
                target = base / "calibration" / f"l{level}-r{index}"
                pilot = arm(trial, "production", seeds[:8], target / "pilot")
                wins = sum(r["won"] for r in pilot)
                row = dict(multiplier=multiplier, pilot_wins=wins, pilot_seeds=seeds[:8])
                if 2 <= wins <= 6:
                    rest = arm(trial, "production", seeds[8:], target / "rest")
                    row.update(wins=wins + sum(r["won"] for r in rest), seeds=seeds)
                    row["selected"] = 7 <= row["wins"] <= 25
                    wins, total = row["wins"], 32
                else:
                    row["selected"] = False
                    total = 8
                row["next_multiplier"] = round(multiplier * (1.25 if wins / total > .8 else .8), 6)
                trials.append(row)
                write(calibration_path, calibration)
                if len(trials) >= 12 and not row["selected"]:
                    raise RuntimeError(f"L{level}: calibration did not reach 20%-80%; inspect production trials")
            task["zombie_count_multiplier"] = trials[-1]["multiplier"]
            task["seeds"] = trials[-1]["seeds"]
            write(panel_path, panel)
        for label in outcomes:
            if label == "production" and task["terrain"] == "roof":
                selected = base / "calibration" / f"l{level}-r{len(calibration[str(level)]) - 1}"
                rows = sum((json.loads((selected / part / "outcomes.json").read_text())
                            for part in ("pilot", "rest")), [])
            else:
                rows = arm(task, label, task["seeds"], base / "panel" / label / f"l{level}")
            outcomes[label].extend(rows)
        result = report(outcomes["candidate"], outcomes["production"])
        terrains = {}
        for terrain in ("fog", "roof"):
            levels = {t["level"] for t in panel["tasks"] if t["terrain"] == terrain}
            candidate = [r for r in outcomes["candidate"] if r["level"] in levels]
            production = [r for r in outcomes["production"] if r["level"] in levels]
            if candidate:
                group = report(candidate, production)
                pooled = group["pooled"]
                pooled["mcnemar_better_p"] = report(production, candidate)["pooled"]["mcnemar_worse_p"]
                group["complete"] = len(candidate) == len(levels) * 32
                group["eligible"] = (group["complete"] and pooled["ci_noninferior_margin_2_of_32"] and all(
                    r["point_noninferior_margin_2_of_32"] and r["mcnemar_worse_p"] >= .05
                    for r in group["per_level"].values()))
                terrains[terrain] = group
        result.update(terrains=terrains, panel=panel, calibration=calibration)
        write(base / "report.json", result)
        print(json.dumps({k: dict(complete=v["complete"], eligible=v["eligible"], pooled=v["pooled"])
                          for k, v in terrains.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
