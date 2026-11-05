"""Task243 leaf replacement: invoke the production game runner, summarize matched games."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pvz_promotion import summarize


def paired(rows, candidate, incumbent):
    pairs = [dict(candidate=r[candidate], incumbent=r[incumbent]) for r in rows]
    result = summarize(pairs)
    n, tail = result["gain"] + result["harm"], min(result["gain"], result["harm"])
    result["mcnemar_two_sided_p"] = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) / 2**n)
    # Historical task43 operating margin: at most two net losses per 32 pairs.
    result["point_noninferior_margin_2_of_32"] = result["delta_rate"] >= -2 / 32
    result["ci_noninferior_margin_2_of_32"] = result["ci95"][0] >= -2 / 32
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/task243")
    parser.add_argument("--executable", type=Path, default=Path.home() / "pvz-mac-collect/build-vn6/pvz-portable")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+", choices=("production", "v5", "v6"),
                        default=("production", "v5", "v6"))
    args = parser.parse_args()
    directory = args.data.resolve()
    models = directory / "models"
    panel = directory / "teacher_panel.json"
    tasks = json.loads(panel.read_text())["tasks"]
    out = directory / "teacher"
    out.mkdir(exist_ok=True)
    environment = dict(os.environ, PVZ_NO_POPUP="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       VECLIB_MAXIMUM_THREADS="1", TMPDIR=str(directory / "work"))
    arms = dict(production=(models / "production_v5.pt", models / "production_v5roof.pt"),
                v5=(models / "v5.pt", models / "v5.pt"), v6=(models / "v6.pt", models / "v6.pt"))
    arms = {arm: arms[arm] for arm in args.arms}
    pairs = []
    for task in tasks:
        level = task["level"]
        outcomes = {}
        for arm, (value, roof) in arms.items():
            target = out / f"l{level}-{arm}"
            command = [sys.executable, str(ROOT / "scripts/search_play.py"),
                "--rollout-policy", str(models / "student.pt"),
                "--rollout-policy-gpt", str(models / "gptbc.pt"),
                "--value", str(value), "--value-roof", str(roof),
                "--static-value", str(models / "static.pt"),
                "--manifest", str(panel), "--task-id", task["task_id"], "--seeds", str(len(task["seeds"])),
                "--arms", "fast_search", "--workers", str(args.workers),
                "--verify-samples", "9", "--student-candidates", "2",
                "--dagger-beta", "1", "--native-games", "--executable", str(args.executable),
                "--resource-dir", str(Path.home() / ".cache/pvz-research-resources"), "--out", str(target)]
            print(f"L{level} {arm}: {len(task['seeds'])} games, {args.workers} workers", flush=True)
            with (out / f"l{level}-{arm}.log").open("a") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=log, check=True)
            games = [json.loads(line) for line in (target / "games.jsonl").read_text().splitlines()]
            outcomes[arm] = {game["seed"]: game["won"] for game in games if game["terminal"]}
            if set(outcomes[arm]) != set(task["seeds"]):
                raise RuntimeError(f"L{level} {arm}: incomplete terminal pairs")
        pairs.extend(dict(level=level, terrain=task["terrain"], seed=seed,
                          **{arm: values[seed] for arm, values in outcomes.items()}) for seed in task["seeds"])
        (out / "pairs.json").write_text(json.dumps(pairs, indent=1))
        print(f"L{level}: " + str({arm: sum(outcomes[arm].values()) for arm in arms}), flush=True)
    if set(arms) != {"production", "v5", "v6"}:
        return
    report = dict(pairs=len(pairs), per_level={}, pooled={},
        noninferiority_margin=-2/32,
        protocol="same reset seed, separate seed blocks by level, shared production search/prior/static; only leaf changes",
        recommendation_rule="each level net >= -2/32 and no one-sided significant harm; pooled paired CI lower >= -2/32")
    for candidate, incumbent in (("v6", "production"), ("v5", "production"), ("v6", "v5")):
        key = f"{candidate}_minus_{incumbent}"
        report["pooled"][key] = paired(pairs, candidate, incumbent)
        for task in tasks:
            level = str(task["level"])
            report["per_level"].setdefault(level, {})[key] = paired(
                [r for r in pairs if r["level"] == task["level"]], candidate, incumbent)
    for arm in ("v5", "v6"):
        key = f"{arm}_minus_production"
        report.setdefault("eligible", {})[arm] = (
            report["pooled"][key]["ci_noninferior_margin_2_of_32"] and all(
                result[key]["point_noninferior_margin_2_of_32"] and result[key]["mcnemar_worse_p"] >= .05
                for result in report["per_level"].values()))
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
