"""Run the task37 three-task, 32-seed teacher gate with a bounded number of owned simulators."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = [("day_l7_train_d7_m150", "multiplier_panel.json"),
         ("night_l14_train_n10", "night_panel.json"), ("pool_l21_train_p9k", "pool_panel.json")]


def compare(out):
    """Require completed paired games; compare choices only on identical search roots."""
    plans = {phase: json.loads((out / f"{phase}-plan.json").read_text()) for phase in ("before", "after")}
    if plans["before"]["contract"] != plans["after"]["contract"]:
        raise ValueError("before/after gate inputs or settings differ")
    report = {}
    for task, _manifest in TASKS:
        same = points = noise_same = noise_points = before_wins = after_wins = 0
        improved = regressed = 0
        before_cpu = after_cpu = 0.0
        rss = {phase: {who: [] for who in ("worker", "engine")} for phase in ("before", "after")}
        for index in range(32):
            paths = {phase: out / phase / task / f"seed_index_{index}" for phase in ("before", "after")}
            results = {phase: json.loads((path / "result.json").read_text()) for phase, path in paths.items()}
            traces = {phase: json.loads((path / "trace.json").read_text()) for phase, path in paths.items()}
            if any(len(traces[p]) != results[p]["decisions"] for p in paths):
                raise ValueError(f"incomplete trace: {task}/{index}")
            before, after = results["before"], results["after"]
            if before["seed"] != after["seed"]:
                raise ValueError(f"unpaired seeds: {task}/{index}")
            roots = lambda trace: [(r["state_sha256"], r["future_seed"]) for r in trace]
            if roots(traces["before"]) != roots(traces["after"]):
                shadow_plan = json.loads((out / "shadow-plan.json").read_text())
                if shadow_plan != plans["after"]:
                    raise ValueError("shadow implementation or gate inputs differ from after")
                shadow_path = out / "shadow" / task / f"seed_index_{index}"
                shadow_result = json.loads((shadow_path / "result.json").read_text())
                shadow = json.loads((shadow_path / "trace.json").read_text())
                if len(shadow) != shadow_result["decisions"] or roots(shadow) != roots(traces["before"]):
                    raise ValueError(f"unpaired shadow roots: {task}/{index}")
                traces["after"] = shadow
            for old, new in zip(traces["before"], traces["after"]):
                if old["searched"]:
                    points += 1
                    same += old["chosen"] == new["chosen"]
            noise_points += before["noise_points"]
            noise_same += before["noise_agreement"]
            before_wins += before["won"]; after_wins += after["won"]
            improved += after["won"] and not before["won"]
            regressed += before["won"] and not after["won"]
            before_cpu += sum(before["host_cpu_excluding_noise_s"].values())
            after_cpu += sum(after["host_cpu_excluding_noise_s"].values())
            for phase in paths:
                for who in rss[phase]:
                    rss[phase][who].append(int(results[phase]["memory"][who]["VmRSS"].split()[0]))
        if not points or not noise_points:
            raise ValueError(f"missing teacher/noise points: {task}")
        discordant = improved + regressed
        p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2**discordant)
        report[task] = dict(games=32, agreement=same / points, decisions_compared=points,
                            noise_agreement=noise_same / noise_points, noise_points=noise_points,
                            wins=dict(before=before_wins, after=after_wins, improved=improved, regressed=regressed,
                                      paired_binomial_p=p),
                            cpu_s_per_game=dict(before=before_cpu / 32, after=after_cpu / 32),
                            cpu_ratio=after_cpu / before_cpu,
                            rss_mean_kib={phase: {who: sum(values)/32 for who, values in rows.items()}
                                          for phase, rows in rss.items()},
                            passed=(same / points >= noise_same / noise_points and
                                    (regressed <= improved or p >= 0.05) and after_cpu <= before_cpu * 1.02))
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(row["passed"] for row in report.values()):
        raise SystemExit("teacher or CPU acceptance gate failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-code", type=Path, required=True)
    parser.add_argument("--baseline-executable", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--code", type=Path, default=ROOT / "scripts/search_play.py")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after", "shadow", "compare"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        parser.error("workers must be 1..2; each job owns one game process")
    if args.phase == "compare":
        compare(args.out)
        return
    tasks = TASKS
    args.out.mkdir(parents=True, exist_ok=True)
    code = args.baseline_code if args.phase == "before" else args.code
    binary = args.baseline_executable if args.phase == "before" else args.executable
    dependencies = [Path(__file__), ROOT / "scripts/profile_value_engine.py", ROOT / "python/pvz_env.py",
                    ROOT / "python/pvz_value_net.py", ROOT / "python/pvz_rollout_policy.py",
                    ROOT / "scripts/render_episode.py", ROOT / "scripts/export_value_net.py",
                    ROOT / "scripts/hotpath_profile.py"]
    common_paths = [args.config, *dependencies] + [args.manifest_dir / filename for _task, filename in tasks]
    config = json.loads(args.config.read_text())
    common_paths += [Path(config[key]) for key in ("rollout_policy", "value", "static_value")]
    digests = lambda paths: {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    plan = dict(contract=digests(common_paths), implementation=digests([code, binary]))
    plan_path = args.out / f"{args.phase}-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise SystemExit("gate inputs changed; choose a new output directory to avoid mixing evidence")
    plan_path.write_text(json.dumps(plan, indent=2)+"\n")

    def run(job):
        task, manifest, seed_index = job
        dest = args.out / args.phase / task / f"seed_index_{seed_index}"
        if args.phase == "shadow":
            before = json.loads((args.out / "before" / task / f"seed_index_{seed_index}" / "trace.json").read_text())
            after = json.loads((args.out / "after" / task / f"seed_index_{seed_index}" / "trace.json").read_text())
            roots = lambda trace: [(row["state_sha256"], row["future_seed"]) for row in trace]
            if roots(before) == roots(after):
                return dict(task=task, index=seed_index, identical_roots=True)
        if (dest / "result.json").is_file() and (dest / "trace.json").is_file():
            try:
                result = json.loads((dest / "result.json").read_text())
                trace = json.loads((dest / "trace.json").read_text())
                if len(trace) == result["decisions"] and result["host_cpu_excluding_noise_s"]:
                    return dict(task=task, index=seed_index, cached=True)
            except (ValueError, KeyError):
                pass  # Interrupted or old partial output is overwritten by a full rerun below.
        dest.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(ROOT / "scripts/profile_value_engine.py"), "--config", str(args.config),
                   "--manifest", str(args.manifest_dir / manifest), "--task-id", task, "--seed-offset", str(seed_index),
                   "--out", str(dest), "--executable", str(binary), "--code", str(code), "--no-profile"]
        if args.phase == "before":
            command += ["--noise-every", "8"]
        if args.phase == "shadow":
            command += ["--shadow-trace", str(args.out / "before" / task / f"seed_index_{seed_index}" / "trace.json")]
        with (dest / "worker.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"})
        result = json.loads((dest / "result.json").read_text())
        return dict(task=task, index=seed_index, won=result["won"], seconds=result["seconds"])

    jobs = [(task, manifest, i) for i in range(32) for task, manifest in tasks]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, job):job for job in jobs}
        failures = []
        for done, future in enumerate(as_completed(futures), 1):
            try:
                print(json.dumps(dict(done=done, total=len(jobs), **future.result())), flush=True)
            except Exception as error:
                failures.append(dict(job=futures[future], error=str(error)))
                print(json.dumps(failures[-1]), flush=True)
        if failures:
            (args.out / f"{args.phase}-failures.json").write_text(json.dumps(failures, indent=2)+"\n")
            raise SystemExit("teacher gate jobs failed; inspect worker.log, rerun resumes completed games")
    summary = {}
    for task, _manifest in tasks:
        paths = [args.out / args.phase / task / f"seed_index_{i}" / "result.json" for i in range(32)]
        rows = [json.loads(path.read_text()) for path in paths if path.is_file()]
        if args.phase != "shadow" and len(rows) != 32:
            raise ValueError(f"missing completed games: {task}")
        summary[task] = dict(games=len(rows), won=sum(r["won"] for r in rows),
                             noise_points=sum(r["noise_points"] for r in rows),
                             noise_agreement=sum(r["noise_agreement"] for r in rows))
    (args.out / f"{args.phase}-summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
