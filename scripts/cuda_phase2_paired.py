"""Paired comparison of two search_play game arms (phase 2 teacher-strength gate).

Both arms must have been run with the same seeds and the same search configuration; the only
intended difference is the rollout backend (original C++ engine vs the CUDA continuation service).
The gate in the task book is: McNemar exact p > 0.1 and |wins_cpu - wins_gpu| <= 4 over the
first 128 z15 evaluation seeds. A passing gate also requires matching config.json inputs/search
settings, the CPU/CUDA backend roles, and actual GPU continuations in every GPU game. Without these
the statistics are still printed, but gate_passed is false.

Usage:
    python3 scripts/cuda_phase2_paired.py artifacts/cuda/codex_phase2_v19_v3_cpu \
        artifacts/cuda/codex_phase2_v19_v3_gpu --out artifacts/cuda/phase2_paired.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH_FIELDS = ("samples", "per_card", "halving", "horizon", "margin", "min_gap",
                 "rollout_sim_skip", "cpp_rollouts", "max_actions", "dagger_beta", "arms")
INPUT_FILES = ("rollout_policy", "value", "static_value", "manifest", "executable")


def comparison_evidence(cpu_path: Path, gpu_path: Path, cpu: dict, gpu: dict, task_id: str) -> dict:
    """Check the experiment being compared, as well as the outcome statistics.

    Matching outcomes from two CPU arms, different students, or DAgger games do not constitute the
    teacher-strength gate. Worker counts affect speed and are reported, but do not change its semantics.
    """
    configs, hashes, errors = {}, {}, []
    for name, path in (("cpu", cpu_path), ("gpu", gpu_path)):
        games = cpu if name == "cpu" else gpu
        if any(game.get("arm") != "fast_search" for game in games.values()):
            errors.append(f"{name}: game records must belong to fast_search")
        directory = path if path.is_dir() else path.parent
        config_path = directory / "config.json"
        if not config_path.is_file():
            errors.append(f"{name}: missing config.json")
            continue
        config = configs[name] = json.loads(config_path.read_text())
        hashes[name] = {}
        expected_backend = "cpp" if name == "cpu" else "cuda"
        if config.get("rollout_backend") != expected_backend:
            errors.append(f"{name}: backend must be {expected_backend}")
        if config.get("task_id") != task_id:
            errors.append(f"{name}: task must be {task_id}")
        if str(config.get("dagger_beta")) != "1.0" or config.get("arms") != "fast_search":
            errors.append(f"{name}: teacher gate requires fast_search with dagger_beta=1.0")
        for field in SEARCH_FIELDS:
            if field not in config:
                errors.append(f"{name}: missing search setting {field}")
        for field in INPUT_FILES:
            value = config.get(field)
            if field == "static_value" and value in (None, "None"):
                value = config.get("value")
            path = Path(value) if value not in (None, "None") else None
            if path is not None and not path.is_absolute():
                path = ROOT / path
            if path is None or not path.is_file():
                errors.append(f"{name}: missing input file {field}")
                continue
            hashes[name][field] = hashlib.sha256(path.read_bytes()).hexdigest()
    if len(configs) == 2:
        for field in SEARCH_FIELDS:
            if configs["cpu"].get(field) != configs["gpu"].get(field):
                errors.append(f"different search setting: {field}")
        for field in INPUT_FILES:
            if hashes["cpu"].get(field) != hashes["gpu"].get(field):
                errors.append(f"different input contents: {field}")
    uncovered = sorted(seed for seed, game in gpu.items()
                       if not (game.get("cuda_batches", 0) > 0 and game.get("cuda_lanes", 0) > 0))
    contaminated = sorted(seed for seed, game in cpu.items()
                          if game.get("cuda_batches", 0) != 0 or game.get("cuda_lanes", 0) != 0)
    # Older evidence only recorded successful CUDA calls. New evidence must also show that no
    # unsupported root, server failure or failed lane silently replaced part of the GPU arm.
    coverage_recorded = any("cuda_coverage_version" in game for game in gpu.values())
    incomplete = sorted(seed for seed, game in gpu.items() if coverage_recorded and (
        game.get("cuda_coverage_version") != 1
        or game.get("rollout_batches") != game.get("cuda_batches")
        or game.get("rollout_lanes") != game.get("cuda_lanes")
        or any(game.get(key, -1) != 0 for key in (
            "cuda_error_lanes", "cuda_fallback_batches", "cuda_fallback_lanes"))))
    return {"gate_matching_configuration": not errors,
            "configuration_errors": errors, "input_sha256": hashes,
            "workers_cpu": configs.get("cpu", {}).get("workers"),
            "workers_gpu": configs.get("gpu", {}).get("workers"),
            "gate_gpu_used_every_game": bool(gpu) and not uncovered,
            "gpu_uncovered_seeds": uncovered,
            "gate_gpu_all_continuations": not incomplete if coverage_recorded else None,
            "gpu_incomplete_continuation_seeds": incomplete,
            "gate_cpu_without_cuda": bool(cpu) and not contaminated,
            "cpu_cuda_seeds": contaminated}


def load_games(path: Path) -> dict[int, dict]:
    """Read a games.jsonl (or the directory holding it) keyed by seed."""
    if path.is_dir():
        path = path / "games.jsonl"
    games: dict[int, dict] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seed = int(row["seed"])
            if seed in games:
                raise ValueError(f"{path}: duplicate seed {seed}")
            games[seed] = row
    return games


def mcnemar_exact(cpu_only: int, gpu_only: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs."""
    discordant = cpu_only + gpu_only
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(min(cpu_only, gpu_only) + 1))
    return min(1.0, 2 * tail / 2 ** discordant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cpu", type=Path, help="directory or games.jsonl of the original-backend arm")
    parser.add_argument("gpu", type=Path, help="directory or games.jsonl of the CUDA-backend arm")
    parser.add_argument("--out", type=Path, default=None, help="write the summary JSON here")
    parser.add_argument("--seeds", type=int, default=128, help="how many of the first seeds must be present")
    parser.add_argument("--seed-start", type=int, default=30000, help="first required evaluation seed")
    parser.add_argument("--task-id", default="day_l7_z15_eval", help="required task in both run configurations")
    args = parser.parse_args()

    cpu = load_games(args.cpu)
    gpu = load_games(args.gpu)
    seeds = sorted(set(cpu) & set(gpu))
    required = set(range(args.seed_start, args.seed_start + args.seeds))
    missing = sorted(required - (set(cpu) & set(gpu)))
    unexpected = sorted((set(cpu) | set(gpu)) - required)
    complete = set(cpu) == set(gpu) == required
    evidence = comparison_evidence(args.cpu, args.gpu, cpu, gpu, args.task_id)
    experiment_valid = (evidence["gate_matching_configuration"] and evidence["gate_gpu_used_every_game"]
                        and evidence["gate_cpu_without_cuda"]
                        and evidence["gate_gpu_all_continuations"] is not False)
    pairs = [(cpu[seed], gpu[seed]) for seed in seeds]

    cpu_only = sum(c["won"] and not g["won"] for c, g in pairs)
    gpu_only = sum(g["won"] and not c["won"] for c, g in pairs)
    wins_cpu = sum(c["won"] for c, _g in pairs)
    wins_gpu = sum(g["won"] for _c, g in pairs)
    p_value = mcnemar_exact(cpu_only, gpu_only)

    disagreements = [
        {
            "seed": seed,
            "cpu_won": c["won"],
            "gpu_won": g["won"],
            "cpu_final_wave": c.get("final_wave"),
            "gpu_final_wave": g.get("final_wave"),
            "cpu_seconds": c.get("seconds"),
            "gpu_seconds": g.get("seconds"),
        }
        for seed, (c, g) in zip(seeds, pairs)
        if c["won"] != g["won"]
    ]

    seconds_cpu = sum(c.get("seconds", 0.0) for c, _g in pairs)
    seconds_gpu = sum(g.get("seconds", 0.0) for _c, g in pairs)
    summary = {
        "games_paired": len(pairs),
        "games_required": args.seeds,
        "missing_seeds": missing,
        "unexpected_seeds": unexpected,
        "gate_complete_seed_set": complete,
        "wins_cpu": wins_cpu,
        "wins_gpu": wins_gpu,
        "win_rate_cpu": round(wins_cpu / max(1, len(pairs)), 4),
        "win_rate_gpu": round(wins_gpu / max(1, len(pairs)), 4),
        "discordant_cpu_only": cpu_only,
        "discordant_gpu_only": gpu_only,
        "mcnemar_exact_p": round(p_value, 4),
        "win_count_difference": wins_cpu - wins_gpu,
        "gate_mcnemar_p_gt_0_1": p_value > 0.1,
        "gate_win_difference_le_4": abs(wins_cpu - wins_gpu) <= 4,
        "gate_passed": complete and experiment_valid and p_value > 0.1 and abs(wins_cpu - wins_gpu) <= 4,
        "final_wave_mean_cpu": round(sum(c.get("final_wave", 0) for c, _g in pairs) / max(1, len(pairs)), 3),
        "final_wave_mean_gpu": round(sum(g.get("final_wave", 0) for _c, g in pairs) / max(1, len(pairs)), 3),
        "decisions_cpu": sum(c.get("decisions", 0) for c, _g in pairs),
        "decisions_gpu": sum(g.get("decisions", 0) for _c, g in pairs),
        "seconds_cpu": round(seconds_cpu, 2),
        "seconds_gpu": round(seconds_gpu, 2),
        "seconds_per_game_cpu": round(seconds_cpu / max(1, len(pairs)), 2),
        "seconds_per_game_gpu": round(seconds_gpu / max(1, len(pairs)), 2),
        "cuda_batches_gpu": sum(g.get("cuda_batches", 0) for _c, g in pairs),
        "cuda_lanes_gpu": sum(g.get("cuda_lanes", 0) for _c, g in pairs),
        "disagreements": disagreements,
        **evidence,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "disagreements"}, indent=1))
    if disagreements:
        print("disagreements:")
        for row in disagreements:
            print(" ", json.dumps(row))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
