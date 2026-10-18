"""Paired comparison of two search_play game arms (phase 2 teacher-strength gate).

Both arms must have been run with the same seeds and the same search configuration; the only
intended difference is the rollout backend (original C++ engine vs the CUDA continuation service).
The gate in the task book is: McNemar exact p > 0.1 and |wins_cpu - wins_gpu| <= 4 over the
first 128 z15 evaluation seeds.

Usage:
    python3 scripts/cuda_phase2_paired.py artifacts/cuda/codex_phase2_v19_v3_cpu \
        artifacts/cuda/codex_phase2_v19_v3_gpu --out artifacts/cuda/phase2_paired.json
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


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
    args = parser.parse_args()

    cpu = load_games(args.cpu)
    gpu = load_games(args.gpu)
    seeds = sorted(set(cpu) & set(gpu))
    required = set(range(args.seed_start, args.seed_start + args.seeds))
    missing = sorted(required - (set(cpu) & set(gpu)))
    unexpected = sorted((set(cpu) | set(gpu)) - required)
    complete = set(cpu) == set(gpu) == required
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
        "gate_passed": complete and p_value > 0.1 and abs(wins_cpu - wins_gpu) <= 4,
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
