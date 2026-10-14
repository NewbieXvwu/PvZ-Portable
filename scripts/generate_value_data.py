"""Generate episode archives for value-network training (DESIGN.md §13, stage 1).

Every game goes through ``episode_query.capture`` -- the same vetted path the teacher and
the failure analysis use (deck and task taken from the training manifest, GRU state carried
step by step, the event-wait env chosen by the model config). Output layout matches the
existing archives: ``<out>/seed_<s>/<label>/{meta.json,frames.jsonl}``.

Seeds come from the **training** task (1400000-1400255), never from the evaluation seeds
(30000+), so a value net trained on this data can be used to evaluate search on held-out seeds.

Usage:
  python scripts/generate_value_data.py --out artifacts/value_net/episodes \
      --checkpoint u0=artifacts/value_net/checkpoints/.../update_000000_....pt \
      --checkpoint u84=... --modes greedy,sampled --workers 14
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "experiments/t7/bc_finetune_level7_v1/train.json"
TASK_ID = "train_full_level7_v1_d1"
DEFAULT_RESOURCE_DIR = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"


def _init_worker() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))


def _run(job: dict) -> dict:
    import episode_query
    from render_episode import load_training_task, training_task_kwargs
    out_dir = Path(job["out"])
    if (out_dir / "meta.json").exists():
        return {"job": job["name"], "skipped": True}
    kw = training_task_kwargs(load_training_task(MANIFEST, TASK_ID), job["seed"])
    started = time.time()
    tmp = out_dir.with_name(out_dir.name + ".partial")
    res = episode_query.capture(job["resource_dir"], kw["seed"], kw["level"], "ppo", kw["deck"],
                                tmp, job["max_actions"], checkpoint=job["checkpoint"],
                                deterministic=job["mode"] == "greedy", task_extra=dict(kw["task_extra"]),
                                policy_rng_seed=job["policy_rng_seed"])
    tmp.rename(out_dir)
    outcome = res["meta"]["outcome"]
    return {"job": job["name"], "won": outcome["won"], "wave": outcome["final_wave"],
            "frames": res["meta"]["frame_count"], "seconds": round(time.time() - started, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", action="append", required=True, help="label=path")
    p.add_argument("--modes", default="greedy,sampled")
    p.add_argument("--seeds", type=int, default=256, help="first N training seeds")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--resource-dir", default=DEFAULT_RESOURCE_DIR)
    args = p.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "python"))
    from render_episode import load_training_task
    seeds = load_training_task(MANIFEST, TASK_ID)["seeds"][: args.seeds]

    jobs = []
    for spec in args.checkpoint:
        label, path = spec.split("=", 1)
        for mode in args.modes.split(","):
            for seed in seeds:
                name = f"{label}_{mode}"
                jobs.append({"name": f"{name}/{seed}", "seed": seed, "mode": mode,
                             "checkpoint": str(Path(path).resolve()),
                             "policy_rng_seed": seed * 7 + len(jobs) % 7,
                             "out": str(args.out / f"seed_{seed}" / name),
                             "resource_dir": args.resource_dir, "max_actions": args.max_actions})
    for job in jobs:
        Path(job["out"]).parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} games on {args.workers} workers", flush=True)

    started, done, wins = time.time(), 0, 0
    log = (args.out / "generation_log.jsonl").open("a")
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        for res in pool.imap_unordered(_run, jobs):
            done += 1
            wins += int(res.get("won", False))
            log.write(json.dumps(res) + "\n"); log.flush()
            if done % 50 == 0 or done == len(jobs):
                rate = done / (time.time() - started) * 3600
                print(f"{done}/{len(jobs)} done, {wins} won, {rate:.0f} games/hour", flush=True)


if __name__ == "__main__":
    main()
