"""Phase 0/1 corpus: real root positions exported for the GPU, plus the CPU reference continuations.

A root is a real decision point of a real game (the student plays the task's seed, we snapshot where
the teacher would search).  Two things come out:

  roots.cudaroot  -- LevelInfo followed by one RootRecord per root, the layout in cuda/CudaRollout.h.
                     This is what the GPU simulator starts from; nothing about the level is
                     re-implemented for it.
  corpus.jsonl    -- one line per (root, random future): the leaf statistics the *original* simulator
                     produced for that exact start (result, waves, kills, plant losses, zombie
                     advance, income) plus the value net's score of the leaf.  The GPU runs the same
                     (root, future) pairs and the phase-1 gate compares the two distributions.

Usage:
  python3 scripts/cuda_root_corpus.py --manifest ~/PvZ-search/artifacts/stage4/day_l7_hard.json \
      --task-id day_l7_z15_train --seeds 12 --roots-per-game 18 --rollouts-per-root 8 \
      --rollout-policy ~/PvZ-search/artifacts/student/v14/model.pt \
      --static-value ~/PvZ-search/artifacts/hard/value_branch1/model.pt \
      --value ~/PvZ-search/artifacts/value_net/v2/model.pt --out-dir ~/pvz-cuda/artifacts/cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _leaf_value(vnet, leaf: dict, context) -> float:
    """Value net score of a leaf: 1/0 for a finished game, sigmoid V(features) otherwise."""
    if leaf.get("terminal"):
        return 1.0 if leaf.get("result") == 1 else 0.0
    import torch
    from pvz_value_net import featurize
    from render_episode import _frame
    board, scalars = featurize(_frame(leaf))
    if vnet.context and context is not None:
        scalars = torch.cat([scalars, context])
    with torch.inference_mode():
        return float(torch.sigmoid(vnet(board[None], scalars[None]))[0])


def _row_min_zombie_x(leaf: dict) -> list[int]:
    """Nearest-to-the-house zombie x per row, fixed point x4 -- what RolloutResult reports.

    Zombies walk *leftward* (mPosX falls from ~780 toward the house at -100), so the most advanced
    zombie of a row is the one with the smallest x.  Keep the minimum, which is what StoreResult
    writes into RolloutResult::rowMinZombieX.
    """
    best = [None] * 6
    for zombie in leaf.get("zombies") or []:
        row, x, on_board = int(zombie[0]), float(zombie[2]), bool(zombie[5])
        if not on_board or not 0 <= row < 6:
            continue
        value = int(round(x * 4))
        if best[row] is None or value < best[row]:
            best[row] = value
    return [-2 ** 31 if item is None else item for item in best]


def build(env, vnet, args, out_dir: Path) -> tuple[list[dict], dict]:
    from pvz_env import TaskSpec
    from pvz_value_net import observation_context
    from render_episode import load_training_task, training_task_kwargs
    from scripted_baseline import profile_for_deck
    import torch

    kw = load_training_task(Path(args.manifest), args.task_id)
    deck = tuple(kw["deck"])
    seeds = kw["seeds"][args.seed_offset: args.seed_offset + args.seeds]
    cudaroot = out_dir / "roots.cudaroot"
    roots: list[dict] = []
    rollouts = 0
    max_events = 0
    started = time.time()

    with (out_dir / "corpus.jsonl").open("w", encoding="utf-8") as corpus:
        for seed in seeds:
            task = TaskSpec(level=kw["level"], seed=seed, playthrough=2,
                            profile=profile_for_deck(deck),
                            **{k: v for k, v in training_task_kwargs(kw, seed)["task_extra"].items()
                               if v is not None})
            obs, _ = env.reset(deck=deck, task=task)
            context = torch.from_numpy(observation_context(obs)) if vnet.context else None
            env.set_lean_observation(True)
            env.set_sim_skip(args.sim_skip)
            root_tick, game_roots = args.root_start, 0
            while not obs["terminal"] and game_roots < args.roots_per_game and obs["tick"] < args.max_root_tick:
                obs = env.rollout(None, root_tick, max_actions=args.max_actions)
                if obs["terminal"]:
                    break
                root_tick = obs["tick"] + args.root_stride
                # the teacher only searches where something is plantable, so a root with no legal
                # plant is not part of the workload
                if not (obs.get("legal_actions") or {}).get("plants"):
                    continue
                game_roots += 1
                root = env.snapshot()
                env.restore(root)
                events = len(env._sun_production_history)
                max_events = max(max_events, events)
                response = env.cuda_dump(cudaroot, write_level=not roots)
                roots.append({"index": len(roots), "seed": seed, "tick": response["tick"],
                              "sun": obs["sun"], "wave": obs["wave"], "income_events": events,
                              "supported": response["supported"], "reason": response["reason"],
                              "plants": response["plants"], "zombies": response["zombies"],
                              "projectiles": response["projectiles"], "coins": response["coins"]})
                if not response["supported"]:
                    env.release_snapshot(root)
                    continue
                for future in range(args.rollouts_per_root):
                    seed_value = (seed * 1000 + future) % 2 ** 32
                    leaf = env.rollout(None, obs["tick"] + args.horizon, sim_skip=args.sim_skip,
                                       snapshot_id=root, seed=seed_value)
                    details = env._last_response
                    record = {"root": len(roots) - 1, "seed": seed, "future": seed_value,
                              "ticks": details["ticks"], "decisions": details["decisions"],
                              "tick": leaf["tick"], "terminal": bool(leaf.get("terminal")),
                              "result": leaf.get("result"), "sun": leaf.get("sun"),
                              "wave": leaf.get("wave"), "income": leaf.get("sun_income_rate"),
                              "events": details.get("events") or {},
                              "plants": len(leaf.get("plants") or []),
                              "zombies": len(leaf.get("zombies") or []),
                              "mowers": len(leaf.get("defenses") or []),
                              "row_min_zombie_x": _row_min_zombie_x(leaf),
                              "leaf_value": _leaf_value(vnet, leaf, context)}
                    rollouts += 1
                    corpus.write(json.dumps(record) + "\n")
                env.restore(root)
                env.release_snapshot(root)
            env.set_sim_skip(0)
            env.set_lean_observation(False)
    return roots, {"roots": len(roots), "rollouts": rollouts, "seeds": len(seeds),
                   "unsupported": sum(1 for record in roots if not record["supported"]),
                   "reasons": sorted({record["reason"] for record in roots if not record["supported"]}),
                   "max_income_events": max_events,
                   "max_root_tick": max((record["tick"] for record in roots), default=0),
                   "seconds": round(time.time() - started, 1)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", required=True)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--roots-per-game", type=int, default=18)
    p.add_argument("--root-start", type=int, default=600)
    p.add_argument("--root-stride", type=int, default=600)
    p.add_argument("--max-root-tick", type=int, default=24000)
    p.add_argument("--horizon", type=int, default=2500)
    p.add_argument("--sim-skip", type=int, default=29)
    p.add_argument("--max-actions", type=int, default=3000)
    p.add_argument("--rollouts-per-root", type=int, default=8)
    p.add_argument("--rollout-policy", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--executable", default=str(ROOT / "build" / "pvz-portable"))
    p.add_argument("--resource-dir", default=os.path.expanduser("~/.cache/pvz-research-resources"))
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.manifest = os.path.expanduser(args.manifest)
    args.resource_dir = os.path.expanduser(args.resource_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import tempfile
    import torch
    torch.set_num_threads(1)
    from pvz_env import PvZEnv
    from pvz_rollout_policy import export_student, load_student
    from pvz_value_net import load_value_net

    tmp = tempfile.TemporaryDirectory(prefix="pvz-corpus-")
    student_file = str(Path(tmp.name) / "student.bin")
    export_student(load_student(args.rollout_policy), student_file)
    vnet = load_value_net(args.value)
    env = PvZEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(student_file)
    roots, summary = build(env, vnet, args, args.out_dir)
    (args.out_dir / "roots.json").write_text(json.dumps({"roots": roots, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
