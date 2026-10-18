"""Compare original/host/GPU value features after forced short waits from real roots.

Unlike long policy rollouts, a one-tick wait exposes timing and import errors without student
decisions in the way. Longer waits show how those errors accumulate. Different RNG streams can
still affect headless bleed-out, future spawns and new animation draws; the report labels roots
and names every differing feature instead of treating distribution agreement as exact agreement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--task-id", default="day_l7_z15_train")
    p.add_argument("--policy", required=True)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--seed-offset", type=int, default=11000)
    p.add_argument("--roots-per-game", type=int, default=24)
    p.add_argument("--stride", type=int, default=600)
    p.add_argument("--waits", default="1,35,60,180")
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--executable", default=str(ROOT / "build/pvz-portable"))
    p.add_argument("--device", default=str(ROOT / "build/rollout_cuda"))
    p.add_argument("--host", default=str(ROOT / "build/sim_host"))
    p.add_argument("--resource-dir", default=os.path.expanduser("~/.cache/pvz-research-resources"))
    p.add_argument("--skip-host", action="store_true")
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    waits = [int(t) for t in args.waits.split(",")]
    if any(t <= 0 for t in waits):
        p.error("--waits must be positive")
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from pvz_env import PvZEnv, TaskSpec
    from pvz_rollout_policy import load_student, export_student
    from pvz_value_net import featurize_numpy, observation_context
    from pvz_cuda_root import read_cudaroot
    from render_episode import load_training_task, training_task_kwargs, _frame
    from scripted_baseline import profile_for_deck
    from cuda_leaf_compare import read_features, describe, read_gpu

    net_path = args.work / "student.pvzrn"
    export_student(load_student(args.policy), str(net_path))
    env = PvZEnv(args.resource_dir, executable=args.executable, headless=True)
    env.load_rollout_net(str(net_path))
    manifest = load_training_task(Path(args.manifest), args.task_id)
    seeds = manifest["seeds"][args.seed_offset:args.seed_offset + args.seeds]
    roots_path = args.work / "roots.cudaroot"
    pairs, reference, metadata = [], [], []
    try:
        for seed in seeds:
            kw = training_task_kwargs(manifest, seed)
            deck = tuple(kw["deck"])
            task = TaskSpec(level=kw["level"], seed=seed, playthrough=2, profile=profile_for_deck(deck),
                            **{k: v for k, v in kw["task_extra"].items() if v is not None})
            env.set_lean_observation(False)
            obs, _ = env.reset(deck=deck, task=task)
            env.set_lean_observation(True)
            context = observation_context(obs)
            for _ in range(args.roots_per_game):
                obs = env.rollout(None, obs["tick"] + args.stride, sim_skip=0)
                if obs["terminal"]:
                    break
                root_id = env.snapshot()
                root_index = len(metadata) // len(waits)
                try:
                    response = env.cuda_dump(roots_path, write_level=root_index == 0)
                    if not response.get("supported"):
                        raise RuntimeError(f"unsupported root: {response}")
                    for ticks in waits:
                        future = (seed * 1000 + root_index) % (2 ** 32)
                        child = env.rollout({"type": "wait", "ticks": ticks, "until": "timeout"},
                                            obs["tick"] + ticks, snapshot_id=root_id, seed=future,
                                            sim_skip=0, max_actions=1)
                        board, scalars = featurize_numpy(_frame(child))
                        reference.append(np.concatenate([board.reshape(-1), scalars, context]))
                        pairs.append(f"{root_index} {future} 2 -1 0 0 {ticks}\n")
                        metadata.append({"root": root_index, "seed": seed, "tick": obs["tick"],
                                         "wait": ticks, "end_tick": child["tick"]})
                    env.restore(root_id)
                finally:
                    env.release_snapshot(root_id)
    finally:
        env.close()
    if not reference:
        raise RuntimeError("no roots collected")
    expected = np.asarray(reference, dtype=np.float32)
    (args.work / "reference.bin").write_bytes(b"PVZLEAF1" + struct.pack("<ii", *expected.shape)
                                               + expected.tobytes())
    (args.work / "pairs.txt").write_text("".join(pairs))
    corpus = read_cudaroot(roots_path)
    for item in metadata:
        root = corpus["roots"][item["root"]]
        item["root_headless"] = sum(not(z["state_bits"] & 4) for z in root["zombies"])
        item["root_zombie_phases"] = dict(Counter(z["phase"] for z in root["zombies"]))
        item["root_plant_states"] = dict(Counter((p["type"], p["state"]) for p in root["plants"]))
        item["may_spawn_wave"] = (0 < root["zombie_count_down"] <= item["wait"] or
                                  0 < root["huge_wave_count_down"] <= item["wait"])
        item["root_plant_states"] = {str(k): v for k, v in item["root_plant_states"].items()}
    (args.work / "inputs.json").write_text(json.dumps({"arguments": vars(args) | {"work": str(args.work)},
                                                       "pairs": metadata}, indent=1))
    base = ["--roots", str(roots_path), "--net", str(net_path), "--pairs", str(args.work / "pairs.txt"),
            "--horizon", "1", "--max-actions", "1"]
    backends = [("gpu", args.device)] if args.skip_host else [("host", args.host), ("gpu", args.device)]
    features = {}
    for name, binary in backends:
        command = [binary, *base, "--leaf-out", str(args.work / f"{name}_leaf.bin")]
        if name == "gpu":
            command += ["--out", str(args.work / "gpu_results.bin"), "--lanes", str(len(pairs))]
        run = subprocess.run(command, capture_output=True, text=True)
        (args.work / f"{name}.txt").write_text(run.stdout + run.stderr)
        if run.returncode:
            raise RuntimeError(f"{name} failed: {run.stderr[-2000:]}")
        features[name] = np.asarray(read_features(args.work / f"{name}_leaf.bin"))
    records = read_gpu(args.work / "gpu_results.bin")
    report = {"roots": len(corpus["roots"]), "pairs": len(pairs), "by_wait": {}, "differences": []}
    for ticks in waits:
        indices = [i for i, item in enumerate(metadata) if item["wait"] == ticks]
        diff = features["gpu"][indices] != expected[indices]
        row = {"pairs": len(indices), "differing_pairs": int(np.any(diff, axis=1).sum()),
               "differing_floats": int(diff.sum()),
               "channels": dict(Counter(int(i // 60) if i < 5520 else f"scalar{int(i - 5520)}"
                                        for i in np.nonzero(diff)[1]))}
        stable = [i for i in indices if not metadata[i]["root_headless"] and not metadata[i]["may_spawn_wave"]]
        row["no_initial_bleed_or_spawn_pairs"] = len(stable)
        row["no_initial_bleed_or_spawn_differing_pairs"] = int(np.any(
            features["gpu"][stable] != expected[stable], axis=1).sum())
        if "host" in features:
            row["host_gpu_differing_floats"] = int(np.count_nonzero(
                features["host"][indices] != features["gpu"][indices]))
        report["by_wait"][str(ticks)] = row
    for i, item in enumerate(metadata):
        where = np.flatnonzero(expected[i] != features["gpu"][i])
        if not len(where):
            continue
        report["differences"].append({**item, "gpu_end_tick": records[i]["endTick"],
            "differing_floats": len(where), "features": [
                {"name": describe(int(index)), "cpu": float(expected[i, index]),
                 "gpu": float(features["gpu"][i, index])} for index in where[:12]]})
    (args.work / "comparison.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "differences"}, indent=1))


if __name__ == "__main__":
    main()
