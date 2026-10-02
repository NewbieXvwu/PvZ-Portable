#!/usr/bin/env python3
"""Repeat one actual PPO update from identical weights, AdamW/RNG and shards."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

import torch
from pvz_agent_model import GameplayModelV1, configure_torch_threads
from pvz_common import sha256_file
from pvz_research import capture_rng, restore_rng
from pvz_seed_jobs import atomic_json, read_episode
from train_pvz_ppo import add_advantages, train_update
from research_observation_interrupt_audit import compare


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("freeze the same CUBLAS_WORKSPACE_CONFIG=:4096:8 as failed probe")
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    if free < 1024:
        raise RuntimeError("insufficient currently free GPU memory for isolated engineering diagnostic")
    initial_path = sorted((source / "runs/run_1").glob("update_000000_initial_*.pt"))[0]
    checkpoint = torch.load(initial_path, map_location="cpu", weights_only=False)
    config = checkpoint["experiment_config"]
    shard_paths = sorted((source / "runs/run_1/.seed_jobs/update_000001").glob("*/seed_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    if len(shard_paths) != 20:
        raise ValueError("require all 20 original first-update shards")
    original = [read_episode(path) for path in shard_paths]
    output.mkdir(parents=True, exist_ok=False)
    configure_torch_threads(1)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    started = time.monotonic()
    report = {"schema_version": 1, "scope": "same-data first-update reproducibility diagnostic, not learning or a resumed-configuration pass",
              "initial_checkpoint": {"path": str(initial_path), "sha256": sha256_file(initial_path)},
              "shards": [{"path": str(path), "sha256": sha256_file(path)} for path in shard_paths],
              "episodes": len(original), "decisions": sum(len(e["transitions"]) for e in original),
              "model_config": checkpoint["config"], "ppo": config["ppo"], "torch": torch.__version__,
              "CUBLAS_WORKSPACE_CONFIG": os.environ["CUBLAS_WORKSPACE_CONFIG"],
              "strict_bit_equality_required": True, "formal_matrix_unchanged": True, "trials": [], "comparisons": []}
    try:
        for deterministic in (False, True):
            pair = []
            for repeat in (0, 1):
                print(f"deterministic_algorithms={deterministic} repeat={repeat}", flush=True)
                torch.use_deterministic_algorithms(deterministic, warn_only=False)
                model = GameplayModelV1(checkpoint["config"]).to("cuda").eval()
                model.load_state_dict(checkpoint["state_dict"])
                optimizer = torch.optim.AdamW(model.parameters(), lr=config["ppo"]["learning_rate"])
                optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
                assignments = random.Random()
                restore_rng(checkpoint["rng_state"], assignments)
                episodes = copy.deepcopy(original)
                add_advantages(episodes, config["ppo"]["gae_lambda"], config["reward"]["gamma"])
                trace = []
                original_step = optimizer.step

                def traced_step(*positional, **keyword):
                    value = original_step(*positional, **keyword)
                    trace.append(parameter_digest(model))
                    return value

                optimizer.step = traced_step
                ppo = config["ppo"]
                trial_started = time.monotonic()
                losses = train_update(model, episodes, optimizer, torch.device("cuda"), ppo["ppo_epochs"],
                    ppo["sequence_length"], ppo["clip_epsilon"], ppo["value_coefficient"],
                    ppo["entropy_coefficient"], ppo["minibatch_chunks"], ppo["attention_backend"],
                    f"determinism-{deterministic}-{repeat}", ppo["target_kl"])
                saved = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                         "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                         "rng_state": capture_rng(assignments), "losses": losses, "optimizer_step_parameter_sha256": trace}
                for values in saved["optimizer_state_dict"]["state"].values():
                    for key, value in values.items():
                        if isinstance(value, torch.Tensor):
                            values[key] = value.cpu().clone()
                path = output / f"deterministic_{int(deterministic)}_repeat_{repeat}.pt"
                torch.save(saved, path)
                pair.append(saved)
                report["trials"].append({"deterministic_algorithms": deterministic, "repeat": repeat,
                    "checkpoint": path.name, "sha256": sha256_file(path), "losses": losses,
                    "optimizer_step_parameter_sha256": trace, "seconds": time.monotonic() - trial_started})
                del model, optimizer, episodes
            differences = []
            for key in ("state_dict", "optimizer_state_dict", "rng_state", "losses"):
                compare(pair[0][key], pair[1][key], key, differences)
            first_step = next((i + 1 for i, (a, b) in enumerate(zip(pair[0]["optimizer_step_parameter_sha256"], pair[1]["optimizer_step_parameter_sha256"], strict=True)) if a != b), None)
            comparison = {"deterministic_algorithms": deterministic, "bit_equal": not differences,
                "first_different_optimizer_step": first_step, "differences": differences,
                "max_parameter_abs_error": max((d.get("max_abs_error", 0) for d in differences if d["path"].startswith("state_dict")), default=0)}
            report["comparisons"].append(comparison)
            print(json.dumps({k: comparison[k] for k in ("deterministic_algorithms", "bit_equal", "first_different_optimizer_step", "max_parameter_abs_error")}), flush=True)
            del pair
        report.update({"gate_result": "diagnostic_only", "seconds": time.monotonic() - started,
                       "supports_deterministic_runtime_hypothesis": report["comparisons"][1]["bit_equal"] and not report["comparisons"][0]["bit_equal"]})
        atomic_json(output / "report.json", report)
    except BaseException as error:
        report.update({"gate_result": "fail", "error": str(error), "traceback": traceback.format_exc(), "seconds": time.monotonic() - started})
        atomic_json(output / f"failure_{time.time_ns()}.json", report)
        raise


if __name__ == "__main__":
    main()
