"""Keep actual new-input rollouts and replay them without changing model parameters."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_agent_model import GameplayModelV1, configure_torch_threads
from pvz_env import PvZEnv
from pvz_seed_jobs import atomic_json, atomic_numpy, read_numpy
from train_pvz_ppo import collect_task_episode, add_advantages, train_update

MODEL = {"layers": 4, "width": 192, "heads": 6, "ff_width": 768,
         "gru_layers": 2, "gru_width": 256, "critic_width": 256, "critic_layers": 1, "input_flags": 7}
SOURCES = ("python/pvz_agent_model.py", "python/pvz_observation_features.py", "python/train_pvz_ppo.py",
           "python/pvz_env.py", "python/pvz_common.py", "python/pvz_seed_jobs.py", "python/pvz_value.py",
           "build/pvz-portable", "artifacts/task_family/train.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("collect-cpu", "replay-cuda"), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    configure_torch_threads(1)
    fingerprints = {source: digest(ROOT / source) for source in SOURCES}
    fingerprints.update({name: digest(args.resource_dir / name) for name in ("main.pak", "properties/partner.xml")})
    report_path = args.output_dir / ("cpu_report.json" if args.stage == "collect-cpu" else "cuda_report.json")
    if report_path.exists():
        raise ValueError("keep existing report and select a new evidence directory")
    device = torch.device("cpu" if args.stage == "collect-cpu" else "cuda")
    if args.stage == "collect-cpu":
        if args.output_dir.exists():
            raise ValueError("collection requires a fresh directory")
        args.output_dir.mkdir(parents=True)
        tasks = [t for t in json.loads((ROOT / "artifacts/task_family/train.json").read_text())["tasks"]
                 if t["wave_cap"] == 1 and t["zombie_count_multiplier"] == 1]
        if len(tasks) != 5 or len({t["terrain"] for t in tasks}) != 5:
            raise ValueError("require the five original cap1 tasks without substitution")
        torch.manual_seed(0)
        model = GameplayModelV1(MODEL).eval()
        checkpoint = args.output_dir / "initial_model.pt"
        torch.save({"config": model.config, "state_dict": model.state_dict()}, checkpoint)
        inventory, episodes = [], []
        with PvZEnv(args.resource_dir) as env:
            for task in tasks:
                for seed in task["seeds"][:4]:
                    job_id = len(episodes)
                    torch.manual_seed(seed + 170000)
                    episode = collect_task_episode(model, env, task, seed, job_id, 4000,
                                                   {"gamma": .99, "shaping_weight": 1}, allow_truncation=False)
                    path = args.output_dir / f"seed_{job_id}.npz"
                    atomic_numpy(path, episode, compressed=True)
                    inventory.append({"path": path.name, "sha256": digest(path), "job_id": job_id,
                                      "task_id": task["task_id"], "environment_seed": seed,
                                      "decisions": len(episode["transitions"]), "won": episode["won"]})
                    episodes.append(episode)
                    print("collected", task["task_id"], seed, len(episode["transitions"]), flush=True)
        atomic_json(args.output_dir / "collection_manifest.json", {
            "model_config": model.config, "initialization_seed": 0,
            "sampling_seed_rule": "torch.manual_seed(environment_seed+170000)",
            "scope": "first four environment seeds of each unchanged cap1 task; independent engineering collection, no learning",
            "fingerprints": fingerprints, "checkpoint_sha256": digest(checkpoint), "episodes": inventory})
    else:
        cpu = json.loads((args.output_dir / "cpu_report.json").read_text())
        manifest = json.loads((args.output_dir / "collection_manifest.json").read_text())
        if cpu["gate_result"] != "pass" or fingerprints != manifest["fingerprints"]:
            raise ValueError("CPU replay not passed or recorded source/resource/binary changed")
        checkpoint = args.output_dir / "initial_model.pt"
        if digest(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("initial checkpoint changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = GameplayModelV1(saved["config"]).eval()
        model.load_state_dict(saved["state_dict"])
        inventory, episodes = manifest["episodes"], []
        for item in inventory:
            path = args.output_dir / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError("stored rollout changed")
            episodes.append(read_numpy(path))
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    original = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.to(device)
    add_advantages(episodes, .95)
    metrics = []
    for sequence_length in (0, 128):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        losses = train_update(model, copy.deepcopy(episodes), optimizer, device, 2, sequence_length,
                              .2, .5, .01, minibatch_chunks=2, attention_backend="dense")
        metrics.append({"sequence_length": sequence_length, "losses": losses})
        print("replay", str(device), sequence_length, losses, flush=True)
        if losses["max_log_prob_change"] >= 3e-6 or losses["clip_fraction"] != 0:
            atomic_json(report_path.with_suffix(".failure.json"), {"metrics": metrics, "fingerprints": fingerprints})
            raise AssertionError("actual new-input likelihood replay exceeds the unchanged tolerance")
    if any(not torch.equal(value, model.state_dict()[key].cpu()) for key, value in original.items()):
        raise AssertionError("lr=0 engineering replay changed model parameters")
    atomic_json(report_path, {"schema_version": 1, "gate_result": "pass", "device": str(device),
                             "scope": "actual new-input cap1 rollouts, real PPO optimizer at lr=0; no learning or capability pass",
                             "model_config": model.config, "fingerprints": fingerprints, "metrics": metrics,
                             "episodes": len(episodes), "decisions": sum(len(e["transitions"]) for e in episodes),
                             "model_state_unchanged": True, "seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
