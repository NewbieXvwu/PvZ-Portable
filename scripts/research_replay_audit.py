"""Audit frozen-policy rollout shards through the real PPO update path."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch
from pvz_agent_model import GameplayModelV1, configure_torch_threads, replay_log_probs
from pvz_seed_jobs import atomic_json, read_episode
from train_pvz_ppo import add_advantages, train_update


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--shard-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    configure_torch_threads(1)
    device = torch.device(args.device)
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    paths = sorted(args.shard_directory.glob("seed_*.npz"), key=lambda path: int(path.stem.split("_")[1]))
    paths = paths[:args.episodes]
    if len(paths) != args.episodes:
        raise ValueError("not enough shards for the preregistered replay audit")
    episodes = [read_episode(path) for path in paths]
    gamma = checkpoint["experiment_config"]["reward"]["gamma"]
    add_advantages(episodes, 0.95, gamma)
    report = {"checkpoint": str(args.checkpoint), "shard_directory": str(args.shard_directory),
              "episode_count": len(episodes), "decision_count": sum(len(e["transitions"]) for e in episodes),
              "job_ids": [episode["seed"] for episode in episodes], "device": args.device,
              "unchanged_weights_log_probability_tolerance": 5e-5, "update_paths": {}}
    for sequence_length in (0, 16):
        model = GameplayModelV1(checkpoint["config"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0)
        losses = train_update(model, copy.deepcopy(episodes), optimizer, device,
                              1, sequence_length, 0.2, 0.5, 0.01,
                              minibatch_chunks=2, attention_backend="dense")
        report["update_paths"][str(sequence_length)] = losses
    model = GameplayModelV1(checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])
    old_reset_error = 0.0
    value_error = 0.0
    with torch.no_grad():
        for episode in episodes:
            transitions = episode["transitions"]
            hidden = None
            for start in range(0, len(transitions), 16):
                sequence = transitions[start:start + 16]
                reset_outputs, _ = model.forward_sequences([sequence], [None])
                reset_lp, _ = replay_log_probs(model, reset_outputs, sequence)
                old_lp = torch.tensor([step["log_prob"] for step in sequence], device=device)
                old_reset_error = max(old_reset_error, float((reset_lp - old_lp).abs().max()))
                outputs, next_hidden = model.forward_sequences([sequence], [hidden])
                hidden = next_hidden[:, 0, :]
                belief = torch.cat([output["belief"] for output in outputs])
                values = model.privileged_value_batch(belief, torch.tensor([step["critic_extra"] for step in sequence], device=device)).squeeze(-1)
                value_error = max(value_error, float((values - torch.tensor([step["value"] for step in sequence], device=device)).abs().max()))
    report["historical_zero_reset_max_log_prob_error"] = old_reset_error
    report["chained_value_max_error"] = value_error
    report["gate_result"] = "pass" if all(
        losses["max_log_prob_change"] < 5e-5 and losses["clip_fraction"] == 0
        for losses in report["update_paths"].values()) and value_error < 5e-5 else "fail"
    atomic_json(args.output, report)
    print(report["gate_result"], report["decision_count"], report["update_paths"], flush=True)
    if report["gate_result"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
