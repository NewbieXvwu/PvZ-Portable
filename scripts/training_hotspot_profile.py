"""Measure where a full T5 training run actually spends its time.

The rollout and the PPO update were both profiled before, but several per-update
fixed costs were never measured:

  * copying the state dict and hashing it for provenance (3.68M parameters),
  * ``episode_hash`` over every episode's transitions,
  * the checkpoint write,
  * the periodic evaluation, which runs the whole held-out set single-threaded.

This script measures each phase on real episodes and extrapolates to a full run
so the largest cost is visible rather than assumed.

Usage::

    python scripts/training_hotspot_profile.py                # full profile
    python scripts/training_hotspot_profile.py --episodes 4   # smaller sample
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_agent_model import GameplayModelV1, configure_torch_threads  # noqa: E402
from pvz_env import PvZEnv, TaskSpec  # noqa: E402
from train_pvz_ppo import (  # noqa: E402
    add_advantages,
    collect_task_episode,
    episode_digest,
    episode_hash,
    train_update,
)
import t4_capability_profile  # noqa: E402

DEFAULT_RESOURCES = Path.home() / ".cache/pvz-research-resources"
LOCAL_RESOURCES = Path("/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN")
ROLLOUT_EPISODES = 2000      # the configured PPO rollout batch
MAX_EPISODES_PER_RUN = 20_000
EVAL_INTERVAL = 5000         # evaluations happen every 5000 episodes


def _task_spec(task: dict) -> TaskSpec:
    return TaskSpec(
        level=task["level"], seed=task["seeds"][0], playthrough=task["playthrough"],
        zombie_count_multiplier=task["zombie_count_multiplier"], wave_cap=task["wave_cap"],
        preplanted=tuple(tuple(item) for item in task["preplanted"]),
    )


def _row(name: str, seconds: float, total: float | None = None, note: str = "") -> None:
    share = f"{seconds / total * 100:6.1f}%" if total else "      "
    print(f"  {name:38s} {seconds * 1e3:10.2f} ms  {share}  {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-dir", type=Path, default=LOCAL_RESOURCES)
    parser.add_argument("--episodes", type=int, default=6,
                        help="rollout episodes to sample for the per-episode costs")
    parser.add_argument("--update-episodes", type=int, default=8,
                        help="episodes fed to train_update when timing the update path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    resource_dir = args.resource_dir.expanduser()
    if not resource_dir.is_dir():
        raise SystemExit(f"resource dir not found: {resource_dir}")

    configure_torch_threads(1)
    torch.manual_seed(0)
    device = torch.device(args.device)
    model = GameplayModelV1().to(device).eval()

    train_manifest = json.loads((ROOT / "artifacts/task_family/train.json").read_text())
    cap1 = [t for t in train_manifest["tasks"]
            if t["wave_cap"] == 1 and t["zombie_count_multiplier"] == 1.0]

    print(f"device={device}  episodes={args.episodes}  rollout batch={ROLLOUT_EPISODES}")
    print()

    # ---------------------------------------------------------------- rollout
    episodes: list[dict] = []
    with PvZEnv(resource_dir) as env:
        print(f"--- Phase 1: rollout ({args.episodes} episodes, real env) ---")
        aggregate: dict[str, float] = {}
        decisions = 0
        wall_start = time.perf_counter()
        for index in range(args.episodes):
            task = cap1[index % len(cap1)]
            episode = collect_task_episode(
                model, env, task, task["seeds"][index % len(task["seeds"])],
                job_id=index, max_actions=4000)
            episodes.append(episode)
            decisions += len(episode["transitions"])
            for key, value in episode["profile_seconds"].items():
                aggregate[key] = aggregate.get(key, 0.0) + value
        wall = time.perf_counter() - wall_start
        per_episode = wall / args.episodes
        for key, value in sorted(aggregate.items(), key=lambda kv: -kv[1]):
            _row(key, value / args.episodes, per_episode)
        _row("WALL (incl. env subprocess I/O)", per_episode, per_episode)
        print(f"  {args.episodes} episodes, {decisions} decisions,"
              f" {decisions / args.episodes:.1f} decisions/episode")
        print()

        # ------------------------------------------------- per-update overhead
        print("--- Phase 2: per-update fixed overhead (never measured before) ---")
        state_start = time.perf_counter()
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        state_seconds = time.perf_counter() - state_start

        hash_start = time.perf_counter()
        digest = t4_capability_profile._state_sha256(model_state)
        hash_seconds = time.perf_counter() - hash_start

        hash_start = time.perf_counter()
        for episode in episodes:
            episode_hash(episode)
        legacy_seconds_per_episode = (time.perf_counter() - hash_start) / len(episodes)

        digest_start = time.perf_counter()
        for episode in episodes:
            episode_digest(episode)
        digest_seconds_per_episode = (time.perf_counter() - digest_start) / len(episodes)

        checkpoint_path = Path("/tmp/pvz_hotspot_checkpoint.pt")
        save_start = time.perf_counter()
        torch.save({"state_dict": model_state, "note": "hotspot probe"}, checkpoint_path)
        save_seconds = time.perf_counter() - save_start
        checkpoint_bytes = checkpoint_path.stat().st_size
        checkpoint_path.unlink()

        _row("state_dict copy (3.68M params)", state_seconds)
        _row("_state_sha256", hash_seconds, note=f"-> {digest[:12]}")
        _row("episode_hash JSON path (per episode)", legacy_seconds_per_episode,
             note=f"x{ROLLOUT_EPISODES} = {legacy_seconds_per_episode * ROLLOUT_EPISODES:.1f} s")
        _row("episode_digest bytes (per episode)", digest_seconds_per_episode,
             note=f"x{ROLLOUT_EPISODES} = {digest_seconds_per_episode * ROLLOUT_EPISODES:.1f} s"
                  f"  ({legacy_seconds_per_episode / digest_seconds_per_episode:.1f}x faster)")
        _row("torch.save checkpoint", save_seconds,
             note=f"{checkpoint_bytes / 1024 / 1024:.1f} MiB")
        fixed = (state_seconds + hash_seconds + save_seconds
                 + digest_seconds_per_episode * ROLLOUT_EPISODES)
        print(f"  {'fixed overhead per update':38s} {fixed:10.2f} s")
        print()

        # ------------------------------------------------------- update phase
        print(f"--- Phase 3: PPO update ({args.update_episodes} episodes, "
              f"ppo_epochs=2, chunks=16) ---")
        update_episodes = episodes[:args.update_episodes]
        add_advantages(update_episodes, 0.95)
        transitions = sum(len(e["transitions"]) for e in update_episodes)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        update_start = time.perf_counter()
        losses = train_update(model, update_episodes, optimizer, device, 2, 16,
                              0.2, 0.5, 0.01, minibatch_chunks=16,
                              attention_backend="auto")
        update_seconds = time.perf_counter() - update_start
        per_transition = update_seconds / transitions
        print(f"  {transitions} transitions -> {update_seconds:.3f} s"
              f"  ({per_transition * 1e6:.1f} us/transition)")
        print(f"  projected for a {ROLLOUT_EPISODES}-episode batch:"
              f" {per_transition * ROLLOUT_EPISODES * (transitions / len(update_episodes)):.1f} s")
        print(f"  losses: policy={losses['policy_loss']:.4f} value={losses['value_loss']:.4f}"
              f" entropy={losses['entropy']:.3f}")
        print()

        # ------------------------------------------------------- evaluation
        if not args.skip_eval:
            print("--- Phase 4: evaluation (single-threaded, held-out set) ---")
            heldout = json.loads((ROOT / "artifacts/task_family/heldout.json").read_text())
            sample = heldout["tasks"][0]
            eval_start = time.perf_counter()
            t4_capability_profile.run_episode(env, sample, sample["seeds"][0], "checkpoint", model)
            one_episode = time.perf_counter() - eval_start
            total_eval_episodes = sum(len(t["seeds"]) for t in heldout["tasks"])
            print(f"  one held-out episode: {one_episode * 1e3:.1f} ms")
            print(f"  held-out set: {len(heldout['tasks'])} tasks,"
                  f" {total_eval_episodes} episodes")
            eval_seconds = one_episode * total_eval_episodes
            print(f"  projected full evaluation: {eval_seconds:.1f} s"
                  f"  (x{MAX_EPISODES_PER_RUN // EVAL_INTERVAL} evaluations per run"
                  f" = {eval_seconds * (MAX_EPISODES_PER_RUN // EVAL_INTERVAL):.1f} s)")
            print()

    # ------------------------------------------------------------ projection
    print("--- Phase 5: projection for one 20,000-episode run ---")
    updates = MAX_EPISODES_PER_RUN // ROLLOUT_EPISODES
    rollout_total = per_episode * MAX_EPISODES_PER_RUN
    update_total = per_transition * ROLLOUT_EPISODES * (decisions / args.episodes) * updates
    fixed_total = fixed * updates
    if not args.skip_eval:
        eval_total = one_episode * total_eval_episodes * (MAX_EPISODES_PER_RUN // EVAL_INTERVAL)
    else:
        eval_total = 0.0
    grand = rollout_total + update_total + fixed_total + eval_total
    print(f"  {'rollout':30s} {rollout_total:9.1f} s  {rollout_total / grand * 100:5.1f}%")
    print(f"  {'PPO update':30s} {update_total:9.1f} s  {update_total / grand * 100:5.1f}%")
    print(f"  {'per-update fixed overhead':30s} {fixed_total:9.1f} s  {fixed_total / grand * 100:5.1f}%")
    print(f"  {'evaluation':30s} {eval_total:9.1f} s  {eval_total / grand * 100:5.1f}%")
    print(f"  {'TOTAL':30s} {grand:9.1f} s  ({grand / 3600:.2f} h)")
    print()
    print("  Note: rollout/update here run single-threaded on CPU; the desktop")
    print("  runs rollout across 18 CPU workers and the update on CUDA.  The")
    print("  fixed overhead and evaluation phases are single-threaded on both.")


if __name__ == "__main__":
    main()
