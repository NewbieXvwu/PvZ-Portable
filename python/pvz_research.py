"""Explicit T5–T7 experiments using the existing collector, PPO and evaluator.

An immutable config identifies one candidate and initialization. Resume restores
the last COMPLETE update, including AdamW and RNG; incomplete rollout shards can
only be reused by their collection-policy/assignment identity.
"""
from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Any

import numpy as np
import torch

from pvz_agent_model import GameplayModelV1, model_architecture_version, configure_torch_threads
from pvz_common import (ENV_PROTOCOL_VERSION, OBSERVATION_VERSION, TASK_VERSION,
                        canonical_digest, git_metadata, sha256_file)
import pvz_curriculum as course
import pvz_task_mutation as mutation
from pvz_initialization import transfer_weights, validate_transfer
from pvz_seed_jobs import atomic_json, atomic_write, run_seed_jobs, seed_job_directory
from train_pvz_ppo import add_advantages, episode_digest, train_update
from pvz_value import VALUE_SEMANTICS
from pvz_wait_events import summarize_wait_records

ROOT = Path(__file__).resolve().parent.parent
RESEARCH_VERSION = 1

# How many `trained` snapshots to keep per run.
#
# Every update writes a full ~43 MB checkpoint, so a 500k-decision run used to
# accumulate 70+ of them (3.1 GB) with nothing ever deleting the older ones.
# Only a few are actually read:
#   * resuming needs the newest one;
#   * `scripts/research_late_policy_probe.py` reads the last 8
#     (`state["update_history"][-8:]`) to measure parameter change across
#     updates, so at least that many must survive.
#
# `evaluated`, `initial` and `boundary` snapshots are milestones referenced by
# the archive and comparison scripts; they are never pruned. Set this to 0 to
# keep only the checkpoint the pointer currently references.
TRAINED_CHECKPOINT_KEEP = max(0, int(os.environ.get("PVZ_TRAINED_CHECKPOINT_KEEP", "8")))


def prune_trained_checkpoints(run_dir: Path, keep: int, protected: set[Path]) -> list[str]:
    """Delete older `trained` snapshots, keeping the newest *keep*.

    Only the `trained` phase is touched -- `evaluated`, `initial` and `boundary`
    snapshots are milestones and stay. Anything in *protected* survives even if
    it is older than the newest *keep*.

    Filenames embed the update number and a nanosecond timestamp
    (``update_000042_trained_1790818381980869592.pt``), so plain lexicographic
    order is chronological order.

    Returns the names of the files that were removed. Unlink failures are
    swallowed: losing a checkpoint to a transient filesystem error must never
    take down a training run.
    """
    if keep < 0:
        return []
    snapshots = sorted(run_dir.glob("update_*_trained_*.pt"))
    if len(snapshots) <= keep:
        return []
    survivors = {path.resolve() for path in snapshots[len(snapshots) - keep:]} if keep else set()
    survivors |= {path.resolve() for path in protected}
    removed: list[str] = []
    for path in snapshots:
        if path.resolve() in survivors:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed.append(path.name)
    return removed


def capture_rng(assignments: random.Random) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "assignments": assignments.getstate()}


def restore_rng(saved: dict[str, Any], assignments: random.Random) -> None:
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch_cpu"].cpu())
    if saved["torch_cuda"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in saved["torch_cuda"]])
    assignments.setstate(saved["assignments"])


class ResourceMonitor:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.peak_tree_rss_bytes = 0
        self.min_available_bytes = _system_memory()[0]
        self.peak_swap_used_bytes = 0
        self.thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self.stop.is_set():
            processes = {}
            for directory in Path("/proc").iterdir():
                if not directory.name.isdigit():
                    continue
                try:
                    fields = dict(line.split(":", 1) for line in (directory / "status").read_text().splitlines())
                    processes[int(directory.name)] = (int(fields["PPid"]),
                                                     int(fields.get("VmRSS", "0 kB").split()[0]) * 1024)
                except (OSError, ValueError, KeyError):
                    pass
            tree = {os.getpid()}
            while True:
                children = {pid for pid, (parent, _) in processes.items() if parent in tree}
                if children <= tree:
                    break
                tree.update(children)
            rss = sum(processes.get(pid, (0, 0))[1] for pid in tree)
            self.peak_tree_rss_bytes = max(self.peak_tree_rss_bytes, rss)
            available, swap_used = _system_memory()
            self.min_available_bytes = min(self.min_available_bytes, available)
            self.peak_swap_used_bytes = max(self.peak_swap_used_bytes, swap_used)
            self.stop.wait(1)

    def snapshot(self) -> dict[str, int]:
        return {"peak_process_tree_rss_bytes": self.peak_tree_rss_bytes,
                "min_system_available_bytes": self.min_available_bytes,
                "peak_system_swap_used_bytes": self.peak_swap_used_bytes,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def _system_memory() -> tuple[int, int]:
    fields = {line.split(":")[0]: int(line.split()[1]) * 1024
              for line in Path("/proc/meminfo").read_text().splitlines()}
    return fields["MemAvailable"], fields["SwapTotal"] - fields["SwapFree"]


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = json.loads(path.read_text())
    expected = {"schema_version", "experiment_id", "purpose", "initialization_seed",
                "sampling_seed", "model", "reward", "ppo", "sampling", "budget",
                "evaluation", "runtime", "prerequisites"}
    if "initialization" in config:
        expected.add("initialization")
        validate_transfer(config["initialization"])
    if set(config) != expected or config["schema_version"] != RESEARCH_VERSION:
        raise ValueError("research config fields/schema do not match version 1")
    model_fields = {"layers", "width", "heads", "ff_width", "gru_layers", "gru_width",
                    "critic_width", "critic_layers", "input_flags"}
    if not model_fields <= set(config["model"]) <= model_fields | {"wait_mode", "wait_mask", "slow_memory", "short_history"}:
        raise ValueError("all model dimensions and input_flags must be explicit")
    model_architecture_version(config["model"])
    reward = config["reward"]
    if set(reward) != {"name", "gamma", "shaping_weight"}:
        raise ValueError("reward must explicitly define name, gamma and shaping_weight")
    if not 0 < reward["gamma"] <= 1 or reward["shaping_weight"] not in (0, 1):
        raise ValueError("invalid reward settings")
    expected_ppo = {"learning_rate", "ppo_epochs", "sequence_length", "minibatch_chunks",
                    "clip_epsilon", "gae_lambda", "value_coefficient", "entropy_coefficient",
                    "attention_backend", "target_kl"}
    if set(config["ppo"]) != expected_ppo:
        raise ValueError("all PPO settings must be explicit")
    ppo = config["ppo"]
    if (ppo["learning_rate"] <= 0 or ppo["ppo_epochs"] < 1 or ppo["sequence_length"] < 0
            or ppo["minibatch_chunks"] < 1 or not 0 < ppo["gae_lambda"] <= 1
            or not 0 < ppo["clip_epsilon"] < 1 or ppo["value_coefficient"] < 0
            or ppo["entropy_coefficient"] < 0 or ppo["target_kl"] <= 0):
        raise ValueError("invalid PPO settings")
    sampling = config["sampling"]
    sampling_fields = {"manifest", "task_ids", "method", "rollout_episodes"}
    if sampling.get("method") in course.COURSE_METHODS:
        sampling_fields.add("curriculum")
    if "task_mutation" in sampling:
        sampling_fields.add("task_mutation")
        if sampling.get("method") != "frontier_v1":
            raise ValueError("append_neighbors_v1 task mutation requires frontier_v1 sampling")
    if set(sampling) != sampling_fields:
        raise ValueError("all sampling settings must be explicit")
    if sampling["method"] not in {"balanced", "legacy_recent", *course.COURSE_METHODS} or sampling["rollout_episodes"] < 1:
        raise ValueError("unsupported sampling method or rollout size")
    if sampling["method"] in course.COURSE_METHODS:
        course.validate_settings(sampling["curriculum"], sampling["method"])
    train = json.loads(_path(sampling["manifest"]).read_text())["tasks"]
    selected = set(sampling["task_ids"])
    if selected:
        if selected - {task["task_id"] for task in train}:
            raise ValueError("unknown selected training task")
        train = [task for task in train if task["task_id"] in selected]
    evaluation = config["evaluation"]
    evaluation_fields = {"manifest", "modes", "decision_nodes"}
    if "idle_baseline" in evaluation:
        evaluation_fields.add("idle_baseline")
        if type(evaluation["idle_baseline"]) is not bool:
            raise ValueError("idle_baseline must be an explicit boolean")
    if set(evaluation) != evaluation_fields:
        raise ValueError("all evaluation settings must be explicit")
    if not evaluation["modes"] or set(evaluation["modes"]) - {"greedy", "sampled"}:
        raise ValueError("evaluation modes must be greedy and/or sampled")
    eval_tasks = json.loads(_path(evaluation["manifest"]).read_text())["tasks"]
    if "task_mutation" in sampling:
        mutation.validate_settings(sampling["task_mutation"], train, eval_tasks)
    for tasks in (train, eval_tasks):
        if not tasks or len(tasks) != len({task["task_id"] for task in tasks}):
            raise ValueError("empty or duplicate task list")
        for task in tasks:
            if not task["seeds"] or len(task["seeds"]) != len(set(task["seeds"])):
                raise ValueError("empty or duplicate environment seeds")
    if sampling["method"] in course.COURSE_METHODS:
        course.initial_state(train)
    budget_fields = {"decisions", "max_episodes"}
    if "boundary_mode" in config["budget"]:
        budget_fields.add("boundary_mode")
        if config["budget"]["boundary_mode"] != "exact_decisions_v1":
            raise ValueError("unsupported explicit decision budget boundary mode")
    if set(config["budget"]) != budget_fields or config["budget"]["decisions"] < 1:
        raise ValueError("budget must specify positive decisions and optional max_episodes")
    if config["budget"]["max_episodes"] is not None and config["budget"]["max_episodes"] < 1:
        raise ValueError("max_episodes must be null or positive")
    nodes = evaluation["decision_nodes"]
    if nodes != sorted(set(nodes)) or any(node <= 0 for node in nodes):
        raise ValueError("evaluation nodes must be sorted distinct positive decision counts")
    if config["budget"].get("boundary_mode") == "exact_decisions_v1":
        if (type(config["budget"]["decisions"]) is not int
                or any(type(node) is not int or node > config["budget"]["decisions"] for node in nodes)):
            raise ValueError("exact decision boundaries require integer nodes within the final budget")
    runtime = config["runtime"]
    if set(runtime) != {"workers", "worker_threads", "worker_device", "update_device", "max_actions",
                        "cudnn_tf32", "matmul_precision", "deterministic_algorithms",
                        "cublas_workspace_config"}:
        raise ValueError("all runtime settings must be explicit")
    if type(runtime["deterministic_algorithms"]) is not bool:
        raise ValueError("deterministic_algorithms must be an explicit boolean")
    if runtime["cublas_workspace_config"] not in (None, ":4096:8"):
        raise ValueError("unsupported explicit CUBLAS workspace configuration")
    if runtime["deterministic_algorithms"] and runtime["cublas_workspace_config"] != ":4096:8":
        raise ValueError("deterministic CUDA updates require the verified :4096:8 workspace")
    if runtime["cudnn_tf32"] is not False or runtime["matmul_precision"] != "highest":
        raise ValueError("research replay requires explicit FP32 CPU/CUDA precision")
    if runtime["update_device"] != "cuda" or runtime["worker_device"] != "cpu":
        raise ValueError("this protocol requires CUDA updates and CPU sampling")
    if min(runtime["workers"], runtime["worker_threads"], runtime["max_actions"]) < 1:
        raise ValueError("runtime counts must be positive")
    for prerequisite in config["prerequisites"]:
        gate = json.loads(_path(prerequisite).read_text())
        if gate.get("gate_result") != "pass":
            raise RuntimeError(f"research prerequisite failed: {prerequisite}")
        if ("allowed_experiment_ids" in gate
                and config["experiment_id"] not in gate["allowed_experiment_ids"]):
            raise RuntimeError(f"research prerequisite does not authorize this experiment: {prerequisite}")
        if ("simulator_sha256" in gate
                and gate["simulator_sha256"] != sha256_file(ROOT / "build/pvz-portable")):
            raise RuntimeError(f"research prerequisite simulator is stale: {prerequisite}")
        for relative, expected_hash in gate.get("required_fingerprints", {}).items():
            if sha256_file(_path(relative)) != expected_hash:
                raise RuntimeError(f"research prerequisite source is stale: {prerequisite}: {relative}")
    return config, train, eval_tasks


def experiment_identity_digest(config: dict[str, Any], fingerprints: dict[str, str]) -> str:
    """Hash semantic experiment settings, excluding only worker sizing knobs.

    Worker count and per-worker thread count affect throughput, not the data
    collected. Keep every other runtime field in the identity: in particular,
    ``max_actions`` changes where an episode is truncated.
    """
    identity_config = dict(config)
    identity_config["runtime"] = {
        key: value for key, value in config["runtime"].items()
        if key not in {"workers", "worker_threads"}
    }
    return canonical_digest({"config": identity_config, "fingerprints": fingerprints})


def evaluate(model: GameplayModelV1, tasks: list[dict[str, Any]], config: dict[str, Any],
             resource_dir: Path, output: Path, state: dict[str, Any]) -> dict[str, Any]:
    import train_pvz_ppo_task_family as family
    import t4_capability_profile as profile
    started = time.monotonic()
    idle = None
    if config["evaluation"].get("idle_baseline", False):
        from pvz_progress_metrics import idle_baseline
        idle = idle_baseline(tasks, resource_dir, output, state["experiment_identity"],
                             config["runtime"]["max_actions"])
    jobs = {}
    labels = []
    for mode in config["evaluation"]["modes"]:
        for task in tasks:
            for seed in task["seeds"]:
                labels.append((mode, task["task_id"]))
                jobs[len(jobs)] = {"task": task, "seed": seed, "deterministic": mode == "greedy",
                                   "max_actions": config["runtime"]["max_actions"],
                                   "allow_truncation": True, "action_seed": seed + 170_000}
    weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    runtime = config["runtime"]
    records = family._run_evaluation_jobs(jobs, weights, resource_dir, runtime["workers"],
                                         runtime["worker_threads"], "cpu", model.config)
    grouped = {mode: {task["task_id"]: [] for task in tasks}
               for mode in config["evaluation"]["modes"]}
    for label, record in zip(labels, records, strict=True):
        mode, task_id = label
        grouped[mode][task_id].append(record)
    summary = {}
    for mode, per_task in grouped.items():
        flat = [record for rows in per_task.values() for record in rows]
        summary[mode] = {"overall": profile.summarize_episodes(flat),
                         "per_task": {key: profile.summarize_episodes(rows)
                                      for key, rows in per_task.items()},
                         "truncated": sum(row["truncated"] for row in flat)}
        if idle is not None:
            from pvz_progress_metrics import progress_summary, paired_idle_summary
            summary[mode]["progress_groups"] = progress_summary(tasks, per_task)
            summary[mode]["paired_idle"] = paired_idle_summary(tasks, per_task, idle)
    path = output / "evaluations" / f"update_{state['updates']:06d}.json.gz"
    atomic_json(path, {"counters": state["counters"], "model_config": model.config,
                       "experiment_id": config["experiment_id"], "seed_results": grouped}, compressed=True)
    record = {"counters": dict(state["counters"]), "updates": state["updates"], "summary": summary,
            "raw_seed_results_path": str(path.relative_to(output)),
            "seconds": time.monotonic() - started}
    if idle is not None:
        record["idle_control_path"] = "evaluations/idle_control.json.gz"
        record["overall_scope"] = "Monitoring aggregate only; use progress_groups and matched idle gains for strategy learning."
    return record


def _assign(tasks: list[dict[str, Any]], job_ids: list[int], rng: random.Random,
            recent: dict[str, list[bool]], method: str,
            weights: list[float] | None = None) -> dict[int, dict[str, Any]]:
    if weights is None:
        if method in course.COURSE_METHODS:
            raise ValueError("curriculum weights require the complete saved course state")
        weights = [1.0 if method == "balanced" else
                   2.0 - (sum(recent[task["task_id"]]) / len(recent[task["task_id"]])
                          if recent[task["task_id"]] else 0.0) for task in tasks]
    result = {}
    for job_id in job_ids:
        task = rng.choices(tasks, weights=weights, k=1)[0]
        result[job_id] = {"task": task, "task_seed": rng.choice(task["seeds"]),
                          "action_seed": rng.randrange(1, 2**31)}
    return result


def decision_quotas(job_ids: list[int], remaining: int, max_actions: int) -> dict[int, int]:
    """Reserve every possible action before collection; never discard overflow.

    A normal terminal can leave unused reservations. The next update refills the
    remainder with fresh episodes; truncated histories use the existing value
    bootstrap and do not become curriculum failures. Quotas are saved in each
    assignment/collection fingerprint, so incomplete batches replay the same plan.
    """
    if (not job_ids or len(set(job_ids)) != len(job_ids)
            or type(remaining) is not int or remaining < len(job_ids)
            or type(max_actions) is not int or max_actions < 1):
        raise ValueError("decision reservations require positive unique jobs and capacity")
    quotient, remainder = divmod(remaining, len(job_ids))
    return {job_id:min(max_actions, quotient + (index < remainder))
            for index,job_id in enumerate(job_ids)}


def _trajectory_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    immediate, zero_ticks, shaping_abs = 0, 0, []
    returns, values = [], []
    for episode in episodes:
        previous_plant = None
        mc = float(episode.get("bootstrap_value", 0.0))
        terminal_outcome = (episode["transitions"][-1].get("terminal_outcome", 0.0)
                            if episode["transitions"] else 0.0)
        for step in reversed(episode["transitions"]):
            mc = step["reward"] + step["discount"] * mc
            returns.append(mc + terminal_outcome)
            values.append(step["value"])
        for step in episode["transitions"]:
            action = step["action"]
            cell = (action.get("row"), action.get("col"))
            immediate += action["type"] == "shovel" and cell == previous_plant
            previous_plant = cell if action["type"] == "plant" else None
            counts[action["type"]] += 1
            zero_ticks += step["action_duration_ticks"] == 0
            shaping_abs.append(abs(step["shaping_reward"]))
    var = float(np.var(returns))
    stats = {"action_counts": dict(counts), "zero_tick_actions": zero_ticks,
            "immediate_plant_shovels": immediate,
            "immediate_shovel_fraction_of_plants": immediate / max(1, counts["plant"]),
            "mean_abs_shaping_reward": float(np.mean(shaping_abs)),
            "mc_value_mse_before_update": float(np.mean((np.array(returns) - values) ** 2)),
            "mc_value_explained_variance_before_update":
                1 - float(np.var(np.array(returns) - values)) / var if var > 1e-12 else None,
            "won": sum(episode["won"] for episode in episodes),
            "truncated": sum(episode["truncated"] for episode in episodes),
            "task_counts": dict(Counter(episode["task_id"] for episode in episodes))}
    if any("wait_result" in step for episode in episodes for step in episode["transitions"]):
        stats["wait_summary"] = summarize_wait_records([
            {"action": step["action"], "wait_result": step["wait_result"],
             "actual_ticks": step["action_duration_ticks"]}
            for episode in episodes for step in episode["transitions"] if step["action"]["type"] == "wait"])
    return stats


def run_experiment(args: Any) -> None:
    invocation_started = time.monotonic()
    import train_pvz_ppo_task_family as family
    import t4_capability_profile as profile
    config, tasks, eval_tasks = load_config(args.experiment_config.resolve())
    if args.ignore_stage0_gate:
        raise ValueError("research configs do not support legacy stage-0 gate overrides")
    if args.init_checkpoint:
        configured_source = config.get("initialization", {}).get("source_checkpoint")
        if configured_source is None or _path(configured_source) != args.init_checkpoint.expanduser().resolve():
            raise ValueError("--init-checkpoint must match the source frozen in the research config")
    if any(value is not None for value in (args.workers, args.rollout_threads, args.rollout_device)):
        raise ValueError("research runtime settings come from the frozen experiment config")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".execution.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    workspace = config["runtime"]["cublas_workspace_config"]
    existing_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing_workspace is not None and existing_workspace != workspace:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG environment differs from frozen runtime config")
    if workspace is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(config["runtime"]["deterministic_algorithms"], warn_only=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; research updates must not silently use CPU")
    # cuDNN's default TF32 RNN path differs from CPU collection by ~1.7e-4
    # in log probability. FP32 restores the frozen-policy ~1e-6 agreement.
    torch.backends.cudnn.allow_tf32 = config["runtime"]["cudnn_tf32"]
    torch.set_float32_matmul_precision(config["runtime"]["matmul_precision"])
    configure_torch_threads(1)
    device = torch.device("cuda")
    resource_dir = args.resource_dir.expanduser().resolve()
    source_paths = [ROOT / "python" / name for name in (
        "pvz_agent_model.py", "pvz_research.py", "train_pvz_ppo.py", "train_pvz_ppo_task_family.py",
        "pvz_env.py", "pvz_seed_jobs.py", "pvz_common.py", "pvz_value.py",
        "pvz_observation_features.py", "pvz_curriculum.py", "pvz_event_env.py", "pvz_wait_events.py")]
    source_paths.append(ROOT / "scripts/t4_capability_profile.py")
    if config["evaluation"].get("idle_baseline", False):
        source_paths.append(ROOT / "python/pvz_progress_metrics.py")
    if "slow_memory" in config["model"]:
        source_paths.append(ROOT / "python/pvz_dual_memory.py")
    if "short_history" in config["model"]:
        source_paths.append(ROOT / "python/pvz_short_memory.py")
    if "task_mutation" in config["sampling"]:
        source_paths.append(ROOT / "python/pvz_task_mutation.py")
    if "initialization" in config:
        source_paths.append(ROOT / "python/pvz_initialization.py")
    fingerprints = {str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths}
    fingerprints.update({"simulator": sha256_file(ROOT / "build/pvz-portable"),
                         "main.pak": sha256_file(resource_dir / "main.pak"),
                         "properties/partner.xml": sha256_file(resource_dir / "properties/partner.xml"),
                         "train_manifest": sha256_file(_path(config["sampling"]["manifest"])),
                         "evaluation_manifest": sha256_file(_path(config["evaluation"]["manifest"]))})
    if "initialization" in config:
        # A complete stage resume needs only its own checkpoint, even when the
        # parent file lives on another machine or has not been downloaded.
        fingerprints["weight_transfer_source_checkpoint"] = config["initialization"]["source_sha256"]
    identity = experiment_identity_digest(config, fingerprints)
    seed = config["initialization_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = GameplayModelV1(config["model"]).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["ppo"]["learning_rate"])
    rng = random.Random(config["sampling_seed"])
    revision, dirty = git_metadata(ROOT)
    pointer = output / "resume.json"
    if args.resume:
        saved_pointer = json.loads(pointer.read_text())
        checkpoint_path = output / saved_pointer["checkpoint"]
        if sha256_file(checkpoint_path) != saved_pointer["sha256"]:
            raise ValueError("resume checkpoint fingerprint mismatch")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["experiment_identity"] != identity:
            raise ValueError("resume config, source, simulator, resources or manifests changed")
        if (checkpoint["config"] != model.config
                or checkpoint["model_architecture_version"] != model_architecture_version(model.config)):
            raise ValueError("resume model config/architecture differs")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        state = checkpoint["training_state"]
        restore_rng(checkpoint["rng_state"], rng)
        state["invocations"].append({"kind": "complete_resume", "started_at": _now(), "commit": revision})
        state["status"] = "running"
    else:
        if pointer.exists() or (output / "training_state.json").exists():
            raise RuntimeError("experiment directory already has state; use --resume for this candidate")
        initialization = None
        if "initialization" in config:
            initialization = transfer_weights(model, config["initialization"],
                                               _path(config["initialization"]["source_checkpoint"]))
        state = {"schema_version": RESEARCH_VERSION, "experiment_id": config["experiment_id"],
                 "experiment_identity": identity, "updates": 0,
                 "counters": {"episodes": 0, "decisions": 0, "ticks": 0}, "wall_seconds": 0.0,
                 "phase": "initial_evaluation", "evaluation_cursor": 0,
                 "recent_passes": {task["task_id"]: [] for task in tasks},
                 "learning_curve": [], "update_history": [],
                 "invocations": [{"kind": "weights_transfer_v1" if initialization else "random_initialization", "seed": seed,
                                  "started_at": _now(), "commit": revision}],
                 "initial_state_sha256": profile._state_sha256(model.state_dict()),
                 "status": "running"}
        if initialization is not None:
            state["initialization_provenance"] = initialization
        atomic_json(output / "experiment_config.json", config)
        atomic_json(output / "provenance.json", {"commit": revision, "worktree_dirty": dirty,
                    "fingerprints": fingerprints, "model_config": model.config,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "experiment_identity": identity,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    **({"initialization": initialization} if initialization is not None else {})})
    base_tasks = tasks
    mutation_settings = config["sampling"].get("task_mutation")
    if mutation_settings is not None:
        if not args.resume:
            state["task_mutation_state"] = mutation.initial_state(base_tasks)
        mutation.validate_state(state["task_mutation_state"], mutation_settings, base_tasks, eval_tasks)
        tasks = state["task_mutation_state"]["pool"]
    if config["sampling"]["method"] in course.COURSE_METHODS:
        if not args.resume:
            state["curriculum_state"] = course.initial_state(tasks)
        course.validate_state(state["curriculum_state"], tasks, config["sampling"]["curriculum"])
    stop_requested = threading.Event()
    old_handlers = {sig: signal.signal(sig, lambda *_: stop_requested.set())
                    for sig in (signal.SIGTERM, signal.SIGINT)}
    monitor = ResourceMonitor()
    torch.cuda.reset_peak_memory_stats()
    monitor.thread.start()
    previous_wall = invocation_started

    def save(phase: str) -> None:
        nonlocal previous_wall
        now = time.monotonic()
        state["wall_seconds"] += now - previous_wall
        state["elapsed_since_first_start_seconds"] = (
            datetime.now(timezone.utc) - datetime.fromisoformat(state["invocations"][0]["started_at"])).total_seconds()
        previous_wall = now
        resources = monitor.snapshot()
        for key, value in state.get("resources", {}).items():
            resources[key] = min(value, resources[key]) if key.startswith("min_") else max(value, resources[key])
        state["resources"] = resources
        # Every save gets a fresh immutable filename, so a resume at the same
        # update cannot overwrite earlier crash evidence. Older `trained`
        # snapshots are pruned right after the pointer moves -- see
        # TRAINED_CHECKPOINT_KEEP for what survives and why.
        path = output / "runs/run_1" / f"update_{state['updates']:06d}_{phase}_{time.time_ns()}.pt"
        state["checkpoint"] = str(path.relative_to(output))
        checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                      "optimizer_state_dict": optimizer.state_dict(), "rng_state": capture_rng(rng),
                      "model_architecture_version": model_architecture_version(model.config),
                      "config": model.config, "experiment_config": config,
                      "experiment_identity": identity, "training_state": state,
                      "value_semantics": VALUE_SEMANTICS, "research_version": RESEARCH_VERSION,
                      "provenance": {"protocol_version": ENV_PROTOCOL_VERSION,
                                     "observation_version": OBSERVATION_VERSION, "task_version": TASK_VERSION,
                                     "commit": revision, "fingerprints": fingerprints}}
        atomic_write(path, lambda temporary: torch.save(checkpoint, temporary))
        atomic_json(pointer, {"checkpoint": str(path.relative_to(output)), "sha256": sha256_file(path)})
        if phase == "trained":
            protected = {path}
            for recorded in [state.get("checkpoint")] + [
                    run.get("checkpoint") for run in state.get("runs", [])]:
                if isinstance(recorded, str):
                    protected.add(output / recorded)
            removed = prune_trained_checkpoints(path.parent, TRAINED_CHECKPOINT_KEEP, protected)
            if removed:
                pruning = state.setdefault("checkpoint_pruning", {
                    "keep": TRAINED_CHECKPOINT_KEEP, "removed_total": 0, "last_removed": []})
                pruning["removed_total"] += len(removed)
                pruning["last_removed"] = removed
                print(f"{config['experiment_id']} pruned {len(removed)} old trained checkpoint(s), "
                      f"keeping the newest {TRAINED_CHECKPOINT_KEEP}", flush=True)
        atomic_json(output / "training_state.json", state)
        atomic_json(output / "learning_curve.json", state["learning_curve"])
        if mutation_settings is not None:
            atomic_json(output / "generated_training_pool.json", state["task_mutation_state"])

    def do_evaluation() -> None:
        record = evaluate(model, eval_tasks, config, resource_dir, output, state)
        state["learning_curve"].append(record)
        state["phase"] = "ready"
        save("evaluated")
        if config["evaluation"].get("idle_baseline", False):
            print(f"{config['experiment_id']} evaluation update={state['updates']} "
                  + " ".join(f"{mode}/{group}={rows['policy']['pass_rate']:.4f} "
                             f"idle={rows['idle']['pass_rate']:.4f} gain={rows['net_win_gain']:+.4f}"
                             for mode, results in record["summary"].items()
                             for group, rows in results["paired_idle"]["by_group"].items()), flush=True)
        else:
            print(f"{config['experiment_id']} evaluation update={state['updates']} "
                  + " ".join(f"{mode}={rows['overall']['pass_rate']:.4f}"
                             for mode, rows in record["summary"].items()), flush=True)

    try:
        save("resumed" if args.resume else "initial")
        if state["phase"] in ("initial_evaluation", "pending_evaluation"):
            do_evaluation()
        invocation_updates = 0
        nodes = config["evaluation"]["decision_nodes"]
        while state["counters"]["decisions"] < config["budget"]["decisions"]:
            counters = state["counters"]
            episode_limit = config["budget"]["max_episodes"]
            if episode_limit is not None and counters["episodes"] >= episode_limit:
                break
            if stop_requested.is_set() or (args.stop_after_updates is not None
                                           and invocation_updates >= args.stop_after_updates):
                break
            if mutation_settings is not None:
                additions = mutation.advance(state["task_mutation_state"], mutation_settings,
                    base_tasks, eval_tasks, state["curriculum_state"],
                    config["sampling"]["curriculum"], counters["decisions"])
                if additions:
                    course.append_tasks(state["curriculum_state"], additions)
                    state["recent_passes"].update({task["task_id"]: [] for task in additions})
                    print(f"{config['experiment_id']} appended {len(additions)} mutation tasks; "
                          f"pool={len(tasks)} at decisions={counters['decisions']}", flush=True)
            batch_size = config["sampling"]["rollout_episodes"]
            if episode_limit is not None:
                batch_size = min(batch_size, episode_limit - counters["episodes"])
            next_node = min([node for node in nodes if node > counters["decisions"]]
                            + [config["budget"]["decisions"]])
            if counters["episodes"]:
                mean_decisions = counters["decisions"] / counters["episodes"]
                batch_size = min(batch_size, max(1, math.ceil((next_node - counters["decisions"]) / mean_decisions)))
            exact_decisions = config["budget"].get("boundary_mode") == "exact_decisions_v1"
            if exact_decisions:
                batch_size = min(batch_size, next_node - counters["decisions"])
            job_ids = list(range(counters["episodes"], counters["episodes"] + batch_size))
            sampling = config["sampling"]
            sampling_weights, course_snapshot = None, None
            if sampling["method"] in course.COURSE_METHODS:
                sampling_weights, course_snapshot = course.probabilities(
                    tasks, state["curriculum_state"], sampling["curriculum"], sampling["method"])
            assignments = _assign(tasks, job_ids, rng, state["recent_passes"], sampling["method"], sampling_weights)
            if exact_decisions:
                quotas = decision_quotas(job_ids, next_node - counters["decisions"],
                                        config["runtime"]["max_actions"])
                for job_id, quota in quotas.items():
                    assignments[job_id]["decision_quota"] = quota
            weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            metadata = {"research_version": RESEARCH_VERSION, "experiment_identity": identity,
                        "update": state["updates"] + 1,
                        "model_state_sha256": profile._state_sha256(weights),
                        "assignments": assignments, "feature_storage_dtype": "float16"}
            shard_dir = seed_job_directory(output / "runs/run_1", f"update_{state['updates'] + 1:06d}", metadata)
            runtime = config["runtime"]
            label = f"{config['experiment_id']} update {state['updates'] + 1}"
            started = time.monotonic()
            episodes = run_seed_jobs(job_ids, shard_dir, metadata, family._rollout_worker,
                                     workers=runtime["workers"], initializer=family._init_worker,
                                     initargs=(str(resource_dir), weights, assignments, runtime["max_actions"],
                                               runtime["worker_threads"], "cpu", model.config,
                                               config["reward"], True), label=label)
            rollout_seconds = time.monotonic() - started
            if exact_decisions:
                for job_id, episode in zip(job_ids, episodes, strict=True):
                    if (episode["seed"] != job_id or episode.get("decision_quota") != quotas[job_id]
                            or not 1 <= len(episode["transitions"]) <= quotas[job_id]):
                        raise RuntimeError("collected episode violated its frozen decision reservation")
                if counters["decisions"] + sum(len(e["transitions"]) for e in episodes) > next_node:
                    raise RuntimeError("actual decisions exceed the evaluation/final budget boundary")
            stats = _trajectory_stats(episodes)
            add_advantages(episodes, config["ppo"]["gae_lambda"], config["reward"]["gamma"])
            started = time.monotonic()
            ppo = config["ppo"]
            losses = train_update(model, episodes, optimizer, device, ppo["ppo_epochs"],
                                  ppo["sequence_length"], ppo["clip_epsilon"], ppo["value_coefficient"],
                                  ppo["entropy_coefficient"], ppo["minibatch_chunks"],
                                  ppo["attention_backend"], label, ppo["target_kl"])
            update_seconds = time.monotonic() - started
            counters["episodes"] += len(episodes)
            counters["decisions"] += sum(len(episode["transitions"]) for episode in episodes)
            counters["ticks"] += sum(step["action_duration_ticks"] for episode in episodes
                                     for step in episode["transitions"])
            state["updates"] += 1
            invocation_updates += 1
            for episode in episodes:
                if exact_decisions and episode["truncated"]:
                    continue
                recent = state["recent_passes"][episode["task_id"]]
                recent.append(episode["won"])
                del recent[:-64]
            if sampling["method"] in course.COURSE_METHODS:
                course.observe(state["curriculum_state"], episodes, sampling["curriculum"])
            state["update_history"].append({"update": state["updates"], "counters": dict(counters),
                        "losses": losses, "trajectory_stats": stats,
                        "rollout_seconds": rollout_seconds, "ppo_seconds": update_seconds,
                        "episode_digests": [episode_digest(episode) for episode in episodes],
                        "shard_directory": str(shard_dir.relative_to(output))})
            if course_snapshot is not None:
                state["update_history"][-1]["curriculum_sampling"] = course_snapshot
            if exact_decisions:
                state["update_history"][-1]["decision_budget"] = {
                    "boundary_mode":"exact_decisions_v1", "boundary":next_node,
                    "reserved_decisions":sum(quotas.values()), "quotas":quotas,
                    "actual_decisions":sum(len(e["transitions"]) for e in episodes),
                    "budget_boundary_truncations":sum(e.get("truncation_reason") == "decision_budget_boundary"
                                                       for e in episodes)}
            due = state["evaluation_cursor"] < len(nodes) and counters["decisions"] >= nodes[state["evaluation_cursor"]]
            if due:
                while state["evaluation_cursor"] < len(nodes) and counters["decisions"] >= nodes[state["evaluation_cursor"]]:
                    state["evaluation_cursor"] += 1
            state["phase"] = "pending_evaluation" if due else "ready"
            save("trained")
            print(f"{label} episodes={counters['episodes']} decisions={counters['decisions']} "
                  f"ticks={counters['ticks']} wall={state['wall_seconds']:.1f}s "
                  f"policy_loss={losses['policy_loss']:.6f} value_loss={losses['value_loss']:.6f} "
                  f"wins={stats['won']}/{len(episodes)} KL={losses['approx_kl']:.5f}", flush=True)
            if due:
                do_evaluation()
        counters = state["counters"]
        budget_done = (counters["decisions"] >= config["budget"]["decisions"] or
                       (config["budget"]["max_episodes"] is not None
                        and counters["episodes"] >= config["budget"]["max_episodes"]))
        if budget_done:
            if state["learning_curve"][-1]["updates"] != state["updates"]:
                state["phase"] = "pending_evaluation"
                save("trained")
                do_evaluation()
            state["status"] = "budget_complete"
        else:
            state["status"] = "update_boundary_stop"
        save("boundary")
    except BaseException as error:
        # Do not checkpoint partly updated weights. The authoritative pointer still
        # names a complete update and all completed shards remain untouched.
        atomic_json(output / f"failure_{time.time_ns()}.json",
                    {"at": _now(), "type": type(error).__name__, "error": str(error),
                     "last_complete_checkpoint": json.loads(pointer.read_text()) if pointer.exists() else None,
                     "resources": monitor.snapshot()})
        raise
    finally:
        monitor.stop.set()
        monitor.thread.join(timeout=2)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        lock.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
