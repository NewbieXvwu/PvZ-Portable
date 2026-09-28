"""Evaluate simulator search and GameplayModel checkpoints on frozen seed sets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from multiprocessing.util import Finalize
from typing import Any

import torch

from pvz_agent_model import (
    GameplayModelV1,
    MODEL_ARCHITECTURE_VERSION,
    configure_torch_threads,
    predict_action,
)
from pvz_common import (
    ENV_PROTOCOL_VERSION,
    OBSERVATION_VERSION,
    TASK_VERSION,
    git_metadata,
    sha256_file,
)
from pvz_env import PvZEnv, training_task
from pvz_search import SearchTeacher
from pvz_search_diagnostics import visible_state_key
from pvz_search_value import SearchValueModel, load_search_value
from pvz_seed_sets import DEFAULT_DEV_SEEDS, DEFAULT_TEST_SEEDS, read_seed_set
from pvz_seed_jobs import atomic_json, run_seed_jobs, seed_job_directory
from pvz_training_artifacts import checkpoint_metadata, task_signature
from pvz_value import SEARCH_LABEL_VERSION, VALUE_SEMANTICS

LEVEL = 7
DECK = (0, 1, 2, 3, 4, 5)
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_ACTIONS = 2000

_BENCH_ENV: PvZEnv | None = None
_BENCH_MODEL: GameplayModelV1 | None = None
_BENCH_VALUE_MODEL: SearchValueModel | None = None
_BENCH_SEARCHER: SearchTeacher | None = None
_BENCH_SETTINGS: dict[str, Any] = {}
_BENCH_FINALIZER: Finalize | None = None


def _close_benchmark_worker() -> None:
    if _BENCH_ENV is not None:
        _BENCH_ENV.close()


def _initialize_benchmark_worker(
    resource_dir: str,
    settings: dict[str, Any],
    policy_kind: str,
    model_state: dict[str, torch.Tensor] | None,
) -> None:
    global _BENCH_ENV, _BENCH_MODEL, _BENCH_VALUE_MODEL, _BENCH_SEARCHER
    global _BENCH_SETTINGS, _BENCH_FINALIZER
    configure_torch_threads(settings["collection_threads"])
    _BENCH_ENV = PvZEnv(resource_dir)
    _BENCH_MODEL = None
    _BENCH_VALUE_MODEL = None
    _BENCH_SEARCHER = None
    if policy_kind == "search_teacher":
        if model_state is None:
            raise ValueError("search-teacher worker needs value-model weights")
        _BENCH_VALUE_MODEL = SearchValueModel()
        _BENCH_VALUE_MODEL.load_state_dict(model_state)
        _BENCH_VALUE_MODEL.eval()
        _BENCH_SEARCHER = SearchTeacher(
            _BENCH_ENV,
            value_model=_BENCH_VALUE_MODEL,
            beam_width=settings["search_width"],
            candidate_limit=settings["search_candidates"],
            horizon_ticks=settings["search_horizon_ticks"],
            simulation_budget=settings["search_simulation_budget"],
            max_decisions=settings["search_max_decisions"],
        )
    else:
        if model_state is None:
            raise ValueError("policy worker needs model weights")
        _BENCH_MODEL = GameplayModelV1()
        _BENCH_MODEL.load_state_dict(model_state)
        if settings["no_relation_bias"]:
            for layer in _BENCH_MODEL.encoder:
                layer.attention.relation_bias_enabled = False
        _BENCH_MODEL.eval()
    _BENCH_SETTINGS = settings
    _BENCH_FINALIZER = Finalize(None, _close_benchmark_worker, exitpriority=10)


def _benchmark_seed_worker(seed: int) -> dict[str, Any]:
    if _BENCH_ENV is None:
        raise RuntimeError("benchmark worker was not initialized")
    settings = _BENCH_SETTINGS
    samples: list[dict[str, Any]] = []
    record = run_episode(
        _BENCH_ENV,
        seed,
        _BENCH_MODEL,
        _BENCH_SEARCHER,
        settings["clear_hidden"],
        settings["zombie_count_multiplier"],
        Path(settings["replay_dir"]),
        settings["policy_label"],
        settings["level"],
        tuple(settings["deck"]),
        settings["max_actions"],
        samples if settings["collect_label_samples"] else None,
    )
    episode = _BENCH_ENV.episode or {}
    resources = {
        "resource_sha256": episode.get("resource_sha256"),
        "properties_partner_sha256": episode.get("properties_partner_sha256"),
    }
    worker_rss_bytes = None
    if _BENCH_ENV._process is not None:
        try:
            rss_kib = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(_BENCH_ENV._process.pid)],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            worker_rss_bytes = int(rss_kib) * 1024
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    return {
        "seed": seed,
        "record": record,
        "label_samples": samples,
        "resources": resources,
        "worker_rss_bytes": worker_rss_bytes,
    }


def _cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode("utf-8"))
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_model(path: Path, device: torch.device) -> tuple[GameplayModelV1, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    provenance = checkpoint["provenance"]
    if (checkpoint["model_architecture_version"] != MODEL_ARCHITECTURE_VERSION
            or checkpoint.get("value_semantics") != VALUE_SEMANTICS
            or provenance.get("search_label_version") != SEARCH_LABEL_VERSION
            or provenance.get("protocol_version") != ENV_PROTOCOL_VERSION
            or provenance["observation_version"] != OBSERVATION_VERSION
            or provenance["task_version"] != TASK_VERSION):
        raise ValueError("checkpoint does not match the current model/search semantics")
    model = GameplayModelV1().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def run_episode(
    env: PvZEnv,
    seed: int,
    model: GameplayModelV1 | None,
    searcher: SearchTeacher | None,
    clear_hidden: bool,
    zombie_count_multiplier: float,
    replay_dir: Path,
    policy_label: str,
    level: int,
    deck: tuple[int, ...],
    max_actions: int,
    label_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    task = training_task(seed, level, zombie_count_multiplier)
    reset_started = time.perf_counter()
    observation, _ = env.reset(deck=deck, task=task)
    reset_seconds = time.perf_counter() - reset_started
    hidden = None
    previous_action = None
    delta_ticks = 0
    events: dict[str, Any] = {}
    totals = Counter()
    actions = 0
    ticks_advanced = 0
    search_simulations = 0
    search_margins: list[float] = []
    search_entropies: list[float] = []
    search_candidate_counts: list[int] = []
    search_elapsed: list[int] = []
    search_screening_simulations = 0
    search_depth_simulations = 0
    search_root_actions: list[int] = []
    search_effective_depth: list[int] = []
    last_label_tick: int | None = None
    started = time.perf_counter()
    while not observation["terminal"] and actions < max_actions:
        if searcher is not None:
            advice = searcher.advice(observation)
            action = advice.action
            if label_samples is not None and (last_label_tick is None or
                                               observation["tick"] - last_label_tick >= 300):
                label_samples.append({
                    "seed": seed,
                    "tick": observation["tick"],
                    "visible_state_key": visible_state_key(observation),
                    "action": advice.action,
                    "policy": [{"action": candidate, "probability": probability}
                               for (candidate, _), probability in zip(advice.candidates, advice.search_policy)],
                })
                last_label_tick = observation["tick"]
            search_simulations += advice.simulation_count
            search_screening_simulations += advice.screening_simulations
            search_depth_simulations += advice.depth_simulations
            search_root_actions.append(advice.root_actions_generated)
            search_effective_depth.append(advice.effective_depth_budget)
            if advice.best_second_margin is not None:
                search_margins.append(advice.best_second_margin)
            if advice.search_policy:
                search_entropies.append(-sum(p * math.log(max(p, 1e-12)) for p in advice.search_policy))
                search_candidate_counts.append(len(advice.search_policy))
            search_elapsed.append(advice.search_elapsed_ticks)
        else:
            if model is None:
                raise RuntimeError("benchmark policy has neither model nor searcher")
            with torch.inference_mode():
                action, hidden, _ = predict_action(
                    model,
                    observation,
                    None if clear_hidden else hidden,
                    previous_action,
                    delta_ticks,
                    events,
                )
        observation, _, done, _, info = env.step(action)
        if not info.get("ok"):
            raise RuntimeError(f"illegal action on seed {seed}: {action}")
        for key, value in info["events"].items():
            if isinstance(value, (int, float)):
                totals[key] += value
        previous_action = action
        delta_ticks = info["ticks_advanced"]
        ticks_advanced += delta_ticks
        events = info["events"]
        actions += 1
        if done:
            break
    if not observation["terminal"]:
        raise RuntimeError(f"benchmark episode exceeded {max_actions} decisions on seed {seed}")
    safe_label = "".join(character if character.isalnum() or character in "-_" else "_" for character in policy_label)
    replay_id = f"{safe_label}_seed_{seed}.jsonl.gz"
    env.save_replay(replay_dir / replay_id)
    elapsed_seconds = max(time.perf_counter() - started, 1e-9)
    record = {
        "seed": seed,
        "replay_id": replay_id,
        "won": observation["result"] == 1,
        "result": observation["result"],
        "terminal": bool(observation["terminal"]),
        "wave": observation["wave"],
        "wave_count": observation["wave_count"],
        "tick": observation["tick"],
        "actions": actions,
        "ticks_advanced": ticks_advanced,
        "reset_seconds": round(reset_seconds, 4),
        "ticks_per_second": round(ticks_advanced / elapsed_seconds, 2),
        "actions_per_second": round(actions / elapsed_seconds, 2),
        "plants_eaten": int(totals["plants_eaten"]),
        "mower_triggers": int(totals["mower_triggered"]),
        "seconds": round(elapsed_seconds, 4),
    }
    if searcher is not None:
        record.update({
            "search_simulations": search_simulations,
            "search_mean_simulations_per_decision": search_simulations / max(actions, 1),
            "search_mean_margin": sum(search_margins) / max(len(search_margins), 1),
            "search_mean_entropy": sum(search_entropies) / max(len(search_entropies), 1),
            "search_mean_candidate_count": sum(search_candidate_counts) / max(len(search_candidate_counts), 1),
            "search_mean_elapsed_ticks": sum(search_elapsed) / max(len(search_elapsed), 1),
            # The screening/depth split, so a budget sweep can be read as depth budget
            # rather than as the raw number handed to SearchTeacher.
            "search_screening_simulations": search_screening_simulations,
            "search_depth_simulations": search_depth_simulations,
            "search_mean_root_actions": sum(search_root_actions) / max(len(search_root_actions), 1),
            "search_mean_effective_depth_budget": sum(search_effective_depth) / max(len(search_effective_depth), 1),
        })
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        raise ValueError("cannot summarize an empty episode set")
    wins = sum(record["won"] for record in records)
    rate = wins / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    margin = z * ((rate * (1 - rate) / count + z * z / (4 * count * count)) ** 0.5) / denominator
    losses = [record for record in records if not record["won"]]

    def mean(key: str) -> float:
        return sum(record[key] for record in records) / count

    summary = {
        "count": count,
        "wins": wins,
        "win_rate": rate,
        "wilson_95": [max(0.0, center - margin), min(1.0, center + margin)],
        "failure_wave_distribution": dict(sorted(
            Counter(str(record["wave"]) for record in losses).items(), key=lambda item: int(item[0])
        )),
        "mean_actions": mean("actions"),
        "mean_plants_eaten": mean("plants_eaten"),
        "mean_ticks_per_second": mean("ticks_per_second"),
        "mean_actions_per_second": mean("actions_per_second"),
        "mean_reset_seconds": mean("reset_seconds"),
        "plants_eaten_total": sum(record["plants_eaten"] for record in records),
        "mower_triggers_total": sum(record["mower_triggers"] for record in records),
        "mean_mower_triggers": mean("mower_triggers"),
        "wall_seconds_total": sum(record["seconds"] for record in records),
        "wall_seconds_mean": mean("seconds"),
    }
    for key in (
        "search_mean_simulations_per_decision",
        "search_mean_margin",
        "search_mean_entropy",
        "search_mean_candidate_count",
        "search_mean_elapsed_ticks",
        "search_mean_root_actions",
        "search_mean_effective_depth_budget",
    ):
        if key in records[0]:
            summary[key] = mean(key)
    if "search_simulations" in records[0]:
        summary["search_simulations_total"] = sum(record["search_simulations"] for record in records)
    if "search_screening_simulations" in records[0]:
        summary["search_screening_simulations_total"] = sum(
            record["search_screening_simulations"] for record in records)
        summary["search_depth_simulations_total"] = sum(
            record["search_depth_simulations"] for record in records)
    return summary


def parse_checkpoint(value: str) -> tuple[str, Path]:
    label, separator, filename = value.partition("=")
    if not separator or not label or not filename:
        raise argparse.ArgumentTypeError("checkpoint must use LABEL=PATH")
    return label, Path(filename).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-dir", default=os.environ.get("PVZ_RESOURCE_DIR"))
    parser.add_argument("--seeds", type=Path)
    parser.add_argument("--final-test", action="store_true")
    parser.add_argument("--search-label-diagnostics", type=Path)
    parser.add_argument("--level", type=int, default=LEVEL)
    parser.add_argument("--deck", type=lambda value: tuple(map(int, value.split(","))), default=DECK)
    parser.add_argument("--workers", type=int, default=1,
                        help="spawned simulator workers used for per-seed collection")
    parser.add_argument("--collection-threads", type=int, default=1,
                        help="torch CPU threads per collection worker")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, default=[])
    parser.add_argument("--search-value", type=Path)
    parser.add_argument("--search-width", type=int, default=3)
    parser.add_argument("--search-candidates", type=int, default=8)
    parser.add_argument("--search-horizon-ticks", type=int, default=900)
    parser.add_argument("--search-simulation-budget", type=int, default=256)
    parser.add_argument("--search-max-decisions", type=int, default=64)
    parser.add_argument("--clear-hidden", action="store_true")
    parser.add_argument("--no-relation-bias", action="store_true")
    parser.add_argument("--zombie-count-multiplier", type=float, default=1.0)
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.resource_dir:
        parser.error("set --resource-dir or PVZ_RESOURCE_DIR")
    if not args.search_value and not args.checkpoint:
        parser.error("select --search-value and/or --checkpoint LABEL=PATH")
    if args.search_label_diagnostics and (not args.search_value or args.final_test):
        parser.error("search label diagnostics require --search-value and development seeds")
    if min(args.search_width, args.search_candidates, args.search_horizon_ticks,
           args.search_simulation_budget, args.search_max_decisions, args.max_actions,
           args.workers, args.collection_threads) < 1:
        parser.error("search parameters and --max-actions must be positive")
    if not 1.0 <= args.zombie_count_multiplier <= 10.0:
        parser.error("--zombie-count-multiplier must be from 1 to 10")
    labels = (["search_teacher"] if args.search_value else []) + [label for label, _ in args.checkpoint]
    if len(labels) != len(set(labels)):
        parser.error("policy labels must be unique")

    evaluation_role = "final_test" if args.final_test else "development"
    seed_path = args.seeds or (DEFAULT_TEST_SEEDS if args.final_test else DEFAULT_DEV_SEEDS)
    seeds = read_seed_set(seed_path, args.level, evaluation_role)
    device = torch.device("cpu")
    current_task_signature = task_signature(
        args.level, args.deck, args.zombie_count_multiplier, args.resource_dir
    )
    git_sha, git_dirty = git_metadata(ROOT)
    source_names = (
        "benchmark_pvz_agent.py", "pvz_agent_model.py", "pvz_common.py", "pvz_env.py",
        "pvz_search.py", "pvz_search_candidates.py", "pvz_search_diagnostics.py",
        "pvz_search_value.py", "pvz_seed_jobs.py", "pvz_seed_sets.py", "pvz_training_artifacts.py",
        "pvz_value.py",
    )
    runtime_signature = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "source_sha256": {name: sha256_file(ROOT / "python" / name) for name in source_names},
        "simulator_sha256": sha256_file(ROOT / "build" / "pvz-portable"),
    }
    policies: list[tuple[str, GameplayModelV1 | None, Any, dict[str, Any]]] = []
    if args.search_value:
        value_model, value_checkpoint = load_search_value(
            args.search_value, device, current_task_signature,
        )
        policies.append((
            "search_teacher",
            None,
            value_model,
            {"sha256": sha256_file(args.search_value), "checkpoint": checkpoint_metadata(value_checkpoint)},
        ))
    for label, path in args.checkpoint:
        model, checkpoint = checkpoint_model(path, device)
        if args.no_relation_bias:
            for layer in model.encoder:
                layer.attention.relation_bias_enabled = False
        policies.append((label, model, None, {
            "sha256": sha256_file(path),
            "provenance": checkpoint.get("provenance"),
        }))

    all_records: dict[str, Any] = {}
    label_samples: list[dict[str, Any]] = []
    resource_metadata = None
    replay_dir = args.output.parent / f"{args.output.stem}_replays"
    worker_rss_values: list[int] = []
    for label, model, value_model, policy_metadata in policies:
        policy_kind = "search_teacher" if value_model is not None else "gameplay"
        policy_model = value_model if value_model is not None else model
        if policy_model is None:
            raise RuntimeError(f"policy {label} has no model")
        model_state = _cpu_state(policy_model)
        collect_label_samples = bool(args.search_label_diagnostics and label == "search_teacher")
        settings = {
            "collection_threads": args.collection_threads,
            "search_width": args.search_width,
            "search_candidates": args.search_candidates,
            "search_horizon_ticks": args.search_horizon_ticks,
            "search_simulation_budget": args.search_simulation_budget,
            "search_max_decisions": args.search_max_decisions,
            "clear_hidden": args.clear_hidden,
            "no_relation_bias": args.no_relation_bias,
            "zombie_count_multiplier": args.zombie_count_multiplier,
            "replay_dir": str(replay_dir),
            "policy_label": label,
            "level": args.level,
            "deck": args.deck,
            "max_actions": args.max_actions,
            "collect_label_samples": collect_label_samples,
        }
        job_metadata = {
            "evaluation_role": evaluation_role,
            "seed_file_sha256": sha256_file(seed_path),
            "seeds": seeds,
            "task_signature": current_task_signature,
            "policy_label": label,
            "policy_sha256": policy_metadata["sha256"],
            "model_state_sha256": _state_sha256(model_state),
            "search": {
                "width": args.search_width,
                "candidates": args.search_candidates,
                "horizon_ticks": args.search_horizon_ticks,
                "simulation_budget": args.search_simulation_budget,
                "max_decisions": args.search_max_decisions,
            },
            "clear_hidden": args.clear_hidden,
            "relation_bias": not args.no_relation_bias,
            "max_actions": args.max_actions,
            "label_diagnostics": collect_label_samples,
            "worker_configuration": {
                "workers": args.workers,
                "torch_threads": args.collection_threads,
            },
            "runtime": runtime_signature,
        }
        job_results = run_seed_jobs(
            seeds,
            seed_job_directory(args.output.parent, "benchmark", job_metadata),
            job_metadata,
            _benchmark_seed_worker,
            workers=args.workers,
            initializer=_initialize_benchmark_worker,
            initargs=(args.resource_dir, settings, policy_kind, model_state),
            label=label,
        )
        records = [row["record"] for row in job_results]
        if collect_label_samples:
            label_samples.extend(sample for row in job_results for sample in row["label_samples"])
        if resource_metadata is None and job_results:
            resource_metadata = job_results[0]["resources"]
        worker_rss_values.extend(
            row["worker_rss_bytes"] for row in job_results if row["worker_rss_bytes"] is not None
        )
        all_records[label] = {
            "checkpoint": policy_metadata,
            "metrics": summarize(records),
            "episodes": records,
        }
    worker_rss_bytes = max(worker_rss_values, default=None)
    result = {
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "protocol_version": ENV_PROTOCOL_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "task_version": TASK_VERSION,
        "level": args.level,
        "playthrough": 2,
        "deck": args.deck,
        "zombie_count_multiplier": args.zombie_count_multiplier,
        "profile": "Adventure-II, six slots, no store items",
        "evaluation_role": evaluation_role,
        "seed_file": str(seed_path.resolve()),
        "seed_file_sha256": sha256_file(seed_path),
        "seed_count": len(seeds),
        "seed_range": [min(seeds), max(seeds)],
        "device": "cpu",
        "worker_configuration": {
            "workers": args.workers,
            "collection_torch_threads": args.collection_threads,
            "max_workers_per_policy": min(args.workers, len(seeds)),
        },
        "clear_hidden": args.clear_hidden,
        "replay_directory": str(replay_dir.resolve()),
        "relation_bias": not args.no_relation_bias,
        "command": {"argv": sys.argv, "resource": args.resource_dir},
        "resources": resource_metadata,
        "worker_rss_bytes": worker_rss_bytes,
        "policies": all_records,
    }
    atomic_json(args.output, result)
    if args.search_label_diagnostics:
        atomic_json(args.search_label_diagnostics, {
                "evaluation_role": evaluation_role,
                "seed_file_sha256": sha256_file(seed_path),
                "task_signature": current_task_signature,
                "horizon_ticks": args.search_horizon_ticks,
                "simulation_budget": args.search_simulation_budget,
                "beam_width": args.search_width,
                "candidate_limit": args.search_candidates,
                "max_decisions": args.search_max_decisions,
                "search_value_sha256": sha256_file(args.search_value),
                "samples": label_samples,
            }, compressed=True)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
