"""Train PPO on the frozen 20-task family and evaluate with the frozen T4 runner."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import multiprocessing
from multiprocessing.util import Finalize
import os
from pathlib import Path
import random
import shlex
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_agent_model import (  # noqa: E402
    GameplayModelV1,
    MODEL_ARCHITECTURE_VERSION,
    model_architecture_version,
    MODEL_CONFIG,
    configure_torch_threads,
    resolve_device,
)
from pvz_common import (  # noqa: E402
    ENV_PROTOCOL_VERSION,
    OBSERVATION_VERSION,
    TASK_VERSION,
    git_metadata,
    sha256_file,
)
from pvz_env import PvZEnv  # noqa: E402
from pvz_event_env import policy_env  # noqa: E402
from pvz_seed_jobs import atomic_json, atomic_write, run_seed_jobs, seed_job_directory  # noqa: E402
from pvz_value import SEARCH_LABEL_VERSION, VALUE_SEMANTICS  # noqa: E402
from train_pvz_ppo import (  # noqa: E402
    EPISODE_DIGEST_ALGORITHM,
    add_advantages,
    collect_task_episode,
    episode_digest,
    train_update,
)
import check_task_manifests  # noqa: E402
import t4_capability_profile  # noqa: E402

TRAIN_PATH = ROOT / "artifacts/task_family/train.json"
HELDOUT_PATH = ROOT / "artifacts/task_family/heldout.json"
T4_GATE_PATH = ROOT / "gates/T4.json"
T4_RAW_PATH = ROOT / "artifacts/task_family/t4_seed_results.json"
THROUGHPUT_PATH = ROOT / "artifacts/t5/throughput.json"
# The PPO-update hyperparameters (2000-episode rollout, chunks=16, seq=16) were tuned
# against this measurement, which is the only recorded evidence for the update device.
UPDATE_BENCHMARK_PATH = ROOT / "artifacts/t5/perf/ppo_update_2000_flex_saved.json"
DEFAULT_RESOURCE_DIR = Path.home() / ".cache/pvz-research-resources"
CORE_THRESHOLD = 5000.0
WIN_THRESHOLD = 0.90
MAX_FORMAL_RUNS = 8
HARD_STOP_EPISODES = 20_000
HARD_STOP_PASS_RATE = 0.20
HARD_STOP_IMPROVEMENT = 0.05
RECENT_WINDOW = 64
CURVE_SECONDS = 30 * 60
TASK_RECENT: dict[str, deque[bool]] = {}
WORKER_MODEL: GameplayModelV1 | None = None
WORKER_ENV: PvZEnv | None = None
WORKER_ASSIGNMENTS: dict[int, dict[str, Any]] = {}
WORKER_MAX_ACTIONS = 4000
WORKER_REWARD_CONFIG: dict[str, Any] | None = None
WORKER_ALLOW_TRUNCATION = False
# Evaluation jobs are ``job_id -> {"task", "seed", "bucket"}``.  They are kept apart
# from the rollout assignments so the two pool initializers cannot be confused.
WORKER_EVAL_JOBS: dict[int, dict[str, Any]] = {}


def _task_family() -> tuple[dict[str, Any], dict[str, Any]]:
    train = json.loads(TRAIN_PATH.read_text(encoding="utf-8"))
    heldout = json.loads(HELDOUT_PATH.read_text(encoding="utf-8"))
    check_task_manifests.check_manifests(train, heldout)
    if len(train["tasks"]) != 20:
        raise ValueError(f"expected 20 frozen training tasks, got {len(train['tasks'])}")
    return train, heldout


def _curriculum_tasks(tasks: list[dict[str, Any]], curriculum: str) -> list[dict[str, Any]]:
    if curriculum == "all":
        selected = tasks
    elif curriculum == "cap1":
        selected = [task for task in tasks
                    if task["wave_cap"] == 1 and task["zombie_count_multiplier"] == 1.0]
    else:
        raise ValueError(f"unsupported curriculum: {curriculum}")
    if not selected:
        raise ValueError(f"curriculum {curriculum!r} selected no training tasks")
    return selected


def _heldout_tasks(heldout: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cap3 = [task for task in heldout["tasks"] if task["wave_cap"] == 3]
    gate = [task for task in cap3 if task["zombie_count_multiplier"] == 1.0]
    if not gate or sum(len(task["seeds"]) for task in gate) < 16:
        raise ValueError("frozen held-out cap3 x1.0 evaluation set is missing or too small")
    return gate, cap3


def _baseline(train_tasks: list[dict[str, Any]], gate_tasks: list[dict[str, Any]],
              reference_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = json.loads(T4_GATE_PATH.read_text(encoding="utf-8"))
    if evidence["gate_result"] != "pass":
        raise ValueError("T4 frozen evidence is not a passing gate")
    if (evidence["metrics"]["manifests"]["train_sha256"] != sha256_file(TRAIN_PATH)
            or evidence["metrics"]["manifests"]["heldout_sha256"] != sha256_file(HELDOUT_PATH)):
        raise ValueError("task manifests differ from the frozen T4 evidence")
    profiles = evidence["metrics"]["per_task"]
    t4_raw = json.loads(T4_RAW_PATH.read_text(encoding="utf-8"))
    t0_failures = [
        row for row in t4_raw["control_seed_results"]["t0_rule_script"] if not row["won"]
    ]
    train_rates = {
        task["task_id"]: profiles["train"][task["task_id"]]["pass_rate"]
        for task in train_tasks
    }

    def summarize(tasks: list[dict[str, Any]]) -> dict[str, Any]:
        task_profiles = {task["task_id"]: profiles["heldout"][task["task_id"]] for task in tasks}
        count = sum(profile["sample_count"] for profile in task_profiles.values())
        passes = sum(round(profile["pass_rate"] * profile["sample_count"])
                     for profile in task_profiles.values())
        return {
            "sample_count": count,
            "passes": passes,
            "pass_rate": passes / count,
            "terminal_wave_histogram": {
                wave: sum(profile["terminal_wave_histogram"].get(wave, 0)
                          for profile in task_profiles.values())
                for wave in sorted({
                    wave for profile in task_profiles.values()
                    for wave in profile["terminal_wave_histogram"]
                }, key=int)
            },
            "per_task": {task_id: profile["pass_rate"] for task_id, profile in task_profiles.items()},
        }

    stage0_tasks = _curriculum_tasks(train_tasks, "cap1")
    stage0_profiles = {task["task_id"]: profiles["train"][task["task_id"]]
                       for task in stage0_tasks}
    stage0_count = sum(profile["sample_count"] for profile in stage0_profiles.values())
    stage0_passes = sum(round(profile["pass_rate"] * profile["sample_count"])
                        for profile in stage0_profiles.values())
    stage0_terminal = Counter()
    for profile in stage0_profiles.values():
        stage0_terminal.update(profile["terminal_wave_histogram"])
    stage0_baseline = {
        "task_count": len(stage0_tasks),
        "sample_count": stage0_count,
        "passes": stage0_passes,
        "pass_rate": stage0_passes / stage0_count,
        "terminal_wave_histogram": dict(stage0_terminal),
        "failure_terminal_wave_histogram": dict(stage0_terminal),
        "per_task": stage0_profiles,
    }

    return {
        "t4_commit": evidence["commit"],
        "train_task_pass_rates": train_rates,
        "gate_set": summarize(gate_tasks),
        "reference_set": summarize(reference_tasks),
        "stage0_set": stage0_baseline,
        "t0_rule_script_control": evidence["metrics"]["controls"]["t0_rule_script"],
        "t0_rule_script_control_fail_terminal_wave_histogram": dict(Counter(
            str(row["terminal_wave"]) for row in t0_failures
        )),
    }


def _close_worker() -> None:
    if WORKER_ENV is not None:
        WORKER_ENV.close()


def _init_worker(resource_dir: str, state_dict: dict[str, torch.Tensor],
                 assignments: dict[int, dict[str, Any]], max_actions: int,
                 worker_threads: int, worker_device: str,
                 model_config: dict[str, Any] | None = None,
                 reward_config: dict[str, Any] | None = None,
                 allow_truncation: bool = False) -> None:
    global WORKER_MODEL, WORKER_ENV, WORKER_ASSIGNMENTS, WORKER_MAX_ACTIONS
    global WORKER_REWARD_CONFIG, WORKER_ALLOW_TRUNCATION
    configure_torch_threads(worker_threads)
    WORKER_MODEL = GameplayModelV1(model_config).eval().to(resolve_device(worker_device))
    WORKER_MODEL.load_state_dict(state_dict)
    WORKER_ENV = policy_env(WORKER_MODEL.config, resource_dir=resource_dir)
    WORKER_ASSIGNMENTS = assignments
    WORKER_MAX_ACTIONS = max_actions
    WORKER_REWARD_CONFIG = reward_config
    WORKER_ALLOW_TRUNCATION = allow_truncation
    Finalize(None, _close_worker, exitpriority=10)


def _rollout_worker(job_id: int) -> dict[str, Any]:
    if WORKER_MODEL is None or WORKER_ENV is None:
        raise RuntimeError("PPO rollout worker was not initialized")
    assignment = WORKER_ASSIGNMENTS[job_id]
    limit = assignment.get("decision_quota", WORKER_MAX_ACTIONS)
    if type(limit) is not int or not 1 <= limit <= WORKER_MAX_ACTIONS:
        raise ValueError("invalid rollout decision reservation")
    torch.manual_seed(assignment["action_seed"])
    episode = collect_task_episode(
        WORKER_MODEL,
        WORKER_ENV,
        assignment["task"],
        assignment["task_seed"],
        job_id,
        limit,
        WORKER_REWARD_CONFIG,
        allow_truncation=WORKER_ALLOW_TRUNCATION,
    )
    if "decision_quota" in assignment:
        episode["decision_quota"] = limit
        episode["truncation_reason"] = (
            "decision_budget_boundary" if episode["truncated"] and limit < WORKER_MAX_ACTIONS
            else "max_actions" if episode["truncated"] else None)
    return episode


def _init_evaluation_worker(resource_dir: str, state_dict: dict[str, torch.Tensor],
                            jobs: dict[int, dict[str, Any]], worker_threads: int,
                            worker_device: str,
                            model_config: dict[str, Any] | None = None) -> None:
    global WORKER_MODEL, WORKER_ENV, WORKER_EVAL_JOBS
    configure_torch_threads(worker_threads)
    WORKER_MODEL = GameplayModelV1(model_config).eval().to(resolve_device(worker_device))
    WORKER_MODEL.load_state_dict(state_dict)
    WORKER_ENV = policy_env(WORKER_MODEL.config, resource_dir=resource_dir)
    WORKER_EVAL_JOBS = jobs
    Finalize(None, _close_worker, exitpriority=10)


def _evaluation_worker(job_id: int) -> tuple[int, dict[str, Any]]:
    if WORKER_MODEL is None or WORKER_ENV is None:
        raise RuntimeError("evaluation worker was not initialized")
    job = WORKER_EVAL_JOBS[job_id]
    torch.manual_seed(job.get("action_seed", job["seed"] + 170_000))
    record = t4_capability_profile.run_episode(
        WORKER_ENV, job["task"], job["seed"], "checkpoint", WORKER_MODEL,
        deterministic=job.get("deterministic", True),
        max_actions=job.get("max_actions", 4000),
        allow_truncation=job.get("allow_truncation", False))
    return job_id, record


def _run_evaluation_jobs(jobs: dict[int, dict[str, Any]], model_state: dict[str, torch.Tensor],
                         resource_dir: Path, workers: int, worker_threads: int,
                         worker_device: str,
                         model_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run every ``(task, seed)`` evaluation episode across *workers* processes.

    Evaluation used to be a single-threaded loop over 1,280 episodes while rollout
    spread 2,000 over 18 workers, so evaluation was the largest serial section of a
    run.      Each episode is fully determined by its ``(task, seed)`` pair and the model
    weights, so spreading them over processes cannot change any result.

    Returns one record per job in ascending job-id order; :func:`_evaluate` relies on
    that ordering to regroup records back onto their tasks.
    """
    if not jobs:
        return []
    context = multiprocessing.get_context("spawn")
    pool = context.Pool(min(workers, len(jobs)), initializer=_init_evaluation_worker,
                        initargs=(str(resource_dir), model_state, jobs, worker_threads,
                                  worker_device, model_config))
    collected: dict[int, dict[str, Any]] = {}
    total_jobs = len(jobs)
    try:
        iterator = pool.imap_unordered(_evaluation_worker, sorted(jobs), chunksize=1)
        while True:
            try:
                job_id, record = iterator.next(timeout=900)
            except StopIteration:
                break
            except multiprocessing.TimeoutError:
                raise RuntimeError("evaluation stalled: no job completed in 900 seconds") from None
            collected[job_id] = record
            # Evaluation is 1,280 episodes on the reference-inclusive pass.  Without
            # this it was a silent gap in the log, indistinguishable from a hang.
            if len(collected) % 64 == 0 or len(collected) == total_jobs:
                print(f"evaluation {len(collected)}/{total_jobs}", flush=True)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()
    missing = sorted(set(jobs) - set(collected))
    if missing:
        raise RuntimeError(f"evaluation jobs {missing[:8]} produced no result")
    return [collected[job_id] for job_id in sorted(jobs)]


def _sampling_weights(tasks: list[dict[str, Any]]) -> list[float]:
    return [
        2.0 - (sum(TASK_RECENT[task["task_id"]]) / len(TASK_RECENT[task["task_id"]])
               if TASK_RECENT[task["task_id"]] else 0.0)
        for task in tasks
    ]


def _assignments(tasks: list[dict[str, Any]], job_ids: list[int], rng: random.Random
                 ) -> dict[int, dict[str, Any]]:
    weights = _sampling_weights(tasks)
    assignments = {}
    for job_id in job_ids:
        task = rng.choices(tasks, weights=weights, k=1)[0]
        assignments[job_id] = {
            "task": task,
            "task_seed": rng.choice(task["seeds"]),
            "action_seed": rng.randrange(1, 2**31),
        }
    return assignments


def _evaluation_jobs(tasks: list[dict[str, Any]], gate_tasks: list[dict[str, Any]],
                     stage0_tasks: list[dict[str, Any]], include_reference: bool
                     ) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]],
                                list[tuple[str, str]]]:
    """Lay out every evaluation episode as a flat, ordered job table.

    Returns the reference tasks actually used, ``job_id -> {"task", "seed"}`` and the
    matching ``(bucket, task_id)`` labels.  Keeping this pure means the layout can be
    asserted without spawning a pool.
    """
    gate_ids = {task["task_id"] for task in gate_tasks}
    reference_tasks = tasks if include_reference else [
        task for task in tasks if task["task_id"] in gate_ids
    ]
    jobs: dict[int, dict[str, Any]] = {}
    layout: list[tuple[str, str]] = []
    for bucket, bucket_tasks in (("reference", reference_tasks), ("stage0", stage0_tasks)):
        for task in bucket_tasks:
            for seed in task["seeds"]:
                jobs[len(jobs)] = {"task": task, "seed": seed}
                layout.append((bucket, task["task_id"]))
    return reference_tasks, jobs, layout


def _evaluate(model: GameplayModelV1, resource_dir: Path, tasks: list[dict[str, Any]],
              gate_tasks: list[dict[str, Any]], stage0_tasks: list[dict[str, Any]],
              episodes: int, output_dir: Path, *,
              workers: int, worker_threads: int, worker_device: str,
              include_reference: bool = True,
              runner: Any = None) -> dict[str, Any]:
    """Evaluate the frozen held-out and stage-0 sets across *workers* processes.

    ``include_reference`` controls the ``cap3 x != 1.0`` reference set.  It gates
    nothing (it is reported, never thresholded), so intermediate evaluations skip it
    and only the run's final evaluation pays for those episodes.

    ``runner`` is injectable so tests can exercise the layout and regrouping without
    spawning processes; it defaults to :func:`_run_evaluation_jobs`.
    """
    eval_model = GameplayModelV1().eval()
    eval_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    model_state = {key: value.detach().cpu() for key, value in eval_model.state_dict().items()}
    gate_ids = {task["task_id"] for task in gate_tasks}
    reference_tasks, jobs, layout = _evaluation_jobs(
        tasks, gate_tasks, stage0_tasks, include_reference)
    run_jobs = runner or _run_evaluation_jobs
    results = run_jobs(jobs, model_state, resource_dir, workers, worker_threads, worker_device)
    if len(results) != len(jobs):
        raise RuntimeError(
            f"evaluation returned {len(results)} records for {len(jobs)} jobs; "
            "the runner must return one record per job in ascending job-id order")
    records: dict[str, list[dict[str, Any]]] = {task["task_id"]: [] for task in reference_tasks}
    stage0_records: dict[str, list[dict[str, Any]]] = {task["task_id"]: [] for task in stage0_tasks}
    for index, record in enumerate(results):
        bucket, task_id = layout[index]
        (records if bucket == "reference" else stage0_records)[task_id].append(record)

    gate_records = [record for task_id, rows in records.items() if task_id in gate_ids for record in rows]
    reference_records = [record for rows in records.values() for record in rows]
    gate_summary = t4_capability_profile.summarize_episodes(gate_records)
    stage0_records_flat = [record for rows in stage0_records.values() for record in rows]
    stage0_summary = t4_capability_profile.summarize_episodes(stage0_records_flat)
    suffix = "" if include_reference else "_gate_only"
    raw_path = output_dir / "evaluations" / f"heldout_{episodes:07d}{suffix}.json.gz"
    # An output dir outside the repo is legitimate, so fall back to the absolute path
    # rather than raising on relative_to.
    try:
        raw_reference = str(raw_path.relative_to(ROOT))
    except ValueError:
        raw_reference = str(raw_path)
    atomic_json(raw_path, {
        "cumulative_episodes": episodes,
        "reference_set_included": include_reference,
        "seed_results": records,
        "stage0_seed_results": stage0_records,
    }, compressed=True)
    if include_reference:
        reference_summary = t4_capability_profile.summarize_episodes(reference_records)
        reference_set: dict[str, Any] = {
            "skipped": False,
            "task_count": len(reference_tasks),
            "sample_count": reference_summary["sample_count"],
            "passes": round(reference_summary["pass_rate"] * reference_summary["sample_count"]),
            "pass_rate": reference_summary["pass_rate"],
            "per_task": {
                task_id: t4_capability_profile.summarize_episodes(rows)
                for task_id, rows in records.items()
            },
        }
    else:
        reference_set = {
            "skipped": True,
            "reason": "intermediate evaluation: the reference set gates nothing, so it "
                      "is evaluated once per run on the final model",
            "task_count": 0,
            "sample_count": 0,
            "passes": 0,
            "pass_rate": None,
            "per_task": {},
        }
    return {
        "cumulative_episodes": episodes,
        "gate_set": {
            "task_count": len(gate_ids),
            "sample_count": gate_summary["sample_count"],
            "passes": round(gate_summary["pass_rate"] * gate_summary["sample_count"]),
            "pass_rate": gate_summary["pass_rate"],
            "terminal_wave_histogram": gate_summary["terminal_wave_histogram"],
            "failure_terminal_wave_histogram": dict(Counter(
                str(record["terminal_wave"]) for record in gate_records if not record["won"]
            )),
            "per_task": {
                task_id: t4_capability_profile.summarize_episodes(rows)
                for task_id, rows in records.items() if task_id in gate_ids
            },
        },
        "reference_set": reference_set,
        "stage0_set": {
            "task_count": len(stage0_tasks),
            "sample_count": stage0_summary["sample_count"],
            "passes": round(stage0_summary["pass_rate"] * stage0_summary["sample_count"]),
            "pass_rate": stage0_summary["pass_rate"],
            "terminal_wave_histogram": stage0_summary["terminal_wave_histogram"],
            "failure_terminal_wave_histogram": dict(Counter(
                str(record["terminal_wave"]) for record in stage0_records_flat if not record["won"]
            )),
            "per_task": {
                task_id: t4_capability_profile.summarize_episodes(rows)
                for task_id, rows in stage0_records.items()
            },
        },
        "raw_seed_results_path": raw_reference,
    }


def _curve_row(episodes: int, train_tasks: list[dict[str, Any]],
               gate_rate: float, reference_rate: float, stage0_rate: float,
               source: str, timing: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cumulative_training_episodes": episodes,
        "curriculum_task_ids": [task["task_id"] for task in train_tasks],
        "train_task_pass_rates_recent_64": {
            task["task_id"]: (sum(TASK_RECENT[task["task_id"]]) / len(TASK_RECENT[task["task_id"]])
                              if TASK_RECENT[task["task_id"]] else None)
            for task in train_tasks
        },
        "heldout_cap3_x1_pass_rate": gate_rate,
        "heldout_cap3_all_reference_pass_rate": reference_rate,
        "stage0_pass_rate": stage0_rate,
        "evaluation_source": source,
    }
    # A curve without its cost cannot answer "did this get slower", which is the
    # question that matters once a run takes hours.
    if timing:
        row["timing_seconds"] = timing
    return row


def _save_checkpoint(path: Path, model: GameplayModelV1, config: dict[str, Any], provenance: dict[str, Any],
                     run_number: int, update: int, losses: dict[str, float]) -> None:
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_architecture_version": model_architecture_version(model.config),
        "value_semantics": VALUE_SEMANTICS,
        "config": model.config,
        "level": None,
        "deck": None,
        "profile": "T5 frozen 20-task family",
        "update": update,
        "run_number": run_number,
        "ppo_config": config,
        "losses": losses,
        "provenance": provenance,
    }
    atomic_write(path, lambda temporary_path: torch.save(checkpoint, temporary_path))


def _save_state(path: Path, state: dict[str, Any], curve_path: Path) -> None:
    atomic_json(path, state)
    atomic_json(curve_path, state["learning_curve"])


def _check_stage0_gate(curriculum_tasks: list[dict[str, Any]], ignore_gate: bool,
                       motivation: str | None, gate_path: Path | None = None) -> dict[str, Any] | None:
    if ignore_gate and not motivation:
        raise ValueError("--ignore-stage0-gate requires --motivation with a reason")
    if len(curriculum_tasks) != 20:
        return None
    if ignore_gate:
        return {"used": True, "reason": motivation}
    gate_path = gate_path or ROOT / "artifacts/t5/stage0_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("阶段 0 未通过，禁止进入阶段 1：stage0_gate.json 不存在")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("result") != "pass":
        raise RuntimeError("阶段 0 未通过，禁止进入阶段 1：stage0_gate.json 未通过")
    return None


def _seed0_initialization_baseline(actual_hash: str, t4_hash: str,
                                  note: str | None) -> dict[str, Any]:
    """Record that seed 0 no longer matches the frozen T4 baseline.

    ``MODEL_ARCHITECTURE_VERSION`` has stood above the frozen T4 value (4) ever
    since lane-token inputs landed, so the seeded state can never equal the T4 one
    again.  The former "hashes match" branch and the "hashes differ but the
    architecture version did not move" branch were therefore both unreachable, and
    the latter would have raised on a state that cannot occur.  What remains is the
    part that still does something: refuse to start without an explicit note.
    """
    if not note:
        raise RuntimeError("网络结构已变更，seed-0 初始化不再与 T4 基线一致；"
                           "请传入 --initialization-note 说明 T4 基线为何已被取代")
    return {"status": "superseded", "actual": actual_hash, "t4": t4_hash, "note": note}


def _check_update_device(device: torch.device,
                         benchmark_path: Path = UPDATE_BENCHMARK_PATH,
                         allow_cpu_update: bool = False,
                         motivation: str | None = None) -> dict[str, Any]:
    """Refuse a formal run whose PPO update silently degraded to the CPU.

    ``resolve_device("auto")`` falls back to the CPU without complaint when CUDA
    stops being visible.  That happens on the WSL box after long uptime (the GPU
    drops off the bus until WSL is restarted), and the failure is invisible in the
    training log: rollout still runs on its 18 CPU workers, the state file is only
    rewritten after an update, so the run looks alive while the first update never
    lands.  A real occurrence burned 75+ minutes of a stage 0 that should have
    produced its first update in ~6 minutes.

    The update hyperparameters are tuned against the measurement recorded in
    ``UPDATE_BENCHMARK_PATH`` (an RTX 5080, ~184 s per 2,000-episode update).  On
    the CPU the same update measures ~1,073 s, so running there is a *different*
    experiment rather than a slower version of this one, and needs
    ``--allow-cpu-update`` plus ``--motivation``.
    """
    record: dict[str, Any] = {
        "resolved_device": str(device),
        "benchmark": str(benchmark_path),
        "benchmark_device": None,
        "benchmark_device_name": None,
        "override": None,
    }
    if benchmark_path.is_file():
        measured = json.loads(benchmark_path.read_text(encoding="utf-8"))
        record["benchmark_device"] = measured.get("device")
        record["benchmark_device_name"] = measured.get("device_name")
    if device.type == "cuda":
        record["device_name"] = torch.cuda.get_device_name(device)
        return record
    if record["benchmark_device"] != "cuda":
        # No CUDA-specific evidence was ever recorded, so a CPU update is the honest
        # default rather than a silent degradation.
        return record
    if allow_cpu_update:
        if not motivation:
            raise ValueError("--allow-cpu-update requires --motivation")
        record["override"] = {"used": True, "reason": motivation}
        return record
    raise RuntimeError(
        "CUDA is unavailable, but the T5 update hyperparameters were measured on "
        f"{record['benchmark_device_name'] or 'CUDA'} (see {record['benchmark']}). "
        "resolve_device('auto') would silently fall back to the CPU, where one "
        "2,000-episode update costs ~1,073 s instead of ~184 s -- a stage 0 that "
        "should finish in ~1 h would instead run for many hours. Fix the GPU "
        "(restart WSL / re-check `nvidia-smi`) and rerun, or pass "
        "--allow-cpu-update --motivation to run the CPU update on purpose."
    )


def _stage0_has_no_signal(pass_rate: float, tasks: list[dict[str, Any]]) -> bool:
    return pass_rate == 0.0 and all(
        len(TASK_RECENT[task["task_id"]]) == RECENT_WINDOW
        and not any(TASK_RECENT[task["task_id"]])
        for task in tasks
    )


def _curve_rises(curve: list[dict[str, Any]]) -> bool:
    rates = [row["heldout_cap3_x1_pass_rate"] for row in curve]
    return any(current > previous for previous, current in zip(rates, rates[1:]))


def _gate_document(state: dict[str, Any], throughput: dict[str, Any], baseline: dict[str, Any],
                   heldout: dict[str, Any], revision: str, worktree_clean: bool,
                   protected_assets_unmodified: bool, reproduce_command: str) -> dict[str, Any]:
    rate = heldout["gate_set"]["pass_rate"]
    rising = _curve_rises(state["learning_curve"])
    passed = (
        throughput["single_core_episodes_per_hour"] >= CORE_THRESHOLD
        and baseline["gate_set"]["pass_rate"] < 0.10
        and heldout["gate_set"]["sample_count"] >= 16
        and rate >= WIN_THRESHOLD
        and rising
    )
    result = {
        "task_id": "T5",
        "commit": revision,
        "worktree_clean": worktree_clean,
        "reproduce_command": reproduce_command,
        "gate_result": "pass" if passed else "fail",
        "metrics": {
            "single_core_episodes_per_hour": throughput["single_core_episodes_per_hour"],
            "throughput_configurations": throughput["configurations"],
            "selected_parallel_workers": throughput["selected_parallel_workers"],
            "pretraining_cap3_x1_pass_rate_from_T4": baseline["gate_set"],
            "posttraining_cap3_x1": heldout["gate_set"],
            "heldout_cap3_all_reference": heldout["reference_set"],
            "cumulative_training_episodes": state["cumulative_episodes"],
            "formal_runs": state["formal_runs"],
            "checkpoint": state["runs"][-1].get("checkpoint"),
            "learning_curve_path": "artifacts/t5/learning_curve.json",
            "throughput_experiment_path": "artifacts/t5/throughput.json",
            "learning_curve_rises": rising,
            "stop_reason": state.get("stop_reason"),
        },
        "thresholds": {
            "single_core_episodes_per_hour_min": CORE_THRESHOLD,
            "pretraining_cap3_x1_pass_rate_max_exclusive": 0.10,
            "posttraining_cap3_x1_pass_rate_min": WIN_THRESHOLD,
            "posttraining_minimum_seed_count": 16,
            "learning_curve_must_rise": True,
            "maximum_formal_runs": MAX_FORMAL_RUNS,
            "hard_stop": {
                "cumulative_episodes_min": HARD_STOP_EPISODES,
                "pass_rate_below": HARD_STOP_PASS_RATE,
                "recent_5000_episode_improvement_below": HARD_STOP_IMPROVEMENT,
            },
        },
        "raw_seed_results_path": heldout["raw_seed_results_path"],
        "protected_assets_unmodified": protected_assets_unmodified,
        "notes": (
            "吞吐实验先于正式训练完成。门禁评估通过冻结 T4 run_episode / summarize_episodes，"
            "筛选 heldout wave_cap=3、倍率 1.0 的全部任务；1.5 倍任务只作参考。"
        ),
    }
    if not passed:
        history = state.get("evaluations", [])
        prior = max(
            (item for item in history if item["cumulative_episodes"] <= state["cumulative_episodes"] - 5000),
            key=lambda item: item["cumulative_episodes"],
            default=None,
        )
        improvement = None if prior is None else rate - prior["gate_set"]["pass_rate"]
        if rate == 0.0:
            failure_layer = "algorithm"
            priority = 1
            current_lr = state["runs"][-1].get("learning_rate")
            proposed = (
                "当前势函数三项等权且范围为 [0,1]，终局奖励仍为 ±1；先保留事实数据，"
                f"下一次运行把学习率从 {current_lr} 调到 1e-5，比较终局回报进入 GAE 后的学习曲线。"
            )
        elif rate < WIN_THRESHOLD:
            failure_layer = "algorithm"
            priority = 2
            proposed = (
                "当前策略未达到 90% 门槛；下一次运行在保持冻结评估集与任务集不变的前提下，"
                "按本轮逐种子终局波次分布定位失败任务，再只调整允许的 PPO 超参数。"
            )
        else:
            failure_layer = "algorithm"
            priority = 0
            proposed = "检查门禁组合指标与学习曲线记录，定位未通过的具体字段。"
        result["diagnostic"] = {
            "failure_layer": failure_layer,
            "priority_check": priority,
            "observed": {
                "heldout_pass_rate": rate,
                "previous_eval_episodes": None if prior is None else prior["cumulative_episodes"],
                "previous_eval_pass_rate": None if prior is None else prior["gate_set"]["pass_rate"],
                "recent_5000_episode_improvement": improvement,
                "failed_seed_terminal_wave_histogram": heldout["gate_set"].get("failure_terminal_wave_histogram"),
                "T0_script_failed_seed_terminal_wave_histogram": baseline["t0_rule_script_control_fail_terminal_wave_histogram"],
                "reward_and_optimizer_diagnostics": state.get("last_training_debug"),
                "single_core_throughput_profile": throughput["configurations"][-1]["mean_profile_seconds"],
            },
            "proposed_next_change": proposed,
        }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path,
                        help="explicit, frozen T5–T7 research configuration")
    parser.add_argument("--resume", action="store_true",
                        help="restore the same research candidate including optimizer and RNG")
    parser.add_argument("--stop-after-updates", type=int,
                        help="end this research invocation at a completed update boundary")
    parser.add_argument("--resource-dir", type=Path,
                        default=Path(os.environ.get("PVZ_RESOURCE_DIR", DEFAULT_RESOURCE_DIR)))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/t5")
    parser.add_argument("--curriculum", choices=("all", "cap1"), default="all",
                        help="sample all frozen tasks, or only wave_cap=1 and multiplier=1.0")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rollout-threads", type=int,
                        help="override the measured Torch threads per rollout worker")
    parser.add_argument("--rollout-device", choices=("cpu", "cuda"),
                        help="override the measured rollout worker device")
    parser.add_argument("--run-number", type=int)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--motivation")
    parser.add_argument("--ignore-stage0-gate", action="store_true",
                        help="explicitly override the stage 0 block; requires --motivation")
    parser.add_argument("--initialization-note",
                        help="explain why the changed network supersedes the T4 seed-0 baseline")
    parser.add_argument("--rollout-episodes", type=int, default=2000,
                        help="episodes collected before each PPO update; 2000 amortizes "
                             "the measured RTX 5080 update cost")
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=16,
                        help="truncated-BPTT steps; 16 was fastest in the measured sweep")
    parser.add_argument("--max-actions", type=int, default=4000)
    parser.add_argument("--max-episodes-per-run", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="measured with 16 chunks per optimizer step")
    parser.add_argument("--minibatch-chunks", type=int, default=16,
                        help="chunks per optimizer step; 16 was fastest in the RTX 5080 sweep "
                             "(32 used more memory and was slower)")
    parser.add_argument("--attention-backend", choices=("auto", "dense", "flex"), default="auto",
                        help="auto uses the dense path, which is the faster of the two on "
                             "CUDA (PPO_UPDATE_ANATOMY.md §10: 81.7 vs 125.8 ms per optimizer "
                             "step at 512 episodes); flex opts into FlexAttention explicitly")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--allow-cpu-update", action="store_true",
                        help="permit a PPO update on the CPU when the measured update "
                             "device was CUDA; requires --motivation")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--initialization-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.experiment_config is not None:
        from pvz_research import run_experiment
        run_experiment(args)
        return
    if args.workers is not None and args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.rollout_threads is not None and args.rollout_threads < 1:
        raise SystemExit("--rollout-threads must be positive")
    if args.rollout_episodes < 1 or args.ppo_epochs < 1 or args.sequence_length < 1 or args.max_actions < 1:
        raise SystemExit("rollout size, PPO epochs, sequence length, and max actions must be positive")
    if args.max_episodes_per_run < 1 or not 0.0 < args.gae_lambda <= 1.0:
        raise SystemExit("max episodes must be positive and GAE lambda must be in (0, 1]")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "training_state.json"
    curve_path = output_dir / "learning_curve.json"
    train, heldout = _task_family()
    stage0_tasks = _curriculum_tasks(train["tasks"], "cap1")
    curriculum_tasks = _curriculum_tasks(train["tasks"], args.curriculum)
    stage0_gate_override = _check_stage0_gate(
        curriculum_tasks, args.ignore_stage0_gate, args.motivation)
    gate_tasks, reference_tasks = _heldout_tasks(heldout)
    baseline = _baseline(train["tasks"], gate_tasks, reference_tasks)
    protected_paths = (
        T4_GATE_PATH, ROOT / "scripts/t4_capability_profile.py", ROOT / "TODO.md",
        ROOT / "DESIGN.md", TRAIN_PATH, HELDOUT_PATH,
    )
    protected_before = {path: sha256_file(path) for path in protected_paths}
    throughput = json.loads(THROUGHPUT_PATH.read_text(encoding="utf-8"))
    if not throughput["single_core_threshold_met"]:
        raise RuntimeError("single-core throughput gate failed; formal training is not allowed")
    workers = args.workers or throughput["selected_parallel_workers"]
    worker_threads = args.rollout_threads or throughput["selected_torch_threads_per_worker"]
    worker_device = args.rollout_device or throughput["selected_rollout_device"]
    # Resolve early so a stale or manually edited throughput report fails clearly.
    resolve_device(worker_device)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("stop_reason"):
            raise RuntimeError(f"T5 already stopped: {state['stop_reason']}")
    else:
        state = {
            "schema_version": 1,
            "task_id": "T5",
            "cumulative_episodes": 0,
            "formal_runs": 0,
            "runs": [],
            "evaluations": [{
                "cumulative_episodes": 0,
                "gate_set": baseline["gate_set"],
                "reference_set": baseline["reference_set"],
                "stage0_set": baseline["stage0_set"],
                "source": "gates/T4.json; no baseline retest",
            }],
            "learning_curve": [{
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cumulative_training_episodes": 0,
                "train_task_pass_rates_recent_64": baseline["train_task_pass_rates"],
                "heldout_cap3_x1_pass_rate": baseline["gate_set"]["pass_rate"],
                "heldout_cap3_all_reference_pass_rate": baseline["reference_set"]["pass_rate"],
                "stage0_pass_rate": baseline["stage0_set"]["pass_rate"],
                "curriculum_task_ids": [task["task_id"] for task in curriculum_tasks],
                "evaluation_source": "gates/T4.json; no baseline retest",
            }],
            "recent_passes": {task["task_id"]: [] for task in train["tasks"]},
            "last_training_debug": {},
        }
    TASK_RECENT.clear()
    TASK_RECENT.update({
        task_id: deque(values, maxlen=RECENT_WINDOW)
        for task_id, values in state["recent_passes"].items()
    })
    for task in train["tasks"]:
        TASK_RECENT.setdefault(task["task_id"], deque(maxlen=RECENT_WINDOW))

    run_number = state["formal_runs"] + 1
    if run_number > MAX_FORMAL_RUNS:
        raise RuntimeError(f"T5 formal training is limited to {MAX_FORMAL_RUNS} runs")
    if args.run_number is not None and args.run_number != run_number:
        raise ValueError(f"next allowed T5 run number is {run_number}, got {args.run_number}")
    if run_number > 1 and not args.motivation:
        raise ValueError("a follow-up run must record why its hyperparameters changed")

    previous_run_dir = output_dir / "runs" / f"run_{run_number - 1}"
    init_checkpoint = args.init_checkpoint.expanduser().resolve() if args.init_checkpoint else None
    if init_checkpoint is None and run_number > 1:
        init_checkpoint = previous_run_dir / "gameplay_model_v1_ppo.pt"
    configure_torch_threads(1)
    device = resolve_device(args.device)
    update_device_check = _check_update_device(
        device, allow_cpu_update=args.allow_cpu_update, motivation=args.motivation)
    if update_device_check["resolved_device"] != "cuda":
        print(f"WARNING: PPO update will run on {update_device_check['resolved_device']}; "
              f"measured benchmark device was {update_device_check['benchmark_device']}",
              file=sys.stderr)
    initialization_baseline = None
    if init_checkpoint:
        initial = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        provenance = initial["provenance"]
        if (initial["model_architecture_version"] != MODEL_ARCHITECTURE_VERSION
                or initial.get("value_semantics") != VALUE_SEMANTICS
                or provenance.get("protocol_version") != ENV_PROTOCOL_VERSION
                or provenance.get("search_label_version") != SEARCH_LABEL_VERSION
                or provenance["observation_version"] != OBSERVATION_VERSION
                or provenance["task_version"] != TASK_VERSION):
            raise ValueError("initial checkpoint does not match current model/search semantics")
        model = GameplayModelV1().to(device)
        model.load_state_dict(initial["state_dict"])
        initial_sha = sha256_file(init_checkpoint)
        initial_kind = "checkpoint_file"
    else:
        torch.manual_seed(args.initialization_seed)
        model = GameplayModelV1().to(device)
        initial_sha = None
        initial_kind = "random_initialization"
        if args.initialization_seed == 0:
            actual_state_hash = t4_capability_profile._state_sha256(model.state_dict())
            expected = json.loads(T4_GATE_PATH.read_text(encoding="utf-8"))["metrics"]["checkpoint"]["state_sha256"]
            initialization_baseline = _seed0_initialization_baseline(
                actual_state_hash, expected, args.initialization_note)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    revision, dirty = git_metadata(ROOT)
    resource_dir = args.resource_dir.expanduser().resolve()
    resource_hashes = {
        "main.pak": sha256_file(resource_dir / "main.pak"),
        "properties/partner.xml": sha256_file(resource_dir / "properties" / "partner.xml"),
    }
    run_dir = output_dir / "runs" / f"run_{run_number}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "run_number": run_number,
        "workers": workers,
        "torch_threads_per_worker": worker_threads,
        "rollout_device": worker_device,
        "update_device": str(device),
        "update_device_check": update_device_check,
        "rollout_worker_benchmark": throughput.get("selected_configuration"),
        "resource_dir": str(resource_dir),
        "train_manifest": str(TRAIN_PATH),
        "curriculum": args.curriculum,
        "curriculum_task_ids": [task["task_id"] for task in curriculum_tasks],
        "stage0_gate_override": stage0_gate_override,
        "heldout_manifest": str(HELDOUT_PATH),
        "train_manifest_sha256": sha256_file(TRAIN_PATH),
        "heldout_manifest_sha256": sha256_file(HELDOUT_PATH),
        "gae_discount": "VALUE_GAMMA ** (action_duration_ticks / DISCOUNT_REFERENCE_TICKS)",
        "potential": "terminal=0; otherwise (clip(sun/1000,0,1) + clip(wave/wave_count,0,1) + mean active-plant health/max_health)/3",
        "adaptive_task_weight": "2 - recent_pass_rate; last 64 completed rollouts per task; unseen task rate=0",
        "reward": "terminal result +/-1 + discount*Phi(next)-Phi(current); no fixed penalties",
        "initialization": {"kind": initial_kind, "seed": args.initialization_seed,
                           "checkpoint": None if init_checkpoint is None else str(init_checkpoint),
                           "checkpoint_sha256": initial_sha,
                           "baseline_comparison": initialization_baseline},
        "initialization_baseline": initialization_baseline,
        "motivation": args.motivation or "Run 1: PPO from the T4 seed-0 random initialization with frozen task-family rollouts.",
        "previous_curve_comparison": (
            {"source": "T4 baseline", "heldout_pass_rate": baseline["gate_set"]["pass_rate"]}
            if run_number == 1 else {
                "previous_run_number": run_number - 1,
                "previous_curve_path": str(curve_path.relative_to(ROOT)),
                "previous_final_heldout_pass_rate": state["evaluations"][-1]["gate_set"]["pass_rate"],
                "new_run_start_heldout_pass_rate": state["evaluations"][-1]["gate_set"]["pass_rate"],
            }
        ),
    }
    atomic_json(run_dir / "hyperparameters.json", run_config)
    state["formal_runs"] = run_number
    state["runs"].append({
        "run_number": run_number,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "hyperparameters_path": str((run_dir / "hyperparameters.json").relative_to(ROOT)),
        "motivation": run_config["motivation"],
        "learning_rate": args.learning_rate,
    })
    _save_state(state_path, state, curve_path)

    cumulative_start = state["cumulative_episodes"]
    run_end = cumulative_start + args.max_episodes_per_run
    run_episodes = 0
    update = 0
    rng = random.Random(args.seed + run_number - 1)
    recent_shaping_abs: deque[float] = deque(maxlen=100_000)
    terminal_outcomes: list[float] = []
    last_curve_time = time.monotonic()
    latest_eval = state["evaluations"][-1]
    stop = False
    while run_episodes < args.max_episodes_per_run and not stop:
        remaining = run_end - state["cumulative_episodes"]
        next_eval = ((state["cumulative_episodes"] // 5000) + 1) * 5000
        batch_size = min(args.rollout_episodes, remaining, next_eval - state["cumulative_episodes"])
        job_ids = list(range(run_episodes, run_episodes + batch_size))
        assignments = _assignments(curriculum_tasks, job_ids, rng)
        model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        hash_started = time.monotonic()
        model_hash = t4_capability_profile._state_sha256(model_state)
        model_hash_seconds = time.monotonic() - hash_started
        metadata = {
            "run_number": run_number,
            "update": update + 1,
            "model_state_sha256": model_hash,
            "assignments": {
                str(job_id): {
                    "task_id": value["task"]["task_id"],
                    "task_seed": value["task_seed"],
                    "action_seed": value["action_seed"],
                }
                for job_id, value in assignments.items()
            },
            "manifest_sha256": run_config["train_manifest_sha256"],
            "max_actions": args.max_actions,
            "potential": run_config["potential"],
            "rollout_device": worker_device,
            "rollout_threads": worker_threads,
        }
        shard_dir = seed_job_directory(run_dir, f"update_{update + 1:04d}", metadata)
        rollout_started = time.monotonic()
        episodes = run_seed_jobs(
            job_ids,
            shard_dir,
            metadata,
            _rollout_worker,
            workers=workers,
            initializer=_init_worker,
            initargs=(str(resource_dir), model_state, assignments, args.max_actions,
                      worker_threads, worker_device),
            label=f"T5 run {run_number} update {update + 1}",
        )
        rollout_seconds = time.monotonic() - rollout_started
        add_advantages(episodes, args.gae_lambda)
        update_started = time.monotonic()
        losses = train_update(
            model, episodes, optimizer, device, args.ppo_epochs, args.sequence_length,
            args.clip_epsilon, args.value_coefficient, args.entropy_coefficient,
            minibatch_chunks=args.minibatch_chunks,
            attention_backend=args.attention_backend,
            label=f"T5 run {run_number} update {update + 1}",
        )
        update_seconds = time.monotonic() - update_started
        update += 1
        run_episodes += len(episodes)
        state["cumulative_episodes"] += len(episodes)
        digest_started = time.monotonic()
        hashes = [episode_digest(episode) for episode in episodes]
        digest_seconds = time.monotonic() - digest_started
        for episode in episodes:
            task_id = episode["task_id"]
            TASK_RECENT[task_id].append(episode["won"])
            terminal_outcomes.append(episode["transitions"][-1]["terminal_outcome"])
            recent_shaping_abs.extend(abs(transition["shaping_reward"])
                                      for transition in episode["transitions"])
        losses["rollout_episode_wins"] = sum(episode["won"] for episode in episodes)
        losses["mean_abs_shaping_reward_recent"] = (
            sum(recent_shaping_abs) / len(recent_shaping_abs) if recent_shaping_abs else 0.0
        )
        losses["terminal_outcome_mean_recent"] = (
            sum(terminal_outcomes[-5000:]) / len(terminal_outcomes[-5000:]) if terminal_outcomes else 0.0
        )
        losses["rollout_episode_hashes"] = hashes
        # Every episode already returns a per-stage timing breakdown; the trainer used
        # to drop it on the floor, so a run could get slower with no record of why.
        profile_keys = ("model", "environment", "critic_inputs", "tokenization")
        if all("profile_seconds" in episode for episode in episodes):
            mean_profile: dict[str, float | None] = {
                key: sum(episode["profile_seconds"][key] for episode in episodes) / len(episodes)
                for key in profile_keys
            }
        else:
            # Shards cached by a build that predates per-stage timing.
            mean_profile = {key: None for key in profile_keys}
        wall_seconds = rollout_seconds + update_seconds
        state["last_training_debug"] = {
            "terminal_outcome_count": len(terminal_outcomes),
            "terminal_outcome_mean_recent": losses["terminal_outcome_mean_recent"],
            "mean_abs_shaping_reward_recent": losses["mean_abs_shaping_reward_recent"],
            "potential_range": [0.0, 1.0],
            "gae_advantage_mean": losses["advantage_mean"],
            "gae_advantage_std": losses["advantage_std"],
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "entropy": losses["entropy"],
            "gradient_norm": losses["gradient_norm"],
            "timing_seconds": {
                "rollout": round(rollout_seconds, 3),
                "ppo_update": round(update_seconds, 3),
                "episode_digest": round(digest_seconds, 3),
                "model_state_sha256": round(model_hash_seconds, 3),
                # Filled in right after the checkpoint is written; the key exists from
                # the first update so the record's shape never changes mid-run.
                "checkpoint_save": None,
                "wall": round(wall_seconds, 3),
                "episodes_per_hour": round(3600.0 * len(episodes) / wall_seconds, 1)
                if wall_seconds else None,
            },
            "mean_episode_profile_seconds": mean_profile,
        }
        state["recent_passes"] = {task_id: list(values) for task_id, values in TASK_RECENT.items()}
        state["last_update"] = {
            "run_number": run_number,
            "update": update,
            "cumulative_episodes": state["cumulative_episodes"],
            "episodes": len(episodes),
            "task_counts": {
                task["task_id"]: sum(episode["task_id"] == task["task_id"] for episode in episodes)
                for task in curriculum_tasks
            },
            # The 2,000 per-episode digests are provenance, and the checkpoint already
            # records them under ``provenance.trajectory_sha256``.  A second copy here
            # made ``training_state.json`` 102 KiB, of which 69% was hashes.
            "losses": {key: value for key, value in losses.items()
                       if key != "rollout_episode_hashes"},
        }

        checkpoint_path = run_dir / "gameplay_model_v1_ppo.pt"
        if (state["cumulative_episodes"] % 5000 == 0
                or time.monotonic() - last_curve_time >= CURVE_SECONDS
                or run_episodes >= args.max_episodes_per_run):
            eval_row = _evaluate(
                model, resource_dir, reference_tasks, gate_tasks,
                stage0_tasks,
                state["cumulative_episodes"], output_dir,
                workers=workers, worker_threads=worker_threads,
                worker_device=worker_device,
                # The run's last scheduled evaluation is the authoritative record, so
                # only it pays for the reference set.
                include_reference=run_episodes >= args.max_episodes_per_run,
            )
            latest_eval = eval_row
            state["evaluations"].append(eval_row)
            state["learning_curve"].append(_curve_row(
                state["cumulative_episodes"], curriculum_tasks,
                eval_row["gate_set"]["pass_rate"], eval_row["reference_set"]["pass_rate"],
                eval_row["stage0_set"]["pass_rate"],
                eval_row["raw_seed_results_path"],
                timing=state.get("last_training_debug", {}).get("timing_seconds"),
            ))
            last_curve_time = time.monotonic()
            rising = _curve_rises(state["learning_curve"])
            if (eval_row["gate_set"]["sample_count"] >= 16
                    and eval_row["gate_set"]["pass_rate"] >= WIN_THRESHOLD and rising):
                state["stop_reason"] = "gate_pass"
                stop = True
            elif state["cumulative_episodes"] >= HARD_STOP_EPISODES:
                prior = max(
                    (item for item in state["evaluations"]
                     if item["cumulative_episodes"] <= state["cumulative_episodes"] - 5000),
                    key=lambda item: item["cumulative_episodes"],
                    default=None,
                )
                improvement = None if prior is None else (
                    eval_row["gate_set"]["pass_rate"] - prior["gate_set"]["pass_rate"]
                )
                if (eval_row["gate_set"]["pass_rate"] < HARD_STOP_PASS_RATE
                        and improvement is not None and improvement < HARD_STOP_IMPROVEMENT):
                    state["stop_reason"] = "20k episodes: heldout pass rate below 20% and recent 5k improvement below 5 points"
                    stop = True
            if (args.curriculum == "cap1"
                    and _stage0_has_no_signal(
                        eval_row["stage0_set"]["pass_rate"], stage0_tasks)):
                state["stop_reason"] = "stage0_no_signal"
                state["stage0_diagnosis"] = {
                    "failure_layer": "stage0 learning signal",
                    "observed": {
                        "stage0_pass_rate": eval_row["stage0_set"]["pass_rate"],
                        "stage0_sample_count": eval_row["stage0_set"]["sample_count"],
                        "cap1_rolling_64_win_rates": {
                            task["task_id"]: sum(TASK_RECENT[task["task_id"]]) / RECENT_WINDOW
                            for task in stage0_tasks
                        },
                        "last_training_debug": state.get("last_training_debug"),
                    },
                    "required_review_order": [
                        "terminal +/-1 reward contribution",
                        "potential shaping magnitude versus terminal reward",
                        "advantage normalization",
                        "exploration and network signal",
                    ],
                    "continuation_answer": "再加样本会把同样的零胜失败放大；按工作令停止，不得跑满 10000 局或进入阶段 1。",
                }
                stop = True
            if stop:
                state["runs"][-1]["status"] = state["stop_reason"]

        revision, dirty = git_metadata(ROOT)
        provenance = {
            "git_sha": revision,
            "git_dirty": dirty,
            "protocol_version": ENV_PROTOCOL_VERSION,
            "command": {"argv": sys.argv, "arguments": run_config},
            "trajectory_sha256": {"ppo_rollouts_by_update": {str(update): hashes}},
            "trajectory_digest_algorithm": EPISODE_DIGEST_ALGORITHM,
            "resource_sha256": resource_hashes,
            "initial_checkpoint_sha256": initial_sha,
            "initialization_baseline": initialization_baseline,
            "task_family_manifest_sha256": {
                "train": run_config["train_manifest_sha256"],
                "heldout": run_config["heldout_manifest_sha256"],
            },
            "random_seeds": {"python_torch": args.seed, "initialization": args.initialization_seed},
            "model_config": MODEL_CONFIG,
            "observation_version": OBSERVATION_VERSION,
            "task_version": TASK_VERSION,
            "search_label_version": SEARCH_LABEL_VERSION,
            "value_semantics": VALUE_SEMANTICS,
        }
        checkpoint_started = time.monotonic()
        _save_checkpoint(checkpoint_path, model, run_config, provenance, run_number, update, losses)
        state["last_training_debug"]["timing_seconds"]["checkpoint_save"] = round(
            time.monotonic() - checkpoint_started, 3)
        rolling = sorted(
            sum(TASK_RECENT[task["task_id"]]) / len(TASK_RECENT[task["task_id"]])
            for task in curriculum_tasks if TASK_RECENT[task["task_id"]]
        )
        lowest = rolling[0] if rolling else 0.0
        median = rolling[len(rolling) // 2] if rolling else 0.0
        highest = rolling[-1] if rolling else 0.0
        per_hour = state["last_training_debug"]["timing_seconds"]["episodes_per_hour"] or 0.0
        print(
            f"run={run_number} update={update} ep={state['cumulative_episodes']} "
            f"win={losses['rollout_episode_wins']}/{len(episodes)} "
            f"taskwin[min/med/max]={lowest:.2f}/{median:.2f}/{highest:.2f} "
            f"term={losses['terminal_outcome_mean_recent']:+.3f} "
            f"shaping={losses['mean_abs_shaping_reward_recent']:.4f} "
            f"pi={losses['policy_loss']:.4f} v={losses['value_loss']:.4f} "
            f"H={losses['entropy']:.3f} "
            f"rollout={rollout_seconds:.1f}s update={update_seconds:.1f}s "
            f"ep/h={per_hour:.0f}",
            flush=True,
        )
        state["runs"][-1]["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
        state["runs"][-1]["updates"] = update
        state["runs"][-1]["status"] = "stopped" if stop else "running"
        if run_episodes >= args.max_episodes_per_run and not stop:
            state["runs"][-1]["status"] = "run_limit"
        _save_state(state_path, state, curve_path)
        if run_episodes >= args.max_episodes_per_run and not stop:
            break

    # A run can stop early (gate pass, zero stage-0 signal, hard stop), and then the
    # last evaluation never carried the reference set.  Evaluate it now so every run
    # ends with a complete record and ``gates/T5.json`` never reports a gap.
    if latest_eval["reference_set"].get("skipped"):
        latest_eval = _evaluate(
            model, resource_dir, reference_tasks, gate_tasks, stage0_tasks,
            state["cumulative_episodes"], output_dir,
            workers=workers, worker_threads=worker_threads,
            worker_device=worker_device, include_reference=True,
        )
        state["evaluations"].append(latest_eval)
        state["learning_curve"].append(_curve_row(
            state["cumulative_episodes"], curriculum_tasks,
            latest_eval["gate_set"]["pass_rate"],
            latest_eval["reference_set"]["pass_rate"],
            latest_eval["stage0_set"]["pass_rate"],
            latest_eval["raw_seed_results_path"],
            timing=state.get("last_training_debug", {}).get("timing_seconds"),
        ))
        print("evaluated the reference set on the final model "
              f"(pass_rate={latest_eval['reference_set']['pass_rate']:.4f})", flush=True)

    if state.get("stop_reason"):
        state["runs"][-1]["status"] = state["stop_reason"]
    else:
        state["runs"][-1]["status"] = "run_limit"
    state["runs"][-1]["finished_at"] = datetime.now(timezone.utc).isoformat()
    previous_rate = run_config["previous_curve_comparison"].get(
        "previous_final_heldout_pass_rate", baseline["gate_set"]["pass_rate"]
    )
    state["runs"][-1]["curve_comparison"] = {
        "previous_curve_final_pass_rate": previous_rate,
        "current_curve_final_pass_rate": latest_eval["gate_set"]["pass_rate"],
        "change": latest_eval["gate_set"]["pass_rate"] - previous_rate,
    }
    _save_state(state_path, state, curve_path)
    if state.get("stop_reason") or state["formal_runs"] >= MAX_FORMAL_RUNS:
        protected_unchanged = all(sha256_file(path) == digest
                                  for path, digest in protected_before.items())
        reproduce_command = " && ".join((
            "python scripts/t5_throughput.py --resource-dir "
            f"{shlex.quote(str(resource_dir))} --minutes 15 --output artifacts/t5/throughput.json",
            "python " + shlex.join(sys.argv),
        ))
        document = _gate_document(
            state, throughput, baseline, latest_eval, revision, not dirty,
            protected_unchanged,
            reproduce_command,
        )
        atomic_json(ROOT / "gates/T5.json", document)
    print(json.dumps({
        "cumulative_training_episodes": state["cumulative_episodes"],
        "formal_runs": state["formal_runs"],
        "stop_reason": state.get("stop_reason"),
        "heldout_cap3_x1_pass_rate": latest_eval["gate_set"]["pass_rate"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
