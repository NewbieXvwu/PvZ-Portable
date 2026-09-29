"""Train PPO on the frozen 20-task family and evaluate with the frozen T4 runner."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
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
from pvz_seed_jobs import atomic_json, atomic_write, run_seed_jobs, seed_job_directory  # noqa: E402
from pvz_value import SEARCH_LABEL_VERSION, VALUE_SEMANTICS  # noqa: E402
from train_pvz_ppo import add_advantages, collect_task_episode, episode_hash, train_update  # noqa: E402
import check_task_manifests  # noqa: E402
import t4_capability_profile  # noqa: E402

TRAIN_PATH = ROOT / "artifacts/task_family/train.json"
HELDOUT_PATH = ROOT / "artifacts/task_family/heldout.json"
T4_GATE_PATH = ROOT / "gates/T4.json"
T4_RAW_PATH = ROOT / "artifacts/task_family/t4_seed_results.json"
THROUGHPUT_PATH = ROOT / "artifacts/t5/throughput.json"
DEFAULT_RESOURCE_DIR = Path.home() / ".cache/pvz-research-resources"
CORE_THRESHOLD = 5000.0
WIN_THRESHOLD = 0.90
MAX_FORMAL_RUNS = 8
T4_MODEL_ARCHITECTURE_VERSION = 4
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
                 worker_threads: int, worker_device: str) -> None:
    global WORKER_MODEL, WORKER_ENV, WORKER_ASSIGNMENTS, WORKER_MAX_ACTIONS
    configure_torch_threads(worker_threads)
    WORKER_MODEL = GameplayModelV1().eval().to(resolve_device(worker_device))
    WORKER_MODEL.load_state_dict(state_dict)
    WORKER_ENV = PvZEnv(resource_dir=resource_dir)
    WORKER_ASSIGNMENTS = assignments
    WORKER_MAX_ACTIONS = max_actions
    Finalize(None, _close_worker, exitpriority=10)


def _rollout_worker(job_id: int) -> dict[str, Any]:
    if WORKER_MODEL is None or WORKER_ENV is None:
        raise RuntimeError("PPO rollout worker was not initialized")
    assignment = WORKER_ASSIGNMENTS[job_id]
    torch.manual_seed(assignment["action_seed"])
    return collect_task_episode(
        WORKER_MODEL,
        WORKER_ENV,
        assignment["task"],
        assignment["task_seed"],
        job_id,
        WORKER_MAX_ACTIONS,
    )


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


def _evaluate(model: GameplayModelV1, resource_dir: Path, tasks: list[dict[str, Any]],
              gate_tasks: list[dict[str, Any]], stage0_tasks: list[dict[str, Any]],
              episodes: int, output_dir: Path) -> dict[str, Any]:
    eval_model = GameplayModelV1().eval()
    eval_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    records: dict[str, list[dict[str, Any]]] = {}
    stage0_records: dict[str, list[dict[str, Any]]] = {}
    configure_torch_threads(1)
    with PvZEnv(resource_dir=resource_dir) as env:
        for task in tasks:
            records[task["task_id"]] = [
                t4_capability_profile.run_episode(env, task, seed, "checkpoint", eval_model)
                for seed in task["seeds"]
            ]
        for task in stage0_tasks:
            stage0_records[task["task_id"]] = [
                t4_capability_profile.run_episode(env, task, seed, "checkpoint", eval_model)
                for seed in task["seeds"]
            ]
    gate_ids = {task["task_id"] for task in gate_tasks}
    gate_records = [record for task_id, rows in records.items() if task_id in gate_ids for record in rows]
    reference_records = [record for rows in records.values() for record in rows]
    gate_summary = t4_capability_profile.summarize_episodes(gate_records)
    reference_summary = t4_capability_profile.summarize_episodes(reference_records)
    stage0_records_flat = [record for rows in stage0_records.values() for record in rows]
    stage0_summary = t4_capability_profile.summarize_episodes(stage0_records_flat)
    raw_path = output_dir / "evaluations" / f"heldout_{episodes:07d}.json.gz"
    atomic_json(raw_path, {
        "cumulative_episodes": episodes,
        "seed_results": records,
        "stage0_seed_results": stage0_records,
    }, compressed=True)
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
        "reference_set": {
            "task_count": len(tasks),
            "sample_count": reference_summary["sample_count"],
            "passes": round(reference_summary["pass_rate"] * reference_summary["sample_count"]),
            "pass_rate": reference_summary["pass_rate"],
            "per_task": {
                task_id: t4_capability_profile.summarize_episodes(rows)
                for task_id, rows in records.items()
            },
        },
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
        "raw_seed_results_path": str(raw_path.relative_to(ROOT)),
    }


def _curve_row(episodes: int, train_tasks: list[dict[str, Any]],
               gate_rate: float, reference_rate: float, stage0_rate: float,
               source: str) -> dict[str, Any]:
    return {
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


def _save_checkpoint(path: Path, model: GameplayModelV1, config: dict[str, Any], provenance: dict[str, Any],
                     run_number: int, update: int, losses: dict[str, float]) -> None:
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_architecture_version": MODEL_ARCHITECTURE_VERSION,
        "value_semantics": VALUE_SEMANTICS,
        "config": MODEL_CONFIG,
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
                                  note: str | None) -> dict[str, Any] | None:
    if actual_hash == t4_hash:
        return None
    if MODEL_ARCHITECTURE_VERSION > T4_MODEL_ARCHITECTURE_VERSION:
        if not note:
            raise RuntimeError("网络结构已变更，seed-0 初始化不再与 T4 基线一致；"
                               "请传入 --initialization-note 说明 T4 基线为何已被取代")
        return {"status": "superseded", "actual": actual_hash, "t4": t4_hash, "note": note}
    raise RuntimeError("seed-0 initialization does not match the T4 baseline checkpoint")


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
                        help="auto uses exact FlexAttention for large CUDA update batches")
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--initialization-seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
        ROOT / "DESIGN.md", ROOT / "FAILURE_ANALYSIS.md", TRAIN_PATH, HELDOUT_PATH,
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
        model_hash = t4_capability_profile._state_sha256(model_state)
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
        add_advantages(episodes, args.gae_lambda)
        losses = train_update(
            model, episodes, optimizer, device, args.ppo_epochs, args.sequence_length,
            args.clip_epsilon, args.value_coefficient, args.entropy_coefficient,
            minibatch_chunks=args.minibatch_chunks,
            attention_backend=args.attention_backend,
        )
        update += 1
        run_episodes += len(episodes)
        state["cumulative_episodes"] += len(episodes)
        hashes = [episode_hash(episode) for episode in episodes]
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
            "losses": losses,
        }

        checkpoint_path = run_dir / "gameplay_model_v1_ppo.pt"
        if (state["cumulative_episodes"] % 5000 == 0
                or time.monotonic() - last_curve_time >= CURVE_SECONDS
                or run_episodes >= args.max_episodes_per_run):
            eval_row = _evaluate(
                model, resource_dir, reference_tasks, gate_tasks,
                stage0_tasks,
                state["cumulative_episodes"], output_dir,
            )
            latest_eval = eval_row
            state["evaluations"].append(eval_row)
            state["learning_curve"].append(_curve_row(
                state["cumulative_episodes"], curriculum_tasks,
                eval_row["gate_set"]["pass_rate"], eval_row["reference_set"]["pass_rate"],
                eval_row["stage0_set"]["pass_rate"],
                eval_row["raw_seed_results_path"],
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
        _save_checkpoint(checkpoint_path, model, run_config, provenance, run_number, update, losses)
        print(
            f"run={run_number} update={update} episodes={state['cumulative_episodes']} "
            f"wins={losses['rollout_episode_wins']}/{len(episodes)} "
            f"policy_loss={losses['policy_loss']:.4f} value_loss={losses['value_loss']:.4f} "
            f"entropy={losses['entropy']:.3f}",
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
