"""Summarize the frozen reward queue from raw evaluation outcomes, including unfinished arms."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_seed_jobs import atomic_json


def wilson(won: int, count: int) -> list[float]:
    z = 1.959963984540054
    p, denominator = won / count, 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z * ((p * (1 - p) / count + z * z / (4 * count * count)) ** .5) / denominator
    return [max(0., center - radius), min(1., center + radius)]


def outcomes(rows: list[dict]) -> dict:
    count = len(rows)
    if not count:
        raise ValueError("empty evaluation cohort")
    won = sum(row["won"] for row in rows)
    if any(row["won"] != (row["result"] == 1) or (row["won"] and row["truncated"]) for row in rows):
        raise ValueError("inconsistent win labels")
    return {"won": won, "count": count, "win_rate": won / count,
            "wilson_95": wilson(won, count), "truncated": sum(row["truncated"] for row in rows),
            "mean_terminal_wave": sum(row["terminal_wave"] for row in rows) / count}


def summarize_evaluation(payload: dict, tasks: list[dict], modes: list[str]) -> dict:
    expected_tasks = {task["task_id"] for task in tasks}
    if set(payload["seed_results"]) != set(modes):
        raise ValueError("evaluation modes differ from frozen config")
    result = {}
    for mode, per_task in payload["seed_results"].items():
        if set(per_task) != expected_tasks:
            raise ValueError("evaluated task list differs from frozen manifest")
        cohorts, per_task_summary = defaultdict(list), {}
        for task in tasks:
            task_id = task["task_id"]
            rows = per_task[task_id]
            if len(rows) != len(task["seeds"]) or sorted(row["seed"] for row in rows) != sorted(task["seeds"]):
                raise ValueError(f"missing, repeated or substituted seeds in {mode}/{task_id}")
            per_task_summary[task_id] = outcomes(rows)
            role = task["evaluation_role"]
            cohorts[f"{role}/cap{task['wave_cap']}/x{task['zombie_count_multiplier']:g}"].extend(rows)
            if role != "training_probe":
                cohorts[f"validation/cap{task['wave_cap']}"].extend(rows)
                cohorts[f"validation/terrain/{task['terrain']}"].extend(rows)
            cohorts["all_tasks_diagnostic"].extend(rows)
        result[mode] = {"cohorts": {name: outcomes(rows) for name, rows in cohorts.items()},
                        "per_task": per_task_summary}
    return result


def format_rate(x: dict) -> str:
    lo, hi = x["wilson_95"]
    return f"{x['won']}/{x['count']} ({100*x['win_rate']:.2f}%; {100*lo:.2f}–{100*hi:.2f})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue_bytes = args.queue.read_bytes()
    queue = json.loads(queue_bytes)
    candidates = []
    common_config_digest, common_fingerprints, paired_initial_states = None, None, {}
    for entry in queue["order"]:
        config = json.loads((ROOT / entry["config"]).read_text())
        common_config = {key: value for key, value in config.items()
                         if key not in ("experiment_id", "initialization_seed", "reward")}
        common_digest = hashlib.sha256(json.dumps(common_config, sort_keys=True).encode()).hexdigest()
        if common_config_digest is not None and common_digest != common_config_digest:
            raise ValueError("reward arms differ in a non-reward, non-initialization configuration")
        common_config_digest = common_digest
        directory = ROOT / entry["output_dir"]
        state_path = directory / "training_state.json"
        candidate = {"experiment_id": config["experiment_id"], "reward": config["reward"],
                     "initialization_seed": config["initialization_seed"], "output_dir": entry["output_dir"],
                     "status": "not_started", "evaluations": []}
        candidates.append(candidate)
        if not state_path.exists():
            continue
        state_bytes = state_path.read_bytes()
        state = json.loads(state_bytes)
        provenance = json.loads((directory / "provenance.json").read_text())
        if common_fingerprints is not None and provenance["fingerprints"] != common_fingerprints:
            raise ValueError("training core, resources, simulator or tasks changed between reward candidates")
        common_fingerprints = provenance["fingerprints"]
        seed = config["initialization_seed"]
        initial_state = state["initial_state_sha256"]
        if seed in paired_initial_states and initial_state != paired_initial_states[seed]:
            raise ValueError("paired reward candidates did not begin from identical model parameters")
        paired_initial_states[seed] = initial_state
        if state["status"] == "budget_complete" and (
                state["counters"]["decisions"] < queue["common_decision_budget"] or
                not state["learning_curve"] or
                state["learning_curve"][-1]["counters"] != state["counters"]):
            raise ValueError("candidate marked complete without full budget and final evaluation")
        candidate.update(status=state["status"], phase=state["phase"], counters=state["counters"],
                         initial_state_sha256=initial_state, provenance=provenance,
                         state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                         active_wall_seconds=state["wall_seconds"], resources=state.get("resources"),
                         latest_losses=state["update_history"][-1]["losses"] if state["update_history"] else None)
        tasks = json.loads((ROOT / config["evaluation"]["manifest"]).read_text())["tasks"]
        for point in state["learning_curve"]:
            raw_path = directory / point["raw_seed_results_path"]
            with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload["experiment_id"] != config["experiment_id"] or payload["counters"] != point["counters"]:
                raise ValueError("raw evaluation identity/counter mismatch")
            evaluation = {"counters": point["counters"], "update": point["updates"],
                          "evaluation_seconds": point["seconds"], "raw_path": str(raw_path.relative_to(ROOT)),
                          "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                          "summary": summarize_evaluation(payload, tasks, config["evaluation"]["modes"])}
            # The immutable evaluation checkpoint stores actual cumulative active
            # wall time and resource peaks at that node. Loading only this existing
            # evidence avoids estimating timing from rollout sums or interpolating wins.
            checkpoint_paths = sorted((directory / "runs/run_1").glob(
                f"update_{point['updates']:06d}_evaluated_*.pt"))
            if not checkpoint_paths:
                raise ValueError("completed evaluation has no immutable checkpoint")
            import torch
            checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
            snapshot = checkpoint["training_state"]
            if snapshot["counters"] != point["counters"]:
                raise ValueError("evaluation checkpoint counters differ from raw results")
            evaluation["active_wall_seconds"] = snapshot["wall_seconds"]
            evaluation["resources"] = snapshot.get("resources")
            evaluation["checkpoint"] = str(checkpoint_paths[0].relative_to(ROOT))
            del checkpoint
            candidate["evaluations"].append(evaluation)
    completed = sum(candidate["status"] == "budget_complete" for candidate in candidates)
    all_complete = completed == len(candidates)
    atomic_json(args.output, {"schema_version": 1, "matrix": queue["matrix"],
                             "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
                             "completed_candidates": completed, "total_candidates": len(candidates),
                             "all_candidates_complete": all_complete,
                             "capability_gate": "not decided by this descriptive summary",
                             "timing": "actual cumulative active wall including startup/evaluation/checkpoint overhead",
                             "intervals": "Wilson 95%; aggregated cohorts pool task seeds and are descriptive, not initialization uncertainty",
                             "candidates": candidates})
    lines = [f"# 奖励对照 {queue['matrix']}：原始结果汇总", "",
             f"完成冻结预算的候选：{completed}/{len(candidates)}。"
             + ("全部候选已完成。" if all_complete else "矩阵尚未完成，不作奖励优胜或多数初始化学习结论。"), "",
             "每个数值均重新核对原始逐局结果、完整任务清单和环境种子。括号内为胜率及Wilson 95%区间。"
             "训练探针与验证分开；验证含历史不同倍率，分项JSON保留倍率。"
             "区间描述环境种子差异，不表示三个初始化的不确定性。", "",
             "墙钟为各节点真实累计活动耗时，包含启动、评估与保存。共同交互节点和耗时曲线都保留，"
             "没有把不同墙钟的胜率称作等耗时比较。", "",
             "| 奖励 / 初始化 | 实际决策 | 活动秒 | 策略 | 验证cap1 | 验证cap3 | 验证cap5 |",
             "|---|---:|---:|---|---|---|---|"]
    for candidate in candidates:
        for point in candidate["evaluations"]:
            for mode, result in point["summary"].items():
                cells = [format_rate(result["cohorts"][f"validation/cap{cap}"]) for cap in (1, 3, 5)]
                lines.append(f"| {candidate['reward']['name']} / {candidate['initialization_seed']} | "
                             f"{point['counters']['decisions']} | {point['active_wall_seconds']:.1f} | {mode} | "
                             + " | ".join(cells) + " |")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summarized {len(candidates)} candidates; budget complete {completed}", flush=True)


if __name__ == "__main__":
    main()
