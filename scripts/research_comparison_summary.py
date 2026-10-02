"""Summarize frozen reward/model trials from their actual evaluation evidence."""
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


def storage_revision(queue: dict) -> dict | None:
    """Validate the preregistered storage-only continuation, without hiding drift."""
    revision = queue.get("storage_revision")
    if revision is None:
        return None
    audit_bytes = (ROOT / revision["audit"]).read_bytes()
    original_bytes = (ROOT / revision["original_queue"]).read_bytes()
    if (hashlib.sha256(audit_bytes).hexdigest() != revision["audit_sha256"]
            or hashlib.sha256(original_bytes).hexdigest() != revision["original_queue_sha256"]):
        raise ValueError("storage audit or original queue changed after preregistration")
    audit, original = json.loads(audit_bytes), json.loads(original_bytes)
    if (audit.get("gate_result") != "pass" or not audit["remaining_executable_ast_exact"]
            or audit["trained_keep"] < 8):
        raise ValueError("storage continuation requires a passed executable-equivalence audit")
    if len(queue["order"]) != len(original["order"]):
        raise ValueError("storage continuation changed candidate count")
    affected = set(revision["candidates"])
    found = set()
    for old_entry, entry in zip(original["order"], queue["order"]):
        old = json.loads((ROOT / old_entry["config"]).read_text())
        current = json.loads((ROOT / entry["config"]).read_text())
        expected = json.loads(json.dumps(old))
        if old["experiment_id"] in affected:
            found.add(old["experiment_id"])
            expected["prerequisites"] = [revision["new_gate"] if p == revision["old_gate"] else p
                                         for p in expected["prerequisites"]]
        if current != expected or any(entry[k] != old_entry[k] for k in ("output_dir", "log")):
            raise ValueError("storage continuation changed a frozen candidate beyond its prerequisite")
    if found != affected:
        raise ValueError("unknown candidate in storage continuation")
    return audit


def comparable_fingerprints(fingerprints: dict, audit: dict | None) -> dict:
    result = dict(fingerprints)
    if audit is not None:
        key = "python/pvz_research.py"
        if result[key] not in (audit["old_source_sha256"], audit["new_source_sha256"]):
            raise ValueError("unapproved research source in storage continuation")
        result[key] = audit["old_source_sha256"]
    return result


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


def summarize_evaluation(payload: dict, tasks: list[dict], modes: list[str],
                         default_role: str | None = None) -> dict:
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
            role = task.get("evaluation_role", default_role)
            if role is None:
                raise ValueError("unlabelled evaluation task requires an explicit manifest split")
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


def common_config(config: dict, axis: str) -> dict:
    """Allow the named comparison axis, without accepting other recipe changes."""
    if axis == "reward":
        excluded = {"experiment_id", "initialization_seed", "reward"}
    elif axis == "model":
        if "initialization" in config:
            raise ValueError("fresh model comparison cannot mix transferred initializations")
        excluded = {"experiment_id", "initialization_seed", "purpose", "model"}
    else:
        raise ValueError("comparison axis must be reward or model")
    return {key: value for key, value in config.items() if key not in excluded}


def validate_model_checkpoint(checkpoint: dict, config: dict, provenance: dict,
                              state: dict, payload: dict) -> None:
    if (checkpoint['experiment_config'] != config
            or checkpoint['experiment_identity'] != state['experiment_identity']
            or checkpoint['training_state']['experiment_identity'] != state['experiment_identity']
            or checkpoint['training_state']['experiment_id'] != config['experiment_id']
            or checkpoint['config'] != provenance['model_config']
            or checkpoint['config'] != payload['model_config']):
        raise ValueError('model evaluation checkpoint belongs to another trial or model')


def budget_views(candidates: list[dict], decision_nodes: list[int],
                 wall_budgets: list[float]) -> dict:
    """Reference real measured points; never interpolate or invent equal-time scores."""
    if decision_nodes != sorted(set(decision_nodes)) or any(node < 0 for node in decision_nodes):
        raise ValueError("decision views require sorted distinct nonnegative nodes")
    if wall_budgets != sorted(set(wall_budgets)) or any(wall <= 0 for wall in wall_budgets):
        raise ValueError("wall views require sorted distinct positive budgets")
    interactions, walls = [], []
    for node in decision_nodes:
        observed, missing = [], []
        for candidate in candidates:
            points = [(index, point) for index, point in enumerate(candidate["evaluations"])
                      if point["counters"]["decisions"] == node]
            if len(points) > 1:
                raise ValueError("candidate has duplicate evaluation decision nodes")
            if not points:
                missing.append(candidate["experiment_id"])
                continue
            index, point = points[0]
            observed.append(dict(experiment_id=candidate["experiment_id"],
                                 initialization_seed=candidate["initialization_seed"],
                                 evaluation_index=index, actual_active_wall_seconds=point["active_wall_seconds"]))
        interactions.append(dict(decisions=node, all_candidates_observed=not missing,
                                 observed=observed, missing=missing))
    for wall in wall_budgets:
        observed, missing = [], []
        for candidate in candidates:
            eligible = [(index, point) for index, point in enumerate(candidate["evaluations"])
                        if point["active_wall_seconds"] <= wall]
            if not eligible:
                missing.append(candidate["experiment_id"])
                continue
            index, point = max(eligible, key=lambda item: (item[1]["active_wall_seconds"],
                                                          item[1]["counters"]["decisions"]))
            observed.append(dict(experiment_id=candidate["experiment_id"],
                                 initialization_seed=candidate["initialization_seed"], evaluation_index=index,
                                 actual_decisions=point["counters"]["decisions"],
                                 actual_active_wall_seconds=point["active_wall_seconds"],
                                 unused_budget_seconds=wall-point["active_wall_seconds"]))
        walls.append(dict(wall_budget_seconds=wall, all_candidates_observed=not missing,
                          observed=observed, missing=missing))
    return dict(interaction_points=interactions, wall_budget_views=walls,
                wall_scope="Latest actually evaluated immutable checkpoint at or before each shared active-wall cap; actual spent time and unused cap are explicit. No interpolated score, exact equal-time observation, or extra evaluation is claimed. Freeze useful caps from reference costs before architecture GPU trials.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--comparison-axis", choices=("reward", "model"), default="reward")
    args = parser.parse_args()
    queue_bytes = args.queue.read_bytes()
    queue = json.loads(queue_bytes)
    storage_audit = storage_revision(queue)
    candidates = []
    common_config_digest, common_fingerprints, paired_initial_states = None, None, {}
    for entry in queue["order"]:
        config = json.loads((ROOT / entry["config"]).read_text())
        comparable_config = common_config(config, args.comparison_axis)
        if storage_audit is not None:
            # All configuration fields were checked against their original above.
            comparable_config.pop("prerequisites")
        common_digest = hashlib.sha256(json.dumps(comparable_config, sort_keys=True).encode()).hexdigest()
        if common_config_digest is not None and common_digest != common_config_digest:
            raise ValueError("candidates differ outside the declared comparison axis and initialization seed")
        common_config_digest = common_digest
        directory = ROOT / entry["output_dir"]
        state_path = directory / "training_state.json"
        candidate = {"experiment_id": config["experiment_id"], "reward": config["reward"],
                     "model": config["model"],
                     "initialization_seed": config["initialization_seed"], "output_dir": entry["output_dir"],
                     "status": "not_started", "evaluations": []}
        candidates.append(candidate)
        if not state_path.exists():
            continue
        state_bytes = state_path.read_bytes()
        state = json.loads(state_bytes)
        provenance = json.loads((directory / "provenance.json").read_text())
        if json.loads((directory / "experiment_config.json").read_text()) != config:
            raise ValueError("saved experiment config differs from the preregistered candidate")
        comparison_fingerprints = comparable_fingerprints(provenance["fingerprints"], storage_audit)
        if common_fingerprints is not None and comparison_fingerprints != common_fingerprints:
            raise ValueError("training core, resources, simulator or tasks changed between candidates")
        common_fingerprints = comparison_fingerprints
        seed = config["initialization_seed"]
        initial_state = state["initial_state_sha256"]
        if (args.comparison_axis == "reward" and seed in paired_initial_states
                and initial_state != paired_initial_states[seed]):
            raise ValueError("paired reward candidates did not begin from identical model parameters")
        paired_initial_states[seed] = initial_state
        if args.comparison_axis == "model" and (
                state.get("initialization_provenance") or not state.get("invocations")
                or state["invocations"][0]["kind"] != "random_initialization"):
            raise ValueError("model comparison requires each trial's actual random initialization")
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
        evaluation_manifest = json.loads((ROOT / config["evaluation"]["manifest"]).read_text())
        tasks = evaluation_manifest["tasks"]
        for point in state["learning_curve"]:
            raw_path = directory / point["raw_seed_results_path"]
            with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload["experiment_id"] != config["experiment_id"] or payload["counters"] != point["counters"]:
                raise ValueError("raw evaluation identity/counter mismatch")
            evaluation = {"counters": point["counters"], "update": point["updates"],
                          "evaluation_seconds": point["seconds"], "raw_path": str(raw_path.relative_to(ROOT)),
                          "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                          "summary": summarize_evaluation(payload, tasks, config["evaluation"]["modes"],
                                                           evaluation_manifest.get("split"))}
            if config["evaluation"].get("idle_baseline"):
                evaluation["paired_idle"] = {
                    mode: point["summary"][mode]["paired_idle"] for mode in config["evaluation"]["modes"]}
            # The immutable evaluation checkpoint stores actual cumulative active
            # wall time and resource peaks at that node. Loading only this existing
            # evidence avoids estimating timing from rollout sums or interpolating wins.
            checkpoint_paths = sorted((directory / "runs/run_1").glob(
                f"update_{point['updates']:06d}_evaluated_*.pt"))
            if not checkpoint_paths:
                raise ValueError("completed evaluation has no immutable checkpoint")
            import torch
            checkpoint = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
            if args.comparison_axis == 'model':
                validate_model_checkpoint(checkpoint, config, provenance, state, payload)
            snapshot = checkpoint["training_state"]
            if snapshot["counters"] != point["counters"]:
                raise ValueError("evaluation checkpoint counters differ from raw results")
            evaluation["active_wall_seconds"] = snapshot["wall_seconds"]
            evaluation["resources"] = snapshot.get("resources")
            evaluation["checkpoint"] = str(checkpoint_paths[0].relative_to(ROOT))
            del checkpoint
            candidate["evaluations"].append(evaluation)
    if args.comparison_axis == "model":
        for candidate in candidates:
            candidate['common_budget_evaluated'] = any(
                point['counters']['decisions'] == queue['common_decision_budget']
                for point in candidate['evaluations'])
        completed = sum(candidate['common_budget_evaluated'] for candidate in candidates)
    else:
        completed = sum(candidate["status"] == "budget_complete" for candidate in candidates)
    all_complete = completed == len(candidates)
    decision_nodes = sorted({0} | {point["counters"]["decisions"]
                                  for candidate in candidates for point in candidate["evaluations"]})
    views = budget_views(candidates, decision_nodes, queue.get("common_wall_budgets_seconds", []) or [])
    atomic_json(args.output, {"schema_version": 1, "matrix": queue["matrix"],
                             "comparison_axis": args.comparison_axis,
                             "queue_scope": queue.get("purpose"),
                             "common_decision_budget": queue["common_decision_budget"],
                             "budget_B": queue.get("budget_B"),
                             "queue_sha256": hashlib.sha256(queue_bytes).hexdigest(),
                             "completed_candidates": completed, "total_candidates": len(candidates),
                             "all_candidates_complete": all_complete,
                             "completion_scope": ("shared exact interaction node evaluated; process status separately recorded"
                                                  if args.comparison_axis == "model" else "declared final budget_complete status"),
                             "storage_revision": queue.get("storage_revision"),
                             "capability_gate": "not decided by this descriptive summary",
                             "timing": "actual cumulative active wall including startup/evaluation/checkpoint overhead",
                             "intervals": "Wilson 95%; aggregated cohorts pool task seeds and are descriptive, not initialization uncertainty",
                             "candidates": candidates, "budget_views": views})
    kind = "奖励" if args.comparison_axis == "reward" else "模型"
    completion = (f"共同预算{queue['common_decision_budget']}决策节点已评估：{completed}/{len(candidates)}。"
                  if args.comparison_axis == "model" else f"完成冻结预算的候选：{completed}/{len(candidates)}。")
    lines = [f"# {kind}对照 {queue['matrix']}：原始结果汇总", "",
             completion + ("共同预算证据完整，进程是否已停见各状态。" if all_complete
                           else "矩阵尚未完成，不作候选优胜或多数初始化学习结论。"), "",
             "每个数值均重新核对原始逐局结果、完整任务清单和环境种子。括号内为胜率及Wilson 95%区间。"
             "训练探针与验证分开；验证含历史不同倍率，分项JSON保留倍率。"
             "区间描述环境种子差异，不表示三个初始化的不确定性。", "",
             "墙钟为各节点真实累计活动耗时，包含启动、评估与保存。共同交互节点和耗时曲线都保留，"
             "没有把不同墙钟的胜率称作精确等耗时比较。JSON耗时视图仅选预算以内最近真实评估点，"
             "同时记录实际用时与未用预算，不插值、不补跑。", ""]
    caps = sorted({name.removeprefix("validation/cap") for candidate in candidates
                   for point in candidate["evaluations"] for result in point["summary"].values()
                   for name in result["cohorts"] if name.startswith("validation/cap")},
                  key=lambda cap: (cap == "None", int(cap) if cap != "None" else 0))
    if caps:
        lines.extend(["波数分组仅作诊断；援助与原始完整任务的分项保留在JSON，不能用合并capNone作正式完整关卡验收。", "",
                      "| 候选 / 初始化 | 实际决策 | 活动秒 | 策略 | "
                      + " | ".join(f"验证cap{cap}" for cap in caps) + " |",
                      "|---|---:|---:|---|" + "---|" * len(caps)])
    else:
        lines.append("尚无完成的逐局评估；不生成成绩表。")
    for candidate in candidates:
        for point in candidate["evaluations"]:
            for mode, result in point["summary"].items():
                cells = [format_rate(result["cohorts"][f"validation/cap{cap}"])
                         if f"validation/cap{cap}" in result["cohorts"] else "—" for cap in caps]
                label = candidate['reward']['name'] if args.comparison_axis == "reward" else candidate['experiment_id']
                lines.append(f"| {label} / {candidate['initialization_seed']} | "
                             f"{point['counters']['decisions']} | {point['active_wall_seconds']:.1f} | {mode} | "
                             + " | ".join(cells) + " |")
    if views['wall_budget_views'] and caps:
        lines.extend(["", "| 活动秒预算 | 候选 / 初始化 | 已评估决策 | 实际活动秒 | 未用预算秒 | 策略 | "
                      + " | ".join(f"验证cap{cap}" for cap in caps) + " |",
                      "|---:|---|---:|---:|---:|---|" + "---|" * len(caps)])
        indexed = {candidate['experiment_id']: candidate for candidate in candidates}
        for view in views['wall_budget_views']:
            for row in view['observed']:
                point = indexed[row['experiment_id']]['evaluations'][row['evaluation_index']]
                for mode, result in point['summary'].items():
                    cells = [format_rate(result['cohorts'][f'validation/cap{cap}'])
                             if f'validation/cap{cap}' in result['cohorts'] else '—' for cap in caps]
                    lines.append(f"| {view['wall_budget_seconds']:g} | {row['experiment_id']} / {row['initialization_seed']} | "
                                 f"{row['actual_decisions']} | {row['actual_active_wall_seconds']:.1f} | {row['unused_budget_seconds']:.1f} | "
                                 f"{mode} | " + " | ".join(cells) + " |")
            if view['missing']:
                lines.extend(["", f"{view['wall_budget_seconds']:g}秒视图缺少评估：" + ", ".join(view['missing']) + "。"])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summarized {len(candidates)} candidates; budget complete {completed}", flush=True)


if __name__ == "__main__":
    main()
