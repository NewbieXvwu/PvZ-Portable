"""Run and count task90's teacher decisions without changing production defaults."""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROLLOUT = "/Users/newbiexvwu/PvZAgent/artifacts/archive/fog36/task79/models/r11/model.pt"
DEFAULT_VALUE = "/Users/newbiexvwu/PvZAgent/artifacts/archive/demo/hard/value_v5/model.pt"
DEFAULT_STATIC = "/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt"
DEFAULT_EXECUTABLE = "/tmp/pvz-boss50-build/pvz-portable"
DEFAULT_RESOURCE = "/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN"


def _read_json(path: Path):
    return json.loads(path.read_text())


def _probe_rows(path: Path) -> list[dict]:
    rows = []
    for part in sorted(path.parent.glob(path.name + ".*.jsonl")):
        rows.extend(json.loads(line) for line in part.read_text().splitlines() if line.strip())
    unique = {(row["task"], row["seed"], row["decision"]): row for row in rows}
    return sorted(unique.values(), key=lambda row: (row["task"], row["seed"], row["decision"]))


def _action_key(action: dict) -> tuple:
    if action.get("type") == "plant":
        return "plant", int(action["packet"]), int(action["row"]), int(action["col"])
    return "wait", -1, -1, -1


def _rank(action: dict, targets: list[list]) -> int | None:
    if not targets:
        return None
    key = _action_key(action)
    ordered = sorted(enumerate(targets), key=lambda item: (-float(item[1][3]), item[0]))
    for rank, (_index, target) in enumerate(ordered, 1):
        target_key = ("plant", int(target[0]), int(target[1]), int(target[2])) \
            if int(target[0]) >= 0 else ("wait", -1, -1, -1)
        if target_key == key:
            return rank
    return None


def _frame_targets(frame: dict) -> list[list]:
    return frame.get("targets") or []


def _frame_for_probe(frames: list[dict], row: dict) -> tuple[dict | None, dict | None]:
    for index, frame in enumerate(frames):
        if frame.get("tick") == row.get("tick"):
            return frame, frames[index + 1] if index + 1 < len(frames) else None
    index = min(len(frames) - 1, int(row.get("decision", 0))) if frames else -1
    return (frames[index], frames[index + 1]) if index >= 0 and index + 1 < len(frames) else (None, None)


def _decision_rows(group: dict, out: Path) -> list[dict]:
    probe = _probe_rows(out / "leaves")
    if not probe:
        return []
    from pvz_frame_store import frame_path, read_frames

    frame_cache = {}
    result = []
    conveyor = bool(group.get("conveyor"))
    for row in probe:
        seed = int(row["seed"])
        if seed not in frame_cache:
            archive = out / "frames" / f"seed_{seed}" / "task90_fast_search"
            frame_cache[seed] = read_frames(frame_path(archive))
        root, after = _frame_for_probe(frame_cache[seed], row)
        targets = _frame_targets(after) if after else []
        packets = root.get("packets") if root else []
        conveyor_packets = {int(packet[0]) for packet in packets or [] if len(packet) > 1 and int(packet[1]) >= 0}
        conveyor_targets = [target for target in targets if int(target[0]) in conveyor_packets]
        policy = json.loads(row["policy"])
        result.append({
            "task": row["task"],
            "seed": seed,
            "decision": int(row["decision"]),
            "tick": int(row["tick"]),
            "candidates": len(targets) or int(row["candidates"]),
            "scored": int(row["scored"]),
            "conveyor_candidate_count": len(conveyor_targets) if conveyor else 0,
            "conveyor_candidate_cards": len({int(target[0]) for target in conveyor_targets}) if conveyor else 0,
            "student_leaf_rank": _rank(policy, targets),
            "top1_top2": float(row["top1_top2"]),
            "std": float(row["std"]),
            "overridden": bool(row["overridden"]),
            "root_packets": [list(packet) for packet in packets or []],
            "rollout_packet_trace": row.get("rollout_packet_trace", "unrecorded"),
        })
    return result


def _describe(rows: list[dict]) -> dict:
    if not rows:
        return {"decisions": 0, "games": 0}
    ranks = [row["student_leaf_rank"] for row in rows if row["student_leaf_rank"] is not None]
    return {
        "decisions": len(rows),
        "games": len({(row["task"], row["seed"]) for row in rows}),
        "mean_candidates": round(statistics.mean(row["candidates"] for row in rows), 4),
        "mean_conveyor_candidate_count": round(statistics.mean(row["conveyor_candidate_count"] for row in rows), 4),
        "mean_conveyor_candidate_share": round(
            statistics.mean(row["conveyor_candidate_count"] / row["candidates"]
                            for row in rows if row["candidates"]), 4),
        "mean_student_leaf_rank": round(statistics.mean(ranks), 4) if ranks else None,
        "median_student_leaf_rank": statistics.median(ranks) if ranks else None,
        "mean_top1_top2": round(statistics.mean(row["top1_top2"] for row in rows), 6),
        "mean_std": round(statistics.mean(row["std"] for row in rows), 6),
        "override_rate": round(statistics.mean(row["overridden"] for row in rows), 6),
        "rollout_packet_trace": sorted({row["rollout_packet_trace"] for row in rows}),
    }


def run(args) -> None:
    plan = _read_json(args.plan)
    groups = plan.get("groups") or []
    if len(groups) != 4 or {len(group.get("seeds") or []) for group in groups} != {8}:
        raise ValueError("task90 plan must contain four groups with eight seeds each")
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    for group in groups:
        seeds = [int(seed) for seed in group["seeds"]]
        if len(set(seeds)) != 8:
            raise ValueError(f"duplicate seeds in {group['name']}")
        out = args.out / group["name"]
        out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(ROOT / "scripts/search_play.py"),
            "--rollout-policy", args.rollout_policy,
            "--value", args.value,
            "--static-value", args.static_value,
            "--manifest", group["manifest"],
            "--task-id", group["task_id"],
            "--seed-list", ",".join(map(str, seeds)),
            "--arms", "fast_search",
            "--workers", str(args.workers),
            "--verify-samples", "9",
            "--student-candidates", "2",
            "--dagger-beta", "1",
            "--executable", args.executable,
            "--resource-dir", args.resource_dir,
            "--out", str(out),
            "--frames", str(out / "frames"),
            "--frames-label", "task90",
            "--leaf-probe", str(out / "leaves"),
        ]
        print(group["name"], flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


def report(args) -> None:
    plan = _read_json(args.plan)
    result = {"protocol": "task90 step1", "groups": {}}
    for group in plan["groups"]:
        rows = _decision_rows(group, args.out / group["name"])
        decisions_path = args.out / group["name"] / "decisions.jsonl"
        decisions_path.parent.mkdir(parents=True, exist_ok=True)
        decisions_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        result["groups"][group["name"]] = {
            "task_id": group["task_id"],
            "conveyor": bool(group.get("conveyor")),
            "games": sorted({row["seed"] for row in rows}),
            "decisions": str(decisions_path),
            "summary": _describe(rows),
        }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, ensure_ascii=False, indent=1) + chr(10))
    print(json.dumps(result, ensure_ascii=False, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "report"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json", type=Path, default=ROOT / "MEMORY/docs/task90-step1-counts.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rollout-policy", default=DEFAULT_ROLLOUT)
    parser.add_argument("--value", default=DEFAULT_VALUE)
    parser.add_argument("--static-value", default=DEFAULT_STATIC)
    parser.add_argument("--executable", default=DEFAULT_EXECUTABLE)
    parser.add_argument("--resource-dir", default=DEFAULT_RESOURCE)
    args = parser.parse_args()
    run(args) if args.phase == "run" else report(args)


if __name__ == "__main__":
    main()
