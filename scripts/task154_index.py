"""task154 step 1: index the qc frame archives (read-only) against the queue's own log.

The queue state (~/pvz-logs/curriculum_state.json) already records, per round and group,
the frame label, task id, student and outcome bookkeeping. This tool only reads it, plus each
episode's small meta.json, and writes one JSON index. Nothing under the pipeline is written.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES = Path.home() / "pvz-logs/curriculum_state.pipeline/frames"
DEFAULT_STATE = Path.home() / "pvz-logs/curriculum_state.json"


def load_tasks(state: dict) -> tuple[dict, dict]:
    """label -> (round, group, task); task metadata from every manifest the queue touched."""
    manifests: dict[str, dict] = {}
    labels: dict[str, dict] = {}
    for entry in state["log"]:
        for group in entry["groups"]:
            path = group.get("manifest")
            if path and path not in manifests:
                try:
                    document = json.loads(Path(path).read_text())
                except (OSError, ValueError):
                    document = {"tasks": []}
                manifests[path] = {task["task_id"]: task for task in document.get("tasks", [])}
            label = group.get("frame_label") or group["label"] + "_fast_search"
            labels[label] = {"round": entry["round"], "group": group,
                             "task": manifests.get(path, {}).get(group.get("task_id"), {})}
    return labels, manifests


def episode_meta(path: Path) -> dict | None:
    try:
        meta = json.loads((path / "meta.json").read_text())
    except (OSError, ValueError):
        return None
    context = meta.get("context") or {}
    outcome = meta.get("outcome") or {}
    task = meta.get("task") or {}
    return {
        "seed": task.get("seed"),
        "policy": task.get("policy"),
        "arm": task.get("arm"),
        "level": context.get("level"),
        "multiplier": context.get("multiplier"),
        "roster": context.get("roster"),
        "won": outcome.get("won"),
        "final_wave": outcome.get("final_wave"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()

    labels, _manifests = load_tasks(json.loads(args.state.read_text()))
    jobs: list[tuple[str, str, Path]] = []
    for seed_dir in sorted(os.listdir(args.frames)):
        base = args.frames / seed_dir
        if not seed_dir.startswith("seed_") or not base.is_dir():
            continue
        for label in os.listdir(base):
            if (base / label).is_dir():
                jobs.append((seed_dir, label, base / label))

    def build(job) -> dict | None:
        seed_dir, label, path = job
        meta = episode_meta(path)
        if meta is None:
            return None
        info = labels.get(label)
        entry = dict(meta, label=label, dir=f"{seed_dir}/{label}")
        if info is None:
            entry.update(round=None, task_id=None)
            return entry
        group, task = info["group"], info["task"]
        entry.update(
            round=info["round"],
            task_id=group.get("task_id"),
            deck=task.get("deck"),
            terrain=task.get("terrain"),
            task_multiplier=task.get("zombie_count_multiplier"),
            student=group.get("student"),
            student_round=group.get("student_round"),
            beta=group.get("beta"),
            status=group.get("status"),
            manifest=group.get("manifest"),
            start_cell=bool(group.get("demo_archive")),
            demo_archive=group.get("demo_archive"),
            demo_decision=group.get("demo_decision"),
        )
        return entry

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        entries = [entry for entry in pool.map(build, jobs) if entry is not None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"frames": str(args.frames), "episodes": entries}))
    rounds: dict = {}
    for entry in entries:
        rounds[entry["round"]] = rounds.get(entry["round"], 0) + 1
    print(f"{len(entries)} episodes")
    print("mapped:", sum(1 for entry in entries if entry.get("task_id")))
    print("rounds:", sorted((r, c) for r, c in rounds.items() if r is not None)[-10:])


if __name__ == "__main__":
    main()
