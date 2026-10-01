"""Audit target-link coverage in every completed rollout of a state snapshot."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_seed_jobs import atomic_json, read_numpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    state_bytes = (args.experiment_dir / "training_state.json").read_bytes()
    state = json.loads(state_bytes)
    tasks = defaultdict(Counter)
    shards = []
    for update in state["update_history"]:
        directory = args.experiment_dir / update["shard_directory"]
        paths = sorted(p for p in directory.glob("seed_*.npz") if ".invalid_" not in p.name)
        expected = sum(update["trajectory_stats"]["task_counts"].values())
        if len(paths) != expected:
            raise ValueError(f"update {update['update']}: expected {expected} shards, got {len(paths)}")
        for path in paths:
            stored = read_numpy(path)
            if stored["metadata"]["update"] != update["update"]:
                raise ValueError(f"wrong update metadata: {path}")
            episode = stored["result"]
            counts = Counter(episodes=1, decisions=len(episode["transitions"]))
            for transition in episode["transitions"]:
                tokens = transition["tokens"]
                targets = np.asarray(tokens["target_indices"])
                kinds = np.asarray(tokens["ids"])[:, 0]
                features = np.asarray(tokens["features"])
                if targets.shape != kinds.shape or (targets < -1).any() or (targets >= len(kinds)).any():
                    raise ValueError(f"invalid target indices: {path}")
                linked = targets >= 0
                if linked.any() and (not np.isin(kinds[linked], [3, 5]).all()
                                     or not (kinds[targets[linked]] == 4).all()):
                    raise ValueError(f"invalid plant/projectile to zombie link: {path}")
                present = ((kinds == 3) & (features[:, 14] > 0)) | ((kinds == 5) & (features[:, 11] > 0))
                counts.update({"decisions_with_resolved_link": int(linked.any()),
                               "resolved_links": int(linked.sum()),
                               "decisions_with_target_reference": int(present.any()),
                               "target_references": int(present.sum()),
                               "unresolved_references": int((present & ~linked).sum()),
                               "plant_links": int((linked & (kinds == 3)).sum()),
                               "projectile_links": int((linked & (kinds == 5)).sum())})
            tasks[episode["task_id"]].update(counts)
            shards.append({"path": str(path.relative_to(args.experiment_dir)),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           "task_id": episode["task_id"], "update": update["update"], **counts})
    # Counter addition drops zero-valued keys; zero coverage must be explicit.
    keys = {key for counts in tasks.values() for key in counts}
    total = Counter({key: sum(counts[key] for counts in tasks.values()) for key in sorted(keys)})
    for key in ("episodes", "decisions"):
        if total[key] != state["counters"][key]:
            raise ValueError(f"state {key} mismatch: {total[key]} != {state['counters'][key]}")
    atomic_json(args.output, {"schema_version": 1,
                              "scope": "all episodes and all decisions of every completed update; no outcome filtering",
                              "experiment_dir": str(args.experiment_dir),
                              "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
                              "token_contract": {"plant": 3, "zombie": 4, "projectile": 5},
                              "provenance": json.loads((args.experiment_dir / "provenance.json").read_text()),
                              "counters": state["counters"], "summary": dict(total),
                              "per_task": {task: dict(counts) for task, counts in sorted(tasks.items())},
                              "shards": shards, "seconds": time.monotonic() - started,
                              "limitation": "engineering rollout coverage; absence of sampled links is not proof the task cannot exercise them"})
    print(json.dumps(dict(total)))


if __name__ == "__main__":
    main()
