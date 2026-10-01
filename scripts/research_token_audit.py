"""Read-only token diagnostics; synthetic field probes are not simulator counterfactuals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
from pvz_agent_model import TOKEN_KINDS, observation_tokens
from pvz_seed_jobs import atomic_json
from test_agent_model import observation


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_probe(base: dict, changed: dict, field: str, scope: str) -> dict:
    before, before_meta = observation_tokens(base)
    after, after_meta = observation_tokens(changed)
    differences = {key: int(torch.count_nonzero(before[key] != after[key])) for key in before}
    return {"field": field, "scope": scope, "tensor_differences": differences,
            "metadata_equal": before_meta == after_meta,
            "encoded_input_equal": not any(differences.values()) and before_meta == after_meta}


def mapping_probes() -> list[dict]:
    base = observation()
    base["zombies"][0]["on_board"] = True
    changed = copy.deepcopy(base)
    changed["zombies"][0]["on_board"] = False
    rows = [synthetic_probe(base, changed, "zombie.on_board: true -> false",
                           "synthetic encoding test; no claim this identical-position pair is reachable")]
    for kind in ("plants", "projectiles"):
        base = observation()
        base[kind][0]["target_zombie_id"] = 101
        changed = copy.deepcopy(base)
        changed[kind][0]["target_zombie_id"] = 202
        rows.append(synthetic_probe(base, changed, f"{kind}.target_zombie_id: 101 -> 202",
                                    "synthetic positive-ID test; target entities are not resolved"))
    for field, value in (("x", 660.0), ("velocity_x", -0.8), ("helm_health", 80)):
        base = observation()
        changed = copy.deepcopy(base)
        changed["zombies"][0][field] = value
        rows.append(synthetic_probe(base, changed, f"zombie.{field}",
                                    "positive control: encoded input should change"))
    base = observation()
    changed = copy.deepcopy(base)
    changed["packets"][0]["cooldown"] = 1200
    rows.append(synthetic_probe(base, changed, "packet.cooldown",
                                "positive control: encoded input should change"))
    if not all(row["encoded_input_equal"] for row in rows[:3]):
        raise ValueError("field omission changed; inspect and version this audit")
    if any(row["encoded_input_equal"] for row in rows[3:]):
        raise ValueError("positive control failed")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.output.exists():
        raise ValueError("positive episode count and a new evidence path are required")
    torch.set_num_threads(1)
    started = time.monotonic()
    state_path = args.experiment_dir / "training_state.json"
    state = json.loads(state_path.read_text())
    directory = args.experiment_dir / state["update_history"][0]["shard_directory"]
    paths = sorted((p for p in directory.glob("seed_*.npz") if ".invalid_" not in p.name),
                   key=lambda p: int(p.stem.split("_")[1]))[:args.episodes]
    if len(paths) != args.episodes:
        raise ValueError("first complete update has too few episodes")
    inverse_kinds = {number: name for name, number in TOKEN_KINDS.items()}
    decisions, kind_totals, task_totals, sources = [], Counter(), Counter(), []
    feature_counts, feature_clipped = Counter(), Counter()
    ranges = defaultdict(lambda: [float("inf"), float("-inf")])
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            episode = json.loads(archive["__manifest__"].tobytes())["value"]["result"]
            task_totals[episode["task_id"]] += 1
            sources.append({"path": str(path.relative_to(args.experiment_dir)),
                            "sha256": digest(path), "job_id": episode["seed"],
                            "task_id": episode["task_id"], "environment_seed": episode["task_seed"]})
            for transition in episode["transitions"]:
                packed = transition["tokens"]
                ids = archive[packed["ids"]["__pvz_array__"]]
                features = archive[packed["features"]["__pvz_array__"]].astype(np.float32)
                counts = Counter(inverse_kinds[int(k)] for k in ids[:, 0])
                kind_totals.update(counts)
                decisions.append({"job_id": episode["seed"], "task_id": episode["task_id"],
                                  "decision_index": transition["decision_index"],
                                  "tokens": len(ids), "kind_counts": dict(counts)})
                for kind in counts:
                    selected = features[ids[:, 0] == TOKEN_KINDS[kind]]
                    for index in range(selected.shape[1]):
                        key = (kind, index)
                        column = selected[:, index]
                        feature_counts[key] += len(column)
                        feature_clipped[key] += int((abs(column) >= 2).sum())
                        ranges[key][0] = min(ranges[key][0], float(column.min()))
                        ranges[key][1] = max(ranges[key][1], float(column.max()))
    token_counts = np.array([row["tokens"] for row in decisions])
    raw_path = args.output.with_suffix(".decisions.json.gz")
    if raw_path.exists():
        raise ValueError("raw evidence path already exists")
    atomic_json(raw_path, decisions, compressed=True)
    atomic_json(args.output, {
        "schema_version": 1, "experiment_id": state["experiment_id"],
        "scope": "first N job IDs from first complete rollout update, selected before reading outcomes",
        "limitations": ["short cap1 episodes only; no full-level token or memory claim",
                        "packed tokens cannot recover discarded on_board/target identity fields",
                        "synthetic probes demonstrate encoding omissions, not unavoidable policy failure",
                        "boundary hits count abs(feature)>=2 after float16 storage; not all imply harmful clipping",
                        "token counts alone do not measure attention wall time or allocator memory"],
        "encoder_sha256": digest(ROOT / "python/pvz_agent_model.py"),
        "native_public_schema_sha256": digest(ROOT / "src/LawnApp.cpp"),
        "source_state_sha256": digest(state_path), "sources": sources,
        "episodes": len(paths), "decisions": len(decisions), "episodes_by_task": dict(task_totals),
        "token_count": {"min": int(token_counts.min()), "median": float(np.median(token_counts)),
                        "p95": float(np.percentile(token_counts, 95)), "max": int(token_counts.max())},
        "kind_totals": dict(kind_totals),
        "cell_fraction_of_tokens": kind_totals["cell"] / int(token_counts.sum()),
        "dense_attention_token_pair_count": int((token_counts ** 2).sum()),
        "features": [{"kind": kind, "index": index, "count": count,
                      "abs_ge_2_count": feature_clipped[kind, index],
                      "range": ranges[kind, index]}
                     for (kind, index), count in sorted(feature_counts.items())],
        "synthetic_field_probes": mapping_probes(), "raw_decisions_path": str(raw_path),
        "wall_seconds": time.monotonic() - started,
        "cost_scope": "bounded CPU-only offline diagnostic, concurrent with reward matrix; no GPU work"})
    print(json.dumps({"episodes": len(paths), "decisions": len(decisions),
                      "median_tokens": float(np.median(token_counts)), "max_tokens": int(token_counts.max()),
                      "seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
