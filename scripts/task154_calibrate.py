"""task154 step 2: is the production leaf value net (value_net/v5) still calibrated?

Reads the qc frame archives (read-only), encodes evenly spaced mid-game decisions with the same
encoder the teacher uses, scores them with v5, and reports per-group calibration: Brier, a 10-bin
reliability curve and the ranking concordance (AUC) of predictions against the game's real result.

Groups: terrain (from the level background), zombie-count multiplier, whether the deck is one the
v5 corpus ever saw, and whether the episode started from a recorded demo start cell.

    python3 scripts/task154_calibrate.py --index artifacts/task154/index.json \
        --min-round 36 --out artifacts/task154/calibration.json --rows artifacts/task154/rows.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import pvz_constants  # noqa: E402
from pvz_frame_store import read_frames  # noqa: E402
from pvz_value_net import (ENCODER_V1, ENCODER_V2, ENCODER_V3, ENCODER_V4,  # noqa: E402
                           ENCODER_V5, context_vector, encode_frame, load_value_net)

DEFAULT_VALUE = Path("/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt")

# Decks the v5 corpus was built from: the pre-2026-10-24 day / night / pool panels
# (MEMORY/docs/task43-value-v5-pool.md, MEMORY/docs/task78-value-v6.md: v5 has no fog, no roof).
LEGACY_PANELS = [
    "artifacts/stage4/day_panel.json",
    "artifacts/stage4/day_panel_d7q.json",
    "artifacts/stage4/day_panel_identity.json",
    "artifacts/stage4/night_panel.json",
    "artifacts/stage4/night_panel_n8.json",
    "artifacts/stage4/night_panel_n9.json",
    "artifacts/stage4/night_panel_n10.json",
    "artifacts/stage4/night_panel_n10p.json",
    "artifacts/stage4/night_panel_n10q.json",
    "artifacts/stage4/pool_panel.json",
]

TERRAIN_NAMES = {
    "BACKGROUND_1_DAY": "day",
    "BACKGROUND_2_NIGHT": "night",
    "BACKGROUND_3_POOL": "pool",
    "BACKGROUND_4_FOG": "fog",
    "BACKGROUND_5_ROOF": "roof",
    "BACKGROUND_6_BOSS": "boss",
}


def legacy_decks() -> set[tuple[int, ...]]:
    decks: set[tuple[int, ...]] = set()
    for relative in LEGACY_PANELS:
        path = Path("/home/newbiexvwu/PvZ-search") / relative
        try:
            document = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for task in document.get("tasks", []):
            decks.add(tuple(task.get("deck") or ()))
    return decks


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Probability a won episode's frame scores above a lost episode's frame."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    combined = np.concatenate([positive, negative])
    sorted_values = combined[order]
    start = 0
    for index in range(1, len(sorted_values) + 1):
        if index == len(sorted_values) or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = (start + 1 + index) / 2.0
            start = index
    rank_positive = ranks[: len(positive)].sum()
    return float((rank_positive - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def reliability(prediction: np.ndarray, label: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (prediction >= low) & (prediction < high if index < bins - 1 else prediction <= high)
        if not mask.any():
            out.append({"bin": [round(low, 2), round(high, 2)], "n": 0})
            continue
        out.append({"bin": [round(low, 2), round(high, 2)], "n": int(mask.sum()),
                    "predicted": float(prediction[mask].mean()),
                    "actual": float(label[mask].mean())})
    return out


def summarise(prediction: np.ndarray, label: np.ndarray) -> dict:
    if len(prediction) == 0:
        return {"n": 0}
    clipped = np.clip(prediction, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(prediction)),
        "brier": float(np.mean((clipped - label) ** 2)),
        "log_loss": float(-np.mean(label * np.log(clipped) + (1 - label) * np.log(1 - clipped))),
        "mean_prediction": float(prediction.mean()),
        "actual": float(label.mean()),
        "auc": auc(prediction, label),
    }


ENCODER_VERSION = {ENCODER_V1: 1, ENCODER_V2: 2, ENCODER_V3: 3, ENCODER_V4: 4, ENCODER_V5: 5}


def checkpoint_encoder(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return ENCODER_VERSION[checkpoint.get("encoder", ENCODER_V1)]


def score_episode(entry: dict, net, frames_dir: Path, positions: int, version: int) -> dict | None:
    # Episodes may live under more than one archive root (the pipeline view and the queue's
    # default frames_v2 tree); the index records which one per entry.
    path = Path(entry.get("frames_root") or frames_dir) / entry["dir"] / "frames.npz"
    try:
        frames = read_frames(path)
    except (OSError, ValueError):
        return None
    if len(frames) < positions + 2:
        return None
    indices = [int(round(len(frames) * (i + 1) / (positions + 1))) for i in range(positions)]
    indices = [min(max(1, index), len(frames) - 1) for index in indices]
    context = context_vector(float(entry.get("multiplier") or 1.0), entry.get("roster") or [])
    boards, scalars = [], []
    for index in indices:
        board, scalar = encode_frame(frames[index], version)
        boards.append(board)
        scalars.append(np.concatenate([scalar.numpy(), context]))
    with torch.no_grad():
        logits = net(torch.from_numpy(np.stack(boards)), torch.from_numpy(np.stack(scalars)))
    probability = torch.sigmoid(logits).numpy()
    scene = frames[len(frames) // 2].get("scene") or {}
    return {
        "dir": entry["dir"],
        "seed": entry["seed"],
        "round": entry["round"],
        "task_id": entry["task_id"],
        "level": entry["level"],
        "terrain": TERRAIN_NAMES.get(pvz_constants.background_for_level(int(entry["level"])), "?"),
        "multiplier": float(entry.get("multiplier") or 1.0),
        "deck": entry.get("deck"),
        "start_cell": bool(entry.get("start_cell")),
        "llm_start": "llm" in str(entry.get("demo_archive") or "") or "llm" in str(entry.get("task_id") or ""),
        "won": bool(entry["won"]),
        "scene_flags": [bool(scene.get(k)) for k in ("night", "pool", "fog", "roof")],
        "positions": probability.astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--min-round", type=int, default=36)
    parser.add_argument("--max-round", type=int, default=10 ** 6)
    parser.add_argument("--positions", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--value", type=Path, default=DEFAULT_VALUE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    index = json.loads(args.index.read_text())
    frames_dir = Path(index["frames"])
    entries = [entry for entry in index["episodes"]
               if entry.get("task_id") and entry.get("won") is not None
               and args.min_round <= (entry.get("round") or -1) <= args.max_round]
    if args.limit:
        entries = entries[: args.limit]
    print(f"{len(entries)} episodes, rounds {args.min_round}..{args.max_round}", flush=True)

    net = load_value_net(args.value)
    version = checkpoint_encoder(args.value)
    decks = legacy_decks()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        scored = [row for row in pool.map(lambda entry: score_episode(entry, net, frames_dir, args.positions, version), entries)
                  if row is not None]
    print(f"scored {len(scored)}", flush=True)

    prediction = np.concatenate([row["positions"] for row in scored])
    label = np.repeat(np.array([1.0 if row["won"] else 0.0 for row in scored]), args.positions)
    deck_seen = np.repeat(np.array([tuple(row["deck"] or ()) in decks for row in scored]), args.positions)
    terrain = np.repeat(np.array([row["terrain"] for row in scored]), args.positions)
    multiplier = np.repeat(np.array([row["multiplier"] for row in scored]), args.positions)
    start_cell = np.repeat(np.array([row["start_cell"] for row in scored]), args.positions)
    llm_start = np.repeat(np.array([row["llm_start"] for row in scored]), args.positions)

    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(prediction), dtype=bool),
    }
    for name in sorted(set(terrain)):
        groups[f"terrain={name}"] = terrain == name
    for name in sorted(set(multiplier)):
        groups[f"multiplier={name:g}"] = multiplier == name
    groups["deck_seen"] = deck_seen
    groups["deck_unseen"] = ~deck_seen
    groups["start_cell"] = start_cell
    groups["start_cell=false"] = ~start_cell
    groups["llm_start"] = llm_start
    for name in sorted(set(terrain)):
        for seen in (True, False):
            groups[f"terrain={name}|deck_seen={seen}"] = (terrain == name) & (deck_seen == seen)
    for name in sorted(set(terrain)):
        groups[f"terrain={name}|start_cell"] = (terrain == name) & start_cell

    summary = {
        "value": str(args.value),
        "encoder_version": version,
        "rounds": [args.min_round, args.max_round],
        "positions": args.positions,
        "episodes": len(scored),
        "won_episodes": int(sum(row["won"] for row in scored)),
        "groups": {name: summarise(prediction[mask], label[mask]) for name, mask in groups.items()},
        "reliability": {name: reliability(prediction[mask], label[mask]) for name, mask in groups.items()
                        if name in ("all", "deck_seen", "deck_unseen", "start_cell")},
        "reliability_by_terrain": {name: reliability(prediction[groups[f"terrain={name}"]],
                                                    label[groups[f"terrain={name}"]])
                                   for name in sorted(set(terrain))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1))
    if args.rows:
        np.savez_compressed(args.rows,
                            prediction=prediction, label=label, terrain=terrain,
                            multiplier=multiplier, deck_seen=deck_seen, start_cell=start_cell,
                            llm_start=llm_start,
                            episode=np.repeat(np.arange(len(scored)), args.positions))
        (args.rows.with_suffix(".episodes.json")).write_text(json.dumps([
            {key: row[key] for key in ("dir", "seed", "round", "task_id", "level", "terrain",
                                       "multiplier", "deck", "start_cell", "llm_start", "won")}
            for row in scored]))
    print(json.dumps({name: summary["groups"][name] for name in
                      ("all", "deck_seen", "deck_unseen", "start_cell")}, indent=1))
    for name in sorted(set(terrain)):
        print(name, summary["groups"][f"terrain={name}"])


if __name__ == "__main__":
    main()
