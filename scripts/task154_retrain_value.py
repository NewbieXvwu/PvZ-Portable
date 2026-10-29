"""task154 step 3: a diagnosis-only value net retrained on the most recent qc frames.

v5 has not changed since 2026-10-24, before the curriculum had roof, fog-wide and random-deck
cells. This tool builds an episode-outcome corpus from the newest qc archives (read-only),
fine-tunes v5 on it, and reports held-out calibration per terrain so the retrained net can be
used as one arm of the teacher attribution. Nothing here changes the production default.

    python3 scripts/task154_retrain_value.py --index artifacts/task154/index.json \
        --min-round 30 --out artifacts/task154/value_recent.pt --report artifacts/task154/value_recent.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import pvz_constants  # noqa: E402
from pvz_frame_store import read_frames  # noqa: E402
from pvz_value_net import (ENCODER_V1, ENCODER_V2, ENCODER_V3, ENCODER_V4,  # noqa: E402
                           ENCODER_V5, ValueNet, context_vector, encode_frame, load_value_net)

DEFAULT_VALUE = Path("/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt")
ENCODER_VERSION = {ENCODER_V1: 1, ENCODER_V2: 2, ENCODER_V3: 3, ENCODER_V4: 4, ENCODER_V5: 5}
TERRAIN_NAMES = {
    "BACKGROUND_1_DAY": "day", "BACKGROUND_2_NIGHT": "night", "BACKGROUND_3_POOL": "pool",
    "BACKGROUND_4_FOG": "fog", "BACKGROUND_5_ROOF": "roof", "BACKGROUND_6_BOSS": "boss",
}


def terrain_of(level) -> str:
    return TERRAIN_NAMES.get(pvz_constants.background_for_level(int(level)), "?")


def encode_episode(entry: dict, frames_dir: Path, positions: int, version: int):
    path = frames_dir / entry["dir"] / "frames.npz"
    try:
        frames = read_frames(path)
    except (OSError, ValueError):
        return None
    if len(frames) < positions + 2:
        return None
    indices = [min(max(1, int(round(len(frames) * (i + 1) / (positions + 1)))), len(frames) - 1)
               for i in range(positions)]
    context = context_vector(float(entry.get("multiplier") or 1.0), entry.get("roster") or [])
    boards, scalars = [], []
    for index in indices:
        board, scalar = encode_frame(frames[index], version)
        boards.append(board.numpy())
        scalars.append(np.concatenate([scalar.numpy(), context]))
    return (np.stack(boards), np.stack(scalars), float(bool(entry["won"])), entry["dir"],
            terrain_of(entry["level"]))


def brier(prediction, label):
    return float(np.mean((prediction - label) ** 2))


def auc(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
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


def predict(net, boards, scalars, batch=1024):
    device = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(boards), batch):
            logits = net(torch.from_numpy(boards[start:start + batch]).to(device),
                         torch.from_numpy(scalars[start:start + batch]).to(device))
            out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=DEFAULT_VALUE)
    parser.add_argument("--min-round", type=int, default=30)
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=None, help="reuse/refresh encoded rows")
    args = parser.parse_args()

    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    version = ENCODER_VERSION[checkpoint.get("encoder", ENCODER_V1)]
    index = json.loads(args.index.read_text())
    frames_dir = Path(index["frames"])
    entries = [entry for entry in index["episodes"]
               if entry.get("task_id") and entry.get("won") is not None
               and (entry.get("round") or -1) >= args.min_round]
    print(f"{len(entries)} candidate episodes", flush=True)

    if args.cache and args.cache.exists():
        stored = np.load(args.cache, allow_pickle=False)
        counts = stored["counts"]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        encoded = [(stored["boards"][offsets[index]:offsets[index + 1]].astype(np.float32),
                    stored["scalars"][offsets[index]:offsets[index + 1]],
                    float(stored["labels"][index]), str(stored["dirs"][index]),
                    str(stored["terrain"][index])) for index in range(len(counts))]
        print(f"loaded {len(encoded)} encoded episodes from cache", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            encoded = [row for row in pool.map(
                lambda entry: encode_episode(entry, frames_dir, args.positions, version), entries)
                if row is not None]
        print(f"encoded {len(encoded)} episodes", flush=True)
        if args.cache:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.cache,
                boards=np.concatenate([row[0] for row in encoded]).astype(np.float16),
                scalars=np.concatenate([row[1] for row in encoded]).astype(np.float32),
                labels=np.array([row[2] for row in encoded], dtype=np.float32),
                dirs=np.array([row[3] for row in encoded]),
                terrain=np.array([row[4] for row in encoded]),
                counts=np.array([len(row[0]) for row in encoded], dtype=np.int64))

    # Episode-level split keyed on the archive path: no game contributes to both sides.
    train_boards, train_scalars, train_labels = [], [], []
    val_boards, val_scalars, val_labels, val_terrain, val_dirs = [], [], [], [], []
    for boards, scalars, label, directory, terrain in encoded:
        digest = int.from_bytes(hashlib.sha1(directory.encode()).digest()[:4], "big") % 1000 / 1000.0
        if digest < args.val_fraction:
            val_boards.append(boards); val_scalars.append(scalars)
            val_labels.append(np.full(len(boards), label, dtype=np.float32))
            val_terrain.extend([terrain] * len(boards)); val_dirs.append(directory)
        else:
            train_boards.append(boards); train_scalars.append(scalars)
            train_labels.append(np.full(len(boards), label, dtype=np.float32))
    train_boards = np.concatenate(train_boards); train_scalars = np.concatenate(train_scalars)
    train_labels = np.concatenate(train_labels)
    val_boards = np.concatenate(val_boards); val_scalars = np.concatenate(val_scalars)
    val_labels = np.concatenate(val_labels); val_terrain = np.array(val_terrain)
    print(f"train rows {len(train_labels)} (won {train_labels.mean():.3f}), "
          f"val rows {len(val_labels)} (won {val_labels.mean():.3f})", flush=True)

    device = torch.device(args.device)
    net = load_value_net(args.init).to(device)
    before = predict(net, val_boards, val_scalars)
    positive = max(1.0, float(train_labels.sum()))
    negative = max(1.0, float((1 - train_labels).sum()))
    weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = np.random.default_rng(1540)
    for epoch in range(args.epochs):
        net.train()
        order = generator.permutation(len(train_labels))
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = order[start:start + args.batch]
            boards = torch.from_numpy(train_boards[batch]).to(device)
            scalars = torch.from_numpy(train_scalars[batch]).to(device)
            labels = torch.from_numpy(train_labels[batch]).to(device)
            optimiser.zero_grad()
            loss = criterion(net(boards, scalars), labels)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)
        print(f"epoch {epoch} loss {total / len(order):.4f}", flush=True)

    net.eval()
    after = predict(net, val_boards, val_scalars)
    groups = {"all": np.ones(len(val_labels), dtype=bool)}
    for terrain in sorted(set(val_terrain)):
        groups[terrain] = val_terrain == terrain
    report = {
        "init": str(args.init), "episodes": len(encoded), "encoder_version": version,
        "train_rows": int(len(train_labels)), "val_rows": int(len(val_labels)),
        "epochs": args.epochs, "lr": args.lr, "positions": args.positions,
        "min_round": args.min_round,
        "validation": {name: {
            "rows": int(mask.sum()),
            "won": float(val_labels[mask].mean()) if mask.any() else None,
            "brier_v5": brier(before[mask], val_labels[mask]) if mask.any() else None,
            "brier_recent": brier(after[mask], val_labels[mask]) if mask.any() else None,
            "auc_v5": auc(before[mask], val_labels[mask]) if mask.any() else None,
            "auc_recent": auc(after[mask], val_labels[mask]) if mask.any() else None,
        } for name, mask in groups.items()},
    }
    print(json.dumps(report["validation"], indent=1))
    torch.save({**checkpoint, "state_dict": net.cpu().state_dict(),
                "trained_on_recent_rounds": [args.min_round, None],
                "source": "task154 diagnosis-only: recent qc episode outcomes",
                "init": str(args.init)}, args.out)
    args.report.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
