"""task154 step 5: train the value-net candidate v6 on the curriculum's own recent frames.

v5 (2026-10-24) predates roof, fog-wide boards, random-deck cells and demo start cells, and the
student has moved from r11 to r36. This tool builds an episode-outcome corpus from every completed
queue round since r36 (read-only), encodes evenly spaced mid-game frames with the teacher's own
encoder (v2, unchanged), fine-tunes v5 on it, and reports held-out calibration per terrain and per
cell type so the three-step acceptance can be run on the result. Nothing here changes the
production default.

    python3 scripts/task154_train_v6.py --index artifacts/task154/index.json \
        --min-round 36 --max-round 46 --out artifacts/value_net/v6/model.pt \
        --report artifacts/task154/v6_report.json --cache artifacts/task154/v6_rows.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
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
                           ENCODER_V5, context_vector, encode_frame, load_value_net)

INIT = Path("/home/newbiexvwu/PvZ-search/artifacts/value_net/v5/model.pt")
ENCODER_VERSION = {ENCODER_V1: 1, ENCODER_V2: 2, ENCODER_V3: 3, ENCODER_V4: 4, ENCODER_V5: 5}
TERRAIN_NAMES = {
    "BACKGROUND_1_DAY": "day", "BACKGROUND_2_NIGHT": "night", "BACKGROUND_3_POOL": "pool",
    "BACKGROUND_4_FOG": "fog", "BACKGROUND_5_ROOF": "roof", "BACKGROUND_6_BOSS": "boss",
}


def terrain_of(level) -> str:
    try:
        return TERRAIN_NAMES.get(pvz_constants.background_for_level(int(level)), "?")
    except (TypeError, ValueError):
        return "?"


def cell_kind(entry: dict) -> str:
    """Which curriculum cell produced the episode: random deck, demo start cell, or plain panel."""
    task_id = str(entry.get("task_id") or "")
    if task_id.startswith("rd137_") or "random_deck" in task_id:
        return "random_deck"
    if entry.get("start_cell"):
        return "start_cell"
    return "panel"


def encode_episode(job: tuple[dict, str, int, int]):
    entry, frames_dir, positions, version = job
    path = Path(entry.get("frames_root") or frames_dir) / entry["dir"] / "frames.npz"
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
    return (np.stack(boards).astype(np.float16), np.stack(scalars).astype(np.float32),
            float(bool(entry["won"])), entry["dir"], terrain_of(entry["level"]),
            cell_kind(entry), entry.get("round"))


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


def predict(net, boards, scalars, index, batch=512):
    device = next(net.parameters()).device
    out = []
    with torch.no_grad():
        for start in range(0, len(index), batch):
            rows = index[start:start + batch]
            logits = net(torch.from_numpy(np.asarray(boards[rows], dtype=np.float32)).to(device),
                         torch.from_numpy(np.asarray(scalars[rows], dtype=np.float32)).to(device))
            out.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(out)


def held_out(directory: str, terrain: str) -> bool:
    digest = int.from_bytes(hashlib.sha1(f"{terrain}/{directory}".encode()).digest()[:4], "big")
    return digest % 1000 / 1000.0 < 0.15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--init", type=Path, default=INIT)
    parser.add_argument("--min-round", type=int, default=36)
    parser.add_argument("--max-round", type=int, default=10 ** 6)
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    version = ENCODER_VERSION[checkpoint.get("encoder", ENCODER_V1)]
    index = json.loads(args.index.read_text())
    frames_dir = Path(index["frames"])
    entries = [entry for entry in index["episodes"]
               if entry.get("won") is not None and entry.get("level") is not None
               and entry.get("round") is not None
               and args.min_round <= entry["round"] <= args.max_round]
    coverage = Counter()
    for entry in entries:
        coverage[(terrain_of(entry["level"]), cell_kind(entry))] += 1
    print(f"{len(entries)} episodes in rounds {args.min_round}..{args.max_round}", flush=True)
    print("coverage:", dict(sorted(coverage.items())), flush=True)
    print("unmapped (no task metadata):", sum(1 for e in entries if not e.get("task_id")), flush=True)

    # The corpus is ~1 GiB of float16 boards, so it lives on disk as .npy and the training loop
    # reads batches straight from a memmap: the service runs under MemoryMax 4G and must not need
    # two full copies of the rows (the encoded list plus the concatenated array) at once.
    if args.cache and (args.cache / "boards.npy").exists():
        boards_all = np.load(args.cache / "boards.npy", mmap_mode="r")
        scalars_all = np.load(args.cache / "scalars.npy")
        labels_all = np.load(args.cache / "labels.npy")
        episodes = json.loads((args.cache / "episodes.json").read_text())
        print(f"loaded {len(episodes)} encoded episodes from {args.cache}", flush=True)
    else:
        jobs = [(entry, str(frames_dir), args.positions, version) for entry in entries]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            encoded = [row for row in pool.map(encode_episode, jobs, chunksize=8) if row is not None]
        print(f"encoded {len(encoded)} episodes", flush=True)
        episodes, offsets = [], 0
        boards_all = np.concatenate([row[0] for row in encoded])
        scalars_all = np.concatenate([row[1] for row in encoded])
        labels_all = np.concatenate([np.full(len(row[0]), row[2], dtype=np.float32) for row in encoded])
        for boards, scalars, label, directory, terrain, kind, round_index in encoded:
            episodes.append({"dir": directory, "terrain": terrain, "kind": kind,
                             "round": round_index, "label": float(label),
                             "offset": offsets, "count": len(boards)})
            offsets += len(boards)
        del encoded
        if args.cache:
            args.cache.mkdir(parents=True, exist_ok=True)
            for name, array in (("boards", boards_all), ("scalars", scalars_all), ("labels", labels_all)):
                np.save(args.cache / f"{name}.npy", array)
            (args.cache / "episodes.json").write_text(json.dumps(episodes))
            print(f"wrote cache to {args.cache}", flush=True)

    # Episode-level split, stratified by terrain so every terrain has a held-out slice.
    train_rows, val_rows, val_terrain, val_kind = [], [], [], []
    for episode in episodes:
        rows = np.arange(episode["offset"], episode["offset"] + episode["count"])
        if held_out(episode["dir"], episode["terrain"]):
            val_rows.append(rows)
            val_terrain.extend([episode["terrain"]] * len(rows))
            val_kind.extend([episode["kind"]] * len(rows))
        else:
            train_rows.append(rows)
    train_index = np.concatenate(train_rows); val_index = np.concatenate(val_rows)
    train_labels = labels_all[train_index]; val_labels = labels_all[val_index]
    val_terrain = np.array(val_terrain); val_kind = np.array(val_kind)
    print(f"train rows {len(train_labels)} (won {train_labels.mean():.3f}), "
          f"val rows {len(val_labels)} (won {val_labels.mean():.3f})", flush=True)

    device = torch.device(args.device)
    net = load_value_net(args.init).to(device)
    before = predict(net, boards_all, scalars_all, val_index)
    positive = max(1.0, float(train_labels.sum()))
    negative = max(1.0, float((1 - train_labels).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negative / positive], dtype=torch.float32,
                                                            device=device))
    optimiser = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = np.random.default_rng(1541)
    for epoch in range(args.epochs):
        net.train()
        order = generator.permutation(len(train_index))
        total = 0.0
        for start in range(0, len(order), args.batch):
            batch = train_index[order[start:start + args.batch]]
            boards = torch.from_numpy(np.asarray(boards_all[batch], dtype=np.float32)).to(device)
            scalars = torch.from_numpy(np.asarray(scalars_all[batch], dtype=np.float32)).to(device)
            labels = torch.from_numpy(labels_all[batch]).to(device)
            optimiser.zero_grad()
            loss = criterion(net(boards, scalars), labels)
            loss.backward()
            optimiser.step()
            total += float(loss) * len(batch)
        print(f"epoch {epoch} loss {total / len(order):.4f}", flush=True)

    net.eval()
    after = predict(net, boards_all, scalars_all, val_index)

    def block(mask):
        if not mask.any():
            return {"rows": 0}
        return {"rows": int(mask.sum()), "won": float(val_labels[mask].mean()),
                "brier_v5": brier(before[mask], val_labels[mask]),
                "brier_v6": brier(after[mask], val_labels[mask]),
                "auc_v5": auc(before[mask], val_labels[mask]),
                "auc_v6": auc(after[mask], val_labels[mask])}

    groups = {"all": np.ones(len(val_labels), dtype=bool)}
    for terrain in sorted(set(val_terrain)):
        groups[f"terrain={terrain}"] = val_terrain == terrain
    for kind in sorted(set(val_kind)):
        groups[f"cell={kind}"] = val_kind == kind
    for terrain in sorted(set(val_terrain)):
        for kind in sorted(set(val_kind)):
            groups[f"terrain={terrain}|cell={kind}"] = (val_terrain == terrain) & (val_kind == kind)
    report = {
        "init": str(args.init), "episodes": len(episodes), "encoder_version": version,
        "rounds": [args.min_round, args.max_round], "positions": args.positions,
        "train_rows": int(len(train_labels)), "val_rows": int(len(val_labels)),
        "epochs": args.epochs, "lr": args.lr,
        "coverage_episodes": {f"{terrain}|{kind}": count for (terrain, kind), count in sorted(coverage.items())},
        "validation": {name: block(mask) for name, mask in groups.items()},
    }
    print(json.dumps({name: report["validation"][name] for name in
                      ["all"] + [n for n in groups if n.startswith("terrain=") and "|" not in n]
                      + [n for n in groups if n.startswith("cell=")]}, indent=1))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Only the architecture fields carry over: v5's training bookkeeping (branch shards, decision
    # dumps, audit manifest) describes a corpus this net was not trained on and would mislead a
    # later reader about what v6 saw.
    torch.save({"encoder": checkpoint.get("encoder", ENCODER_V2),
                "width": int(checkpoint.get("width", 64)),
                "context": bool(checkpoint.get("context", False)),
                "state_dict": net.cpu().state_dict(),
                "trained_on_rounds": [args.min_round, args.max_round],
                "source": "task154 v6: qc episode outcomes r%d..r%d, encoder v2 unchanged"
                          % (args.min_round, args.max_round),
                "init": str(args.init),
                "init_sha256": hashlib.sha256(Path(args.init).read_bytes()).hexdigest(),
                "train_rows": int(len(train_labels)), "val_rows": int(len(val_labels)),
                "episodes": len(episodes), "positions": args.positions,
                "coverage": {f"{terrain}|{kind}": count for (terrain, kind), count in sorted(coverage.items())},
                }, args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
