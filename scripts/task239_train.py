"""task239 pilot: train v5 and v6 value heads on the same replayed rows and compare.

Both arms use the same games, frames, labels and optimisation; the only difference is the input
view.  v5 is the production encoder 2 (92 board channels, 119 scalars); v6 is engine encoder 6
(139 board channels, 125 scalars).  The v6 dump keeps the v5 board/scalars as an exact prefix, so
the control is a slice rather than a second replay.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from pvz_value_net import ValueNet  # noqa: E402

TERRAINS = ("day", "night", "pool", "fog", "roof")


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
                 / (positive.sum() * negative.sum()))


def metrics(prob: np.ndarray, label: np.ndarray) -> dict:
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return dict(n=len(label), wins=int(label.sum()),
                log_loss=float(-np.mean(label * np.log(prob) + (1 - label) * np.log(1 - prob))),
                brier=float(np.mean((prob - label) ** 2)),
                auc=auc(prob, label), accuracy=float(np.mean((prob >= 0.5) == label)),
                mean_prediction=float(prob.mean()), actual=float(label.mean()))


def load_rows(directory: Path) -> dict[str, dict[str, np.ndarray]]:
    data = {}
    for terrain in TERRAINS:
        packed = directory / terrain
        if packed.is_dir():
            data[terrain] = {path.stem: np.load(path, mmap_mode="r") for path in packed.glob("*.npy")}
            continue
        path = directory / f"{terrain}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as stored:
            data[terrain] = {key: stored[key] for key in stored.files}
    return data


def split_seeds(data: dict, every: int = 5) -> tuple[set, set]:
    train, holdout = set(), set()
    for terrain, rows in data.items():
        seeds = sorted({int(seed) for seed in rows["seed"]})
        for index, seed in enumerate(seeds):
            (holdout if index % every == 0 else train).add((terrain, seed))
    return train, holdout


def torch_view(rows: dict[str, np.ndarray], encoder: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    board = torch.from_numpy(rows["board"].astype(np.float32)).reshape(-1, rows["board"].shape[1], 6, 10)
    scalars = torch.from_numpy(rows["scalars"].astype(np.float32))
    context = torch.from_numpy(rows["context"].astype(np.float32))
    if encoder == 2:
        board, scalars = board[:, :92], scalars[:, :119]
    return board, scalars, context, torch.from_numpy(rows["y"].astype(np.float32))


def run_arm(data: dict, encoder: int, width: int, seed: int, epochs: int, device: torch.device,
            train_seeds: set, holdout_seeds: set, lr: float = 2e-4,
            dropout: float = 0.3, weight_decay: float = 1e-3, init_state: dict | None = None,
            batch_size: int = 256) -> tuple[dict, dict]:
    terrain_of, seed_of = [], []
    for terrain in TERRAINS:
        if terrain not in data:
            continue
        rows = data[terrain]
        terrain_of.extend([terrain] * len(rows["y"]))
        seed_of.extend([int(value) for value in rows["seed"]])
    terrain_of = np.asarray(terrain_of)
    seed_of = np.asarray(seed_of)
    train_mask = np.asarray([(terrain_of[i], seed_of[i]) in train_seeds for i in range(len(seed_of))])
    holdout_mask = np.asarray([(terrain_of[i], seed_of[i]) in holdout_seeds for i in range(len(seed_of))])
    labels = np.concatenate([data[t]["y"] for t in TERRAINS if t in data]).astype(np.float32)
    offsets, offset = [], 0
    for terrain in TERRAINS:
        if terrain in data:
            end = offset + len(data[terrain]["y"])
            offsets.append((offset, end, data[terrain]))
            offset = end

    def batch(index):
        # Gather only the requested rows. Full exports are float16 memmaps and stay off MPS.
        channels, scalar_count = (92, 119) if encoder == 2 else (139, 125)
        boards = np.empty((len(index), channels, 6, 10), dtype=np.float32)
        scalars = np.empty((len(index), scalar_count + 35), dtype=np.float32)
        for start, end, rows in offsets:
            mask = (index >= start) & (index < end)
            local = index[mask] - start
            boards[mask] = rows["board"][local, :channels].reshape(-1, channels, 6, 10)
            scalars[mask, :scalar_count] = rows["scalars"][local, :scalar_count]
            scalars[mask, scalar_count:] = rows["context"][local]
        return (torch.from_numpy(boards).to(device), torch.from_numpy(scalars).to(device),
                torch.from_numpy(labels[index]).to(device))

    torch.manual_seed(seed)
    net = ValueNet(width=width, context=True, encoder_version=encoder, dropout=dropout).to(device)
    if init_state is not None:
        # Warm-start a wider v6 input from a v5 checkpoint: keep the first 92 input channels and
        # every width-shaped tensor, leave the new internal-state channels at their initialisation.
        state = net.state_dict()
        for key, value in init_state.items():
            if key not in state:
                continue
            if key == "conv.0.weight":
                state[key][:, :value.shape[1]] = value
            elif state[key].shape == value.shape:
                state[key] = value
        net.load_state_dict(state)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_index = np.flatnonzero(train_mask)
    holdout_index = np.flatnonzero(holdout_mask)
    order_generator = torch.Generator().manual_seed(seed)

    best = (float("inf"), -1, None)
    history = []
    for epoch in range(epochs):
        net.train()
        order = train_index[torch.randperm(len(train_index), generator=order_generator).numpy()]
        total = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            board_batch, scalar_batch, label_batch = batch(index)
            optimizer.zero_grad()
            logits = net(board_batch, scalar_batch)
            loss = loss_fn(logits, label_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
        net.eval()
        with torch.no_grad():
            valid_total = 0.0
            for start in range(0, len(holdout_index), batch_size):
                index = holdout_index[start:start + batch_size]
                board_batch, scalar_batch, label_batch = batch(index)
                valid_total += float(loss_fn(net(board_batch, scalar_batch), label_batch)) * len(index)
            valid_loss = valid_total / len(holdout_index)
        history.append(dict(epoch=epoch, fit=total / len(train_index), validation=valid_loss))
        print(f"  epoch {epoch:02d} fit {total / len(train_index):.4f} validation {valid_loss:.4f}", flush=True)
        if valid_loss < best[0]:
            best = (valid_loss, epoch, {key: value.detach().cpu().clone() for key, value in net.state_dict().items()})
    net.load_state_dict(best[2])
    net.eval()
    prob = np.empty(len(labels), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(labels), batch_size):
            index = np.arange(start, min(start + batch_size, len(labels)))
            board_batch, scalar_batch, _ = batch(index)
            prob[index] = torch.sigmoid(net(board_batch, scalar_batch)).cpu().numpy()
    report = {"encoder": encoder, "width": width, "best_epoch": best[1], "best_validation": best[0],
              "history": history, "all_holdout": metrics(prob[holdout_mask], labels[holdout_mask]),
              "by_terrain": {}}
    for terrain in TERRAINS:
        mask = holdout_mask & (terrain_of == terrain)
        if mask.any():
            report["by_terrain"][terrain] = metrics(prob[mask], labels[mask])
    checkpoint = dict(encoder="pvz_value_net.featurize v6" if encoder == 6 else "pvz_value_net.featurize v2",
                      width=width, context=True, state_dict=net.state_dict())
    return report, checkpoint, prob, holdout_mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--all-data", action="store_true", help="fit every game; metrics are in-sample")
    parser.add_argument("--init-v6", type=Path, help="warm-start the v6 arm from this checkpoint")
    args = parser.parse_args()
    data = load_rows(args.data)
    if not data:
        raise SystemExit(f"no terrain rows under {args.data}")
    train_seeds, holdout_seeds = split_seeds(data)
    if args.all_data:
        train_seeds = set(train_seeds) | set(holdout_seeds)
        holdout_seeds = set(train_seeds)
    print(f"rows {sum(len(v['y']) for v in data.values())}, train seeds {len(train_seeds)}, holdout {len(holdout_seeds)}", flush=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    reports, checkpoints = {}, {}
    for name, encoder in (("v5", 2), ("v6", 6)):
        print(f"== {name} (encoder {encoder})", flush=True)
        init_state = None
        if encoder == 6 and args.init_v6:
            init_state = torch.load(args.init_v6, map_location="cpu", weights_only=False)["state_dict"]
        report, checkpoint, _prob, _mask = run_arm(data, encoder, args.width, args.seed, args.epochs, device,
                                                   train_seeds, holdout_seeds, dropout=args.dropout,
                                                   weight_decay=args.weight_decay, init_state=init_state)
        reports[name] = report
        checkpoints[name] = checkpoint
        torch.save(checkpoint, args.out / f"{name}.pt")
    (args.out / "report.json").write_text(json.dumps(dict(width=args.width, epochs=args.epochs, seed=args.seed,
                                                          dropout=args.dropout, weight_decay=args.weight_decay,
                                                          split_every=5, reports=reports), ensure_ascii=False, indent=1) + "\n")
    for name, report in reports.items():
        print(name, "holdout", report["all_holdout"])
        for terrain, value in report["by_terrain"].items():
            print(" ", terrain, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
