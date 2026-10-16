"""Train the fast rollout policy on pi's own games (see python/pvz_rollout_policy.py).

Data: episode archives whose frames carry the executed action (``frames[i]["action"]`` produced
frame i). Input is frame i-1, target is the action at frame i. Split by environment seed.

Usage:
  python scripts/train_rollout_policy.py --data artifacts/value_net/episodes --labels u84_greedy,u84_sampled \
      --out artifacts/rollout_policy/v1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pvz_rollout_policy import LAWN_COLS, PACKETS, RolloutNet, features  # noqa: E402


def load(data: Path, labels: set[str], wait_keep: float = 1.0, label_weight: dict[str, float] | None = None):
    """Rows: frame i-1 -> action at frame i. With ``wait_keep`` < 1 only that fraction of the wait rows
    (~94% of all rows) is kept, each weighted 1 / wait_keep, so more games fit in memory.
    ``label_weight`` multiplies the weight of every row of a label (a small, newer teacher among many older games)."""
    paths = [p for p in sorted(data.glob("*/*/frames.jsonl")) if p.parent.name in labels]
    # Decide the kept rows first and preallocate exactly: a list of per-row tensors plus torch.stack needs
    # twice the memory, which on 887k rows (~10 GB of boards) took down the 23 GB WSL VM (2026-10-15).
    rng = random.Random(0)
    keep = {}
    for path in paths:
        mask = []
        for line in path.open():
            tail = line[line.rfind('"action":'):]          # the executed action is the last field
            mask.append('"type":"plant"' in tail or rng.random() < wait_keep)
        keep[path] = mask
    capacity = sum(map(sum, keep.values()))
    boards = scalars = None
    plant, packet, cell, weight, seed_of = [], [], [], [], []
    n = 0
    for path in paths:
        seed = int(path.parent.parent.name.split("_")[-1])
        scale = (label_weight or {}).get(path.parent.name, 1.0)
        frames = [json.loads(line) for line in path.open()]
        for i in range(1, len(frames)):
            action = frames[i].get("action")
            if not action or frames[i].get("action_rejected") or not keep[path][i]:
                continue
            b, s = features(frames[i - 1], frames[i - 2] if i >= 2 else None)
            if boards is None:
                boards = torch.empty((capacity, *b.shape), dtype=torch.half)
                scalars = torch.empty((capacity, *s.shape), dtype=s.dtype)
            boards[n] = b; scalars[n] = s; n += 1
            seed_of.append(seed)
            is_plant = action["type"] == "plant"
            plant.append(float(is_plant))
            weight.append(scale * (1.0 if is_plant else 1.0 / wait_keep))
            packet.append(action["packet"] if is_plant else -1)
            cell.append(action["row"] * LAWN_COLS + action["col"] if is_plant else -1)
    return (boards[:n], scalars[:n], torch.tensor(plant), torch.tensor(packet), torch.tensor(cell),
            torch.tensor(weight), seed_of)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", default="u84_greedy,u84_sampled")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--wait-keep", type=float, default=1.0, help="fraction of wait rows kept (reweighted)")
    p.add_argument("--seed", type=int, default=None, help="torch seed for init and shuffling")
    p.add_argument("--width", type=int, default=64, help="channels of the net (smaller = faster rollouts)")
    p.add_argument("--label-weight", default="", help="label=factor,... row-weight multipliers per label")
    p.add_argument("--packet-balance", type=float, default=0.0,
                   help="weight plant rows by (packet frequency)^-x, so rare cards (cherry 1.5%%) are not drowned out")
    p.add_argument("--pool", choices=["mean", "meanmax"], default="mean", help="global pooling of the trunk")
    args = p.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    started = time.time()
    label_weight = {k: float(v) for k, v in (kv.split("=") for kv in args.label_weight.split(",") if kv)}
    boards, scalars, plant, packet, cell, weight, seed_of = load(args.data, set(args.labels.split(",")),
                                                                 args.wait_keep, label_weight)
    seeds = sorted(set(seed_of)); random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: len(seeds) // 10])
    is_val = torch.tensor([s in val_seeds for s in seed_of])
    print(f"{len(plant)} decisions ({int(plant.sum())} plants), {len(seeds)} seeds, "
          f"{int(is_val.sum())} validation rows", flush=True)

    if args.packet_balance:
        planted_rows = packet >= 0
        freq = torch.bincount(packet[planted_rows], minlength=PACKETS).float() / int(planted_rows.sum())
        scale = freq.clamp(min=1e-6) ** -args.packet_balance
        scale = (scale / scale[packet[planted_rows]].mean()).clamp(max=20.0)  # mean plant-row weight ~1
        weight[planted_rows] *= scale[packet[planted_rows]]
        print("packet weights", {k: round(float(scale[k]), 2) for k in range(PACKETS) if freq[k] > 0}, flush=True)
    net = RolloutNet(args.width, args.pool).to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    waits = float(weight[plant < 0.5].sum())
    pos_weight = torch.tensor([waits / max(1.0, float(plant.sum()))], device=args.device).sqrt()

    def batch_loss(idx, train):
        b = boards[idx].float().to(args.device); s = scalars[idx].to(args.device)
        y = plant[idx].to(args.device); pk = packet[idx].to(args.device); c = cell[idx].to(args.device)
        w = weight[idx].to(args.device)
        planted = y > 0.5
        out = net(b, s, pk.clamp(min=0))
        loss = (F.binary_cross_entropy_with_logits(out["plant"], y, pos_weight=pos_weight, reduction="none") * w
                ).sum() / w.sum()
        if planted.any():
            wp = w[planted]                               # all 1 unless --label-weight is used
            loss = loss + ((F.cross_entropy(out["packet"][planted], pk[planted], reduction="none") +
                            F.cross_entropy(out["cell"][planted], c[planted], reduction="none")) * wp).sum() / wp.sum()
        stats = {}
        if not train:
            with torch.no_grad():
                pred_plant = out["plant"] > 0
                stats = {"type_acc": float(((pred_plant == planted).float() * w).sum() / w.sum()),
                         "plant_recall": float(pred_plant[planted].float().mean()) if planted.any() else 0.0,
                         "packet_acc": float((out["packet"][planted].argmax(1) == pk[planted]).float().mean()) if planted.any() else 0.0,
                         "cell_acc": float((out["cell"][planted].argmax(1) == c[planted]).float().mean()) if planted.any() else 0.0}
        return loss, stats

    train_idx = torch.nonzero(~is_val).squeeze(1); val_idx = torch.nonzero(is_val).squeeze(1)
    history = []
    for epoch in range(1, args.epochs + 1):
        net.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(perm), 2048):
            loss, _ = batch_loss(perm[start:start + 2048], True)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        # Validate in chunks: one ~95k-row batch is a >2 GB tensor, which MPS silently gets wrong
        # (2026-10-15: garbage val metrics while the model itself was fine on the CPU).
        with torch.no_grad():
            parts = [(len(chunk), *batch_loss(chunk, False)) for chunk in val_idx.split(4096)]
        total = sum(n for n, _l, _s in parts)
        vl = sum(float(l) * n for n, l, _s in parts) / total
        vs = {k: sum(st[k] * n for n, _l, st in parts) / total for k in parts[0][2]}
        history.append({"epoch": epoch, "val_loss": float(vl), **vs})
        print(json.dumps(history[-1]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "labels": args.labels, "width": args.width,
                "label_weight": label_weight, "pool": args.pool, "packet_balance": args.packet_balance,
                "history": history}, args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps({"history": history, "seconds": time.time() - started}, indent=1))
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
