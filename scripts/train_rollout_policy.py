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


def load(data: Path, labels: set[str]):
    boards, scalars, plant, packet, cell, seed_of = [], [], [], [], [], []
    for path in sorted(data.glob("*/*/frames.jsonl")):
        if path.parent.name not in labels:
            continue
        seed = int(path.parent.parent.name.split("_")[-1])
        frames = [json.loads(line) for line in path.open()]
        for i in range(1, len(frames)):
            action = frames[i].get("action")
            if not action or frames[i].get("action_rejected"):
                continue
            b, s = features(frames[i - 1], frames[i - 2] if i >= 2 else None)
            boards.append(b.half()); scalars.append(s); seed_of.append(seed)
            is_plant = action["type"] == "plant"
            plant.append(float(is_plant))
            packet.append(action["packet"] if is_plant else -1)
            cell.append(action["row"] * LAWN_COLS + action["col"] if is_plant else -1)
    return (torch.stack(boards), torch.stack(scalars), torch.tensor(plant), torch.tensor(packet),
            torch.tensor(cell), seed_of)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--labels", default="u84_greedy,u84_sampled")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    started = time.time()
    boards, scalars, plant, packet, cell, seed_of = load(args.data, set(args.labels.split(",")))
    seeds = sorted(set(seed_of)); random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: len(seeds) // 10])
    is_val = torch.tensor([s in val_seeds for s in seed_of])
    print(f"{len(plant)} decisions ({int(plant.sum())} plants), {len(seeds)} seeds, "
          f"{int(is_val.sum())} validation rows", flush=True)

    net = RolloutNet().to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(len(plant) - plant.sum()) / max(1.0, plant.sum())], device=args.device).sqrt()

    def batch_loss(idx, train):
        b = boards[idx].float().to(args.device); s = scalars[idx].to(args.device)
        y = plant[idx].to(args.device); pk = packet[idx].to(args.device); c = cell[idx].to(args.device)
        planted = y > 0.5
        out = net(b, s, pk.clamp(min=0))
        loss = F.binary_cross_entropy_with_logits(out["plant"], y, pos_weight=pos_weight)
        if planted.any():
            loss = loss + F.cross_entropy(out["packet"][planted], pk[planted]) + \
                F.cross_entropy(out["cell"][planted], c[planted])
        stats = {}
        if not train:
            with torch.no_grad():
                pred_plant = out["plant"] > 0
                stats = {"type_acc": float((pred_plant == planted).float().mean()),
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
        with torch.no_grad():
            vl, vs = batch_loss(val_idx, False)
        history.append({"epoch": epoch, "val_loss": float(vl), **vs})
        print(json.dumps(history[-1]), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "labels": args.labels,
                "history": history}, args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps({"history": history, "seconds": time.time() - started}, indent=1))
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
