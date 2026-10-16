"""Train the value net on branched leaves instead of whole games (DESIGN.md §14).

Value nets trained on game results learn "which game is this" (difficulty, seed) and rank the moves of one
position little better than chance (scripts/value_ranking_check.py, 2026-10-16: value_self1 pair accuracy
0.55). Here every sample comes from ``value_ranking_check.py --dump``: from one position each candidate move
is played and continued by the student several times, so samples next to each other differ by the move and
by the random future, and the label is how that very run ended. That is the question search asks the net.

Dumps made with ``--cells`` / ``--random-cells`` also hold, per plant candidate, the board right after planting
(kind 1, labelled with the mean result of all its runs) -- what the search's static scoring sees when it picks a
card's cell. Older dumps only have leaves (kind 0).

Split by game seed (10% held out). Reported on the held-out seeds, per value net: pair accuracy over the
candidates of one position (true win rates at least ``--min-gap`` apart), the same number the ranking check
reports -- for leaves over all candidates, for boards right after planting over the cells of one card. Leaves
whose run ended before the horizon are not in the dump, so a candidate's true win rate there is over its live
leaves only.

Usage:
  python scripts/train_value_branches.py --data a.pt,b.pt --out artifacts/value_net/branch1 \
      --compare v2=artifacts/value_net/v2/model.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_value_net import SCALARS, ValueNet, load_value_net  # noqa: E402


def predict(net: ValueNet, boards, scalars, device: str) -> torch.Tensor:
    if not net.context:
        scalars = scalars[:, :SCALARS]
    out = []
    with torch.inference_mode():
        for i in range(0, len(boards), 4096):
            out.append(torch.sigmoid(net(boards[i:i + 4096].float().to(device), scalars[i:i + 4096].to(device))).cpu())
    return torch.cat(out)


def pair_accuracy(pred, d, idx, min_gap: float, by_packet: bool = False) -> dict:
    """Mean prediction vs mean result per (seed, position, candidate), compared within each position
    (``by_packet``: within the cells of one card)."""
    groups = defaultdict(lambda: defaultdict(list))
    for k in idx.tolist():
        key = (int(d["seed"][k]), int(d["position"][k])) + ((int(d["packet"][k]),) if by_packet else ())
        groups[key][int(d["candidate"][k])].append(k)
    agree = pairs = 0
    for cands in groups.values():
        stats = [(float(d["won"][ks].mean()), float(pred[ks].mean())) for ks in map(torch.tensor, cands.values())]
        for i in range(len(stats)):
            for j in range(i + 1, len(stats)):
                if abs(stats[i][0] - stats[j][0]) >= min_gap:
                    pairs += 1
                    agree += (stats[i][0] - stats[j][0]) * (stats[i][1] - stats[j][1]) > 0
    return {"pair_accuracy": agree / max(1, pairs), "pairs": pairs, "positions": len(groups)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dumps of value_ranking_check.py, comma separated")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare", default="", help="name=path,... value nets scored on the same held-out leaves")
    p.add_argument("--init", help="start from this value checkpoint (must take the task context)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--min-gap", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    parts = [torch.load(path, weights_only=False) for path in args.data.split(",")]
    for part in parts:                      # dumps from before kind 1 existed: leaves only, packet unknown
        part.setdefault("kind", torch.zeros(len(part["won"]), dtype=torch.long))
        part.setdefault("packet", torch.full((len(part["won"]),), -2, dtype=torch.long))
    keys = ("boards", "scalars", "won", "seed", "position", "candidate", "packet", "kind")
    d = {k: torch.cat([part[k] for part in parts]) for k in keys}
    d["won"] = d["won"].float()
    seeds = sorted(set(d["seed"].tolist()))
    random.Random(0).shuffle(seeds)
    held = torch.tensor([s in set(seeds[: max(1, len(seeds) // 10)]) for s in d["seed"].tolist()])
    fit_idx, val_idx = torch.nonzero(~held).squeeze(1), torch.nonzero(held).squeeze(1)
    print(f"{len(d['won'])} rows ({int((d['kind'] == 1).sum())} right after planting) from {len(seeds)} seeds; "
          f"fit {len(fit_idx)}, held out {len(val_idx)}; win rate {float(d['won'].mean()):.3f}", flush=True)

    net = ValueNet(context=True)
    if args.init:
        net.load_state_dict(load_value_net(args.init).state_dict())
    net.to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    won = d["won"]

    def run(idx, train: bool) -> float:
        net.train(train)
        order = idx[torch.randperm(len(idx))] if train else idx
        total = 0.0
        for i in range(0, len(order), 1024):
            batch = order[i:i + 1024]
            with torch.set_grad_enabled(train):
                logits = net(d["boards"][batch].float().to(args.device), d["scalars"][batch].to(args.device))
                loss = F.binary_cross_entropy_with_logits(logits, won[batch].to(args.device))
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(batch)
        return total / len(order)

    best, best_state, patience, history = float("inf"), None, 0, []
    for epoch in range(args.epochs):
        fit_loss = run(fit_idx, True)
        val_loss = run(val_idx, False)
        history.append({"epoch": epoch, "fit": fit_loss, "held_out": val_loss})
        print(f"epoch {epoch} fit {fit_loss:.4f} held-out {val_loss:.4f}", flush=True)
        if val_loss < best - 1e-4:
            best, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            patience += 1
            if patience >= 3:
                break
    net.load_state_dict(best_state)
    net.eval()

    held_d = {k: v[val_idx] for k, v in d.items()}
    leaf_idx = torch.nonzero(held_d["kind"] == 0).squeeze(1)
    static_idx = torch.nonzero(held_d["kind"] == 1).squeeze(1)

    def score(model) -> dict:
        pred = predict(model, held_d["boards"], held_d["scalars"], args.device)
        out = {"leaves": pair_accuracy(pred, held_d, leaf_idx, args.min_gap)}
        if len(static_idx):
            out["cells"] = pair_accuracy(pred, held_d, static_idx, args.min_gap, by_packet=True)
        return out

    report = {"new": score(net)}
    for kv in filter(None, args.compare.split(",")):
        name, path = kv.split("=", 1)
        report[name] = score(load_value_net(path).to(args.device))
    print(json.dumps(report, indent=1))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.cpu().state_dict(), "encoder": "pvz_value_net.featurize v1", "context": True,
                "trained_on": seeds, "source": "branches", "data": args.data}, args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps({"held_out": report, "history": history,
                                                       "leaves": len(d["won"]), "seeds": len(seeds)}, indent=1))


if __name__ == "__main__":
    main()
