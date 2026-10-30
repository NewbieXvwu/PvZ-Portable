#!/usr/bin/env python3
"""Per-row policy surprise: KL(search target || production student prior) on an existing pack.

KataGo's policy surprise weighting (KataGoMethods.md, g170) draws half of the training
samples uniformly and half by how much the search target disagrees with the network prior.
Here the target is what the pack already stores -- per-packet best search values and the
per-cell soft labels -- and the prior is the current production student evaluated on the
same rows. Nothing is collected or replayed: one forward pass over the training pack.

Output is one row-aligned npz cache (float16 KL plus masks and the components); the
trainer reads it through the recipe's ``kl_weights`` field.

Usage:
  python3 scripts/kl_policy_surprise.py --recipe <joint64 recipe> --student <model.pt> \
      --out <cache.npz> --fingerprint abdb7e217e0c5ea5 --device mps
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import train_rollout_policy as T  # noqa: E402
from pvz_recipe import load_recipe, training_row_options  # noqa: E402
from pvz_rollout_policy import load_student  # noqa: E402


def packet_kl(values, net_out, temperature):
    """KL(softmax(search values / T) || student plant/packet prior) over the compared slots.

    The value store keeps the best search value per seed-bank slot and the wait slot; the
    student prior over the same slots is P(plant) * packet_softmax(slot) with P(wait) =
    1 - P(plant), renormalized on the compared slots.
    """
    known = ~torch.isnan(values)
    count = known.sum(1)
    target = torch.softmax(values.nan_to_num(0).masked_fill(~known, -1e9) / temperature, 1)
    plant = torch.sigmoid(net_out["plant"])
    packet = torch.softmax(net_out["packet"], 1)
    prior = torch.cat([plant[:, None] * packet, 1 - plant[:, None]], 1)
    prior = prior.masked_fill(~known, 0)
    prior = prior / prior.sum(1, keepdim=True).clamp_min(1e-12)
    kl = (target * (torch.log(target.clamp_min(1e-12)) - torch.log(prior.clamp_min(1e-12)))).sum(1)
    compared = count >= 2
    return kl * compared, compared


def cell_kl(cands, target_packet, net_out, temperature):
    """KL(cell soft target of the labelled packet || student cell prior on the same cells)."""
    planted = target_packet >= 0
    same = ((cands[:, :, 0] == target_packet[:, None].float()) & ~torch.isnan(cands[:, :, 2])
            & planted[:, None])
    rows = same.sum(1) >= 2
    out = torch.zeros(cands.shape[0], dtype=torch.float32, device=cands.device)
    if rows.any():
        mask = same[rows]
        cells = cands[rows][:, :, 1].nan_to_num(0).clamp(min=0).long()
        logits = net_out["cell"][rows].gather(1, cells)
        prior = torch.log_softmax(logits.masked_fill(~mask, -1e9), 1)
        target = torch.softmax((cands[rows][:, :, 2].nan_to_num(0) / temperature).masked_fill(~mask, -1e9), 1)
        kl = (target * (torch.log(target.clamp_min(1e-12)) - prior)).masked_fill(~mask, 0).sum(1)
        out[rows] = kl
    return out, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--student", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fingerprint", required=True, help="archive fingerprint recorded in the pack index")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--temperature", type=float, help="defaults to the recipe's soft_temperature")
    args = ap.parse_args()

    recipe = load_recipe(args.recipe, "training")
    temperature = recipe["soft_temperature"] if args.temperature is None else args.temperature
    T.archive_fingerprint = lambda data, labels: args.fingerprint
    loaded = T.load(recipe["data"], **training_row_options(recipe))
    boards, scalars, plant, packet, cell, _weight, _seed, values, cands = loaded[:9]
    packet_info = loaded[9] if recipe["packet_head"] == "identity" else None
    n = len(plant)
    net = load_student(args.student).to(args.device).eval()

    total = np.zeros(n, dtype=np.float32)
    packet_rows = np.zeros(n, dtype=bool)
    cell_rows = np.zeros(n, dtype=bool)
    packet_part = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, args.batch_size):
            idx = torch.arange(start, min(start + args.batch_size, n), dtype=torch.int64)
            board = boards.batch(idx, args.device)
            scalar = scalars.batch(idx, args.device)
            target_packet = packet[idx].to(args.device).long()
            info = packet_info[idx].to(args.device) if packet_info is not None else None
            out = net(board, scalar, target_packet.clamp(min=0), packet_info=info)
            kp, kp_rows = packet_kl(values[idx].to(args.device), out, temperature)
            kc, kc_rows = cell_kl(cands[idx].to(args.device).float(), target_packet, out, temperature)
            stop = start + len(idx)
            packet_part[start:stop] = kp.cpu().numpy()
            total[start:stop] = (kp + kc).cpu().numpy()
            packet_rows[start:stop] = kp_rows.cpu().numpy()
            cell_rows[start:stop] = kc_rows.cpu().numpy()

    nonzero = total > 0
    stats = {
        "rows": int(n), "packet_rows": int(packet_rows.sum()), "cell_rows": int(cell_rows.sum()),
        "nonzero": int(nonzero.sum()), "mean": float(total[nonzero].mean()),
        "p50": float(np.percentile(total[nonzero], 50)), "p90": float(np.percentile(total[nonzero], 90)),
        "p99": float(np.percentile(total[nonzero], 99)), "max": float(total.max()),
    }
    print(json.dumps(stats), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, kl=total.astype(np.float16),
                        kl_packet=packet_part.astype(np.float16), packet_rows=packet_rows,
                        cell_rows=cell_rows,
                        meta=json.dumps({"student": str(args.student), "recipe": str(args.recipe),
                                         "data": recipe["data"], "fingerprint": args.fingerprint,
                                         "temperature": temperature, "rows": int(n)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
