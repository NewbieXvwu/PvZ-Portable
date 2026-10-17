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
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from pvz_rollout_policy import LAWN_COLS, PACKETS, RolloutNet, features  # noqa: E402
from pvz_frame_store import cache_episode, frame_bytes, frame_paths, read_frames, target_values  # noqa: E402


def _episode_context(path: Path) -> np.ndarray:
    """The task context of one archive (meta.json, written by search_play or scripts/backfill_frame_context.py)."""
    from pvz_value_net import context_vector
    meta = json.loads((path.parent / "meta.json").read_text())
    if "context" not in meta:
        raise SystemExit(f"{path.parent} has no task context: run scripts/backfill_frame_context.py first")
    return context_vector(meta["context"]["multiplier"], meta["context"]["roster"])


def _board_store(shape, cache_dir: Path) -> torch.Tensor:
    """The boards of all kept rows (float16, ~11 KB a row) in a file-backed map beside the feature cache, not in
    process memory: D15 held 17.4 GB of boards in RAM and three trainings side by side ran the 23 GB WSL VM out of
    memory (2026-10-17). The kernel keeps what fits in its page cache and drops the rest, so a big data set costs
    disk reads instead of a crash. Same values as an in-memory tensor; the file is unlinked at once and freed on exit."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"boards_{os.getpid()}.npy"
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    path.unlink()
    return torch.from_numpy(store)


def load(data: Path, labels: set[str], wait_keep: float = 1.0, label_weight: dict[str, float] | None = None,
         cache_dir: Path | None = None, with_context: bool = False):
    """Rows: frame i-1 -> action at frame i. With ``wait_keep`` < 1 only that fraction of the wait rows
    (~94% of all rows) is kept, each weighted 1 / wait_keep, so more games fit in memory.
    ``label_weight`` multiplies the weight of every row of a label (a small, newer teacher among many older games)."""
    paths = [p for p in frame_paths(data) if p.parent.name in labels]
    # Decide the kept rows first and preallocate exactly: a list of per-row tensors plus torch.stack needs
    # twice the memory, which on 887k rows (~10 GB of boards) took down the 23 GB WSL VM (2026-10-15).
    rng = random.Random(0)
    keep = {}
    cached = {}
    for path in paths:
        mask = []
        if cache_dir is not None:
            cached[path] = cache_episode(path, cache_dir)
            with np.load(cached[path], allow_pickle=False) as stored:
                mask = [bool(hint) or rng.random() < wait_keep for hint in stored["hints"]]
            keep[path] = mask
            continue
        for line in frame_bytes(path).decode().splitlines():
            # The target is the last field: the DAgger label if there is one, else the executed action.
            tail = line[max(line.rfind('"label":'), line.rfind('"action":')):]
            mask.append('"type":"plant"' in tail or '"targets":' in line or rng.random() < wait_keep)
        keep[path] = mask
    capacity = sum(map(sum, keep.values()))
    boards = scalars = None
    plant, packet, cell, weight, seed_of, values = [], [], [], [], [], []
    n = 0
    for path in paths:
        seed = int(path.parent.parent.name.split("_")[-1])
        scale = (label_weight or {}).get(path.parent.name, 1.0)
        if path in cached:
            with np.load(cached[path], allow_pickle=False) as stored:
                selected = np.flatnonzero(np.asarray(keep[path])[stored["rows"]])
                count = len(selected)
                rows_scalars = stored["scalars"][selected]
                if with_context:
                    ctx = _episode_context(path)
                    rows_scalars = np.concatenate([rows_scalars, np.broadcast_to(ctx, (count, len(ctx)))], axis=1)
                rows_boards = stored["boards"]
                if boards is None:
                    boards = _board_store((capacity, *rows_boards.shape[1:]), cache_dir)
                    scalars = torch.empty((capacity, rows_scalars.shape[1]), dtype=torch.float32)
                boards[n:n + count] = torch.from_numpy(rows_boards[selected])
                scalars[n:n + count] = torch.from_numpy(np.ascontiguousarray(rows_scalars))
                types = stored["plant"][selected].tolist()
                plant.extend(types)
                packet.extend(stored["packet"][selected].tolist())
                cell.extend(stored["cell"][selected].tolist())
                vals = stored["values"][selected]
                values.extend(vals.tolist())
                searched = (~np.isnan(vals)).any(1).tolist()
                weight.extend(scale * (1.0 if kind or done else 1.0 / wait_keep) for kind, done in zip(types, searched))
                seed_of.extend([seed] * count)
                n += count
            continue
        frames = read_frames(path)
        ctx = _episode_context(path) if with_context else None
        for i in range(1, len(frames)):
            action = frames[i].get("label") or frames[i].get("action")
            if not action or frames[i].get("action_rejected") or not keep[path][i]:
                continue
            b, s = features(frames[i - 1], frames[i - 2] if i >= 2 else None, ctx)
            if boards is None:
                boards = torch.empty((capacity, *b.shape), dtype=torch.half)
                scalars = torch.empty((capacity, *s.shape), dtype=s.dtype)
            boards[n] = b; scalars[n] = s; n += 1
            seed_of.append(seed)
            is_plant = action["type"] == "plant"
            plant.append(float(is_plant))
            weight.append(scale * (1.0 if is_plant or frames[i].get("targets") else 1.0 / wait_keep))
            packet.append(action["packet"] if is_plant else -1)
            cell.append(action["row"] * LAWN_COLS + action["col"] if is_plant else -1)
            values.append(target_values(frames[i].get("targets")))
    return (boards[:n], scalars[:n], torch.tensor(plant), torch.tensor(packet), torch.tensor(cell),
            torch.tensor(weight), seed_of, torch.tensor(values, dtype=torch.float32).reshape(-1, PACKETS + 1))


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
    p.add_argument("--context", action="store_true",
                   help="the student also reads the task context (zombie-count multiplier, roster) from meta.json")
    p.add_argument("--soft-weight", type=float, default=0.0,
                   help="also match the search's preference over its candidates (wait / each card): "
                        "KL(softmax(values / --soft-temperature) || student), on rows whose frame has search targets")
    p.add_argument("--soft-temperature", type=float, default=0.05, help="in win-probability units")
    p.add_argument("--spread-weight", type=float, default=0.0,
                   help="cost-weighted imitation: a searched row (>= 2 candidates) weighs (1 + x * spread / mean spread) / (1 + x) (at most 5), "
                        "spread = best minus worst candidate value, so decisions where the choice matters count more")
    p.add_argument("--save-epochs", type=int, default=0,
                   help="also keep the last N epochs as epoch_XX.pt (candidates for scripts/select_student.py)")
    p.add_argument("--init", help="start from this student / fast_ppo checkpoint (same width and pool) instead of random")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="cache exact per-episode features (--no-cache reads JSON directly)")
    p.add_argument("--cache-dir", type=Path, help="shared feature cache (default: .feature_cache beside the output directory)")
    args = p.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    started = time.time()
    label_weight = {k: float(v) for k, v in (kv.split("=") for kv in args.label_weight.split(",") if kv)}
    boards, scalars, plant, packet, cell, weight, seed_of, values = load(args.data, set(args.labels.split(",")),
                                                                 args.wait_keep, label_weight,
                                                                 (args.cache_dir or args.out.parent / ".feature_cache") if args.cache else None,
                                                                 args.context)
    load_seconds = time.time() - started
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
    if args.spread_weight:
        # 2026-10-17: only ~3% of rows are the teacher overriding the student, and those carry what the student lacks;
        # plain cross-entropy weighs a trivial wait as much as a decision worth 10 points of win rate.
        known = ~torch.isnan(values)
        compared = known.sum(1) >= 2
        spread = (values.nan_to_num(-1e9).amax(1) - values.nan_to_num(1e9).amin(1))[compared]
        # capped: a few spreads near 1 are games decided inside the horizon, not choices 30x more important
        weight[compared] *= ((1 + args.spread_weight * spread / spread.mean()) / (1 + args.spread_weight)).clamp(max=5.0)
        print(f"spread weights: {int(compared.sum())} compared rows, mean spread {float(spread.mean()):.4f}, "
              f"weight range {float(weight[compared].min()):.2f}-{float(weight[compared].max()):.2f}", flush=True)
    net = RolloutNet(args.width, args.pool, args.context)
    if args.init:
        from pvz_rollout_policy import load_student
        net.load_state_dict(load_student(args.init).state_dict())
    net = net.to(args.device)
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
        if args.soft_weight:
            v = values[idx].to(args.device)
            known = ~torch.isnan(v)
            soft = known.sum(1) >= 2                     # a preference needs at least two candidates
            if soft.any():
                vk, kk = v[soft], known[soft]
                # student: log p(wait) = log sigmoid(-plant), log p(card k) = log sigmoid(plant) + log softmax(packet)_k,
                # renormalised over the candidates the search compared
                student = torch.cat([F.logsigmoid(out["plant"][soft])[:, None] + F.log_softmax(out["packet"][soft], 1),
                                     F.logsigmoid(-out["plant"][soft])[:, None]], dim=1)
                student = torch.log_softmax(student.masked_fill(~kk, -1e9), 1)
                teacher = torch.softmax((vk / args.soft_temperature).masked_fill(~kk, -1e9), 1)
                kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).masked_fill(~kk, 0).sum(1)
                ws = w[soft]
                loss = loss + args.soft_weight * (kl * ws).sum() / ws.sum()
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

    def checkpoint() -> dict:
        return {"state_dict": {k: v.cpu() for k, v in net.state_dict().items()}, "labels": args.labels, "width": args.width,
                "label_weight": label_weight, "pool": args.pool, "packet_balance": args.packet_balance,
                "context": args.context, "soft_weight": args.soft_weight, "spread_weight": args.spread_weight, "history": list(history)}

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
        if args.save_epochs and epoch > args.epochs - args.save_epochs:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint(), args.out / f"epoch_{epoch:02d}.pt")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(), args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps({"history": history, "seconds": time.time() - started,
                                                       "load_seconds": load_seconds}, indent=1))
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
