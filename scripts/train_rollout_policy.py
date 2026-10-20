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
import math
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
from pvz_frame_store import (CELL_CANDIDATES, cache_candidates, cache_episode, candidate_cells, frame_bytes,  # noqa: E402
                              frame_paths, read_frames, target_values)


def _episode_context(path: Path) -> np.ndarray:
    """The task context of one archive (meta.json, written by search_play or scripts/backfill_frame_context.py)."""
    from pvz_value_net import context_vector
    meta = json.loads((path.parent / "meta.json").read_text())
    if "context" not in meta:
        raise SystemExit(f"{path.parent} has no task context: run scripts/backfill_frame_context.py first")
    return context_vector(meta["context"]["multiplier"], meta["context"]["roster"])


def _board_store(shape, cache_dir: Path) -> torch.Tensor:
    """Legacy float16 file-backed boards for --dense-boards; unlinked at once and freed on exit."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"boards_{os.getpid()}.npy"
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    path.unlink()
    return torch.from_numpy(store)


class CompactBoards:
    """CSR rows: uint16 element indices, values (float16 boards, float32 scalars -- lossless either way), int64 row offsets.
    Scalars are 84% zeros, so 488 bytes a row become ~120 (2026-10-19)."""

    def __init__(self, shape, dtype=np.float16):
        self.shape = (0, *shape)
        self.dtype = np.dtype(dtype)
        self.indices = bytearray()
        self.values = bytearray()
        self.offsets = bytearray(np.zeros(1, dtype=np.int64).tobytes())
        self.nnz = 0

    def append(self, boards):
        flat = np.asarray(boards, dtype=self.dtype).reshape(len(boards), -1)
        rows, columns = np.nonzero(flat.view(f"u{self.dtype.itemsize}"))  # bit test: keeps -0.0
        counts = np.bincount(rows, minlength=len(boards))
        self.indices.extend(columns.astype(np.uint16).tobytes())
        self.values.extend(flat[rows, columns].tobytes())
        self.offsets.extend((counts.cumsum(dtype=np.int64) + self.nnz).tobytes())
        self.nnz += len(columns)
        self.shape = (self.shape[0] + len(boards), *self.shape[1:])

    def finish(self):
        self.indices = np.frombuffer(self.indices, dtype=np.uint16)
        self.values = np.frombuffer(self.values, dtype=self.dtype)
        self.offsets = np.frombuffer(self.offsets, dtype=np.int64)
        return self

    @property
    def nbytes(self):
        return self.indices.nbytes + self.values.nbytes + self.offsets.nbytes

    def batch(self, idx, device):
        selected = np.asarray(idx)
        starts = self.offsets[selected]
        counts = self.offsets[selected + 1] - starts
        ends = counts.cumsum()
        positions = np.arange(ends[-1] if len(ends) else 0) + np.repeat(starts - (ends - counts), counts)
        size = int(np.prod(self.shape[1:]))
        destinations = np.repeat(np.arange(len(selected)) * size, counts) + self.indices[positions]
        output = torch.zeros(len(selected) * size, dtype=torch.float32, device=device)
        output.index_put_((torch.from_numpy(destinations).to(device),),
                          torch.from_numpy(self.values[positions]).to(device=device, dtype=torch.float32))
        return output.reshape(len(selected), *self.shape[1:])


def load(data: Path, labels: set[str], wait_keep: float = 1.0, label_weight: dict[str, float] | None = None,
         cache_dir: Path | None = None, with_context: bool = False, obs: str = "v1", precompute_workers: int = 1,
         dense_boards: bool = False, cells: bool = False, row_observer=None):
    """Rows: frame i-1 -> action at frame i. With ``wait_keep`` < 1 only that fraction of the wait rows
    (~94% of all rows) is kept, each weighted 1 / wait_keep, so more games fit in memory.
    ``label_weight`` multiplies the weight of every row of a label (a small, newer teacher among many older games)."""
    paths = [p for p in frame_paths(data) if p.parent.name in labels]
    # Decide the kept rows first and preallocate exactly: a list of per-row tensors plus torch.stack needs
    # twice the memory, which on 887k rows (~10 GB of boards) took down the 23 GB WSL VM (2026-10-15).
    rng = random.Random(0)
    keep = {}
    cached = {}
    cell_cached = {}
    if cache_dir is not None and precompute_workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=precompute_workers, mp_context=mp.get_context("spawn")) as pool:
            cached = dict(zip(paths, pool.map(partial(cache_episode, directory=cache_dir, obs=obs), paths, chunksize=8)))
            if cells:
                cell_cached = dict(zip(paths, pool.map(partial(cache_candidates, directory=cache_dir), paths, chunksize=8)))
    for path in paths:
        mask = []
        if cache_dir is not None:
            if path not in cached:
                cached[path] = cache_episode(path, cache_dir, obs)
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
    # Per-row labels go straight into preallocated arrays: as Python lists (8 floats a row for `values`) they cost
    # ~2 GB of objects on 6.5M rows and the freed arenas stayed in the training process (13 GB RSS, 2026-10-19).
    plant, weight = np.empty(capacity, np.float32), np.empty(capacity, np.float32)
    packet, cell = np.empty(capacity, np.int64), np.empty(capacity, np.int64)
    values = np.empty((capacity, PACKETS + 1), np.float32)
    seed_of = []
    cands = np.full((capacity, CELL_CANDIDATES, 3), np.nan, dtype=np.float16) if cells else None
    n = 0
    for path in paths:
        seed = int(path.parent.parent.name.split("_")[-1])
        scale = (label_weight or {}).get(path.parent.name, 1.0)
        if path in cached:
            with np.load(cached[path], allow_pickle=False) as stored:
                selected = np.flatnonzero(np.asarray(keep[path])[stored["rows"]])
                count = len(selected)
                if row_observer is not None:
                    row_observer(path, stored["rows"][selected])
                rows_scalars = stored["scalars"][selected]
                if with_context:
                    ctx = _episode_context(path)
                    context_rows = np.broadcast_to(ctx, (count, len(ctx)))
                    rows_scalars = np.concatenate([rows_scalars[:, :115], context_rows, rows_scalars[:, 115:]], axis=1)
                rows_boards = stored["boards"]
                if boards is None:
                    boards = (_board_store((capacity, *rows_boards.shape[1:]), cache_dir) if dense_boards
                              else CompactBoards(rows_boards.shape[1:]))
                    scalars = (torch.empty((capacity, rows_scalars.shape[1]), dtype=torch.float32) if dense_boards
                               else CompactBoards(rows_scalars.shape[1:], np.float32))
                if dense_boards:
                    boards[n:n + count] = torch.from_numpy(rows_boards[selected])
                elif count:
                    boards.append(rows_boards[selected])
                if dense_boards:
                    scalars[n:n + count] = torch.from_numpy(np.ascontiguousarray(rows_scalars))
                elif count:
                    scalars.append(rows_scalars)
                types = stored["plant"][selected]
                plant[n:n + count] = types
                packet[n:n + count] = stored["packet"][selected]
                cell[n:n + count] = stored["cell"][selected]
                vals = stored["values"][selected]
                if cells:
                    if path not in cell_cached:
                        cell_cached[path] = cache_candidates(path, cache_dir)
                    with np.load(cell_cached[path], allow_pickle=False) as side:
                        assert np.array_equal(side["rows"], stored["rows"]), path
                        cands[n:n + count] = side["cands"][selected]
                values[n:n + count] = vals
                weight[n:n + count] = np.where((types != 0) | (~np.isnan(vals)).any(1), 1.0, 1.0 / wait_keep) * scale
                seed_of.extend([seed] * count)
                n += count
            continue
        frames = read_frames(path)
        ctx = _episode_context(path) if with_context else None
        if obs != "v1":
            from pvz_student_observation import StudentHistory
            history = StudentHistory(obs)
        for i in range(1, len(frames)):
            if obs != "v1":
                b, s = history.features(frames[i - 1], frames[i - 2] if i >= 2 else None, ctx)
            action = frames[i].get("label") or frames[i].get("action")
            if not action or frames[i].get("action_rejected") or not keep[path][i]:
                continue
            if obs == "v1":
                b, s = features(frames[i - 1], frames[i - 2] if i >= 2 else None, ctx)
            if boards is None:
                boards = (torch.empty((capacity, *b.shape), dtype=torch.half) if dense_boards
                          else CompactBoards(b.shape))
                scalars = (torch.empty((capacity, *s.shape), dtype=s.dtype) if dense_boards
                           else CompactBoards(s.shape, np.float32))
            if dense_boards:
                boards[n] = b
                scalars[n] = s
            else:
                boards.append(b.numpy()[None])
                scalars.append(s.numpy()[None])
            n += 1
            seed_of.append(seed)
            is_plant = action["type"] == "plant"
            plant[n - 1] = float(is_plant)
            weight[n - 1] = scale * (1.0 if is_plant or frames[i].get("targets") else 1.0 / wait_keep)
            packet[n - 1] = action["packet"] if is_plant else -1
            cell[n - 1] = action["row"] * LAWN_COLS + action["col"] if is_plant else -1
            values[n - 1] = target_values(frames[i].get("targets"))
            if cells:
                cands[n - 1] = candidate_cells(frames[i].get("targets"))
    return (*((boards[:n], scalars[:n]) if dense_boards else (boards.finish(), scalars.finish())), *(torch.from_numpy(a[:n]) for a in (plant, packet, cell, weight)),
            seed_of, torch.from_numpy(values[:n]),
            None if cands is None else torch.from_numpy(cands[:n]))


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
    p.add_argument("--obs", choices=["v1", "v2lite", "v2"], default="v1", help="student feature version")
    p.add_argument("--soft-weight", type=float, default=0.0,
                   help="also match the search's preference over its candidates (wait / each card): "
                        "KL(softmax(values / --soft-temperature) || student), on rows whose frame has search targets")
    p.add_argument("--soft-temperature", type=float, default=0.05, help="in win-probability units")
    p.add_argument("--cell-soft-weight", type=float, default=0.0,
                   help="cell-level soft labels: on planted rows whose search compared >= 2 cells of the labelled card, "
                        "add KL(softmax(cell values / --soft-temperature) || student cell distribution over those cells). "
                        "The hard label teaches only the chosen cell; this also teaches how much worse the other cells were")
    p.add_argument("--spread-weight", type=float, default=0.0,
                   help="cost-weighted imitation: a searched row (>= 2 candidates) weighs (1 + x * spread / mean spread) / (1 + x) (at most --spread-cap), "
                        "spread = best minus worst candidate value, so decisions where the choice matters count more")
    p.add_argument("--spread-cap", type=float, default=5.0, help="upper bound of the --spread-weight factor")
    p.add_argument("--save-epochs", type=int, default=0,
                   help="also keep the last N epochs as epoch_XX.pt (candidates for scripts/select_student.py)")
    p.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant",
                   help="cosine: learning rate falls from 2e-3 to 0 over the run (per batch), so the final weights are not "
                        "a snapshot of a still-bouncing optimiser; constant = the old behaviour")
    p.add_argument("--ema", type=float, default=0.0,
                   help="keep an exponential moving average of the weights with this decay (e.g. 0.999, per batch); "
                        "validation and the saved model use the average. 0 = off (the old behaviour)")
    p.add_argument("--init", help="start from this student / fast_ppo checkpoint (same width and pool) instead of random")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True, help="cache exact per-episode features (--no-cache reads JSON directly)")
    p.add_argument("--cache-dir", type=Path, help="shared feature cache (default: .feature_cache beside the output directory)")
    p.add_argument("--precompute-workers", type=int, default=1, help="parallel per-episode feature cache workers")
    p.add_argument("--dense-boards", action="store_true", help="use the legacy dense board store instead of in-memory CSR")
    p.add_argument("--row-weights", type=Path, help="optional checksummed per-row experiment manifest; cached rows required, default off")
    p.add_argument("--max-steps", type=int, help="optional optimizer-step budget and cosine length; default all epochs")
    args = p.parse_args()
    if args.row_weights and not args.cache:
        p.error("--row-weights requires --cache")
    if args.max_steps is not None and args.max_steps<=0:
        p.error("--max-steps must be positive")
    row_plan = None
    if args.row_weights:
        from pvz_training_rows import RowWeights
        row_plan = RowWeights(args.row_weights, args.data)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    started = time.time()
    label_weight = {k: float(v) for k, v in (kv.split("=") for kv in args.label_weight.split(",") if kv)}
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = load(args.data, set(args.labels.split(",")),
                                                                 args.wait_keep, label_weight,
                                                                 (args.cache_dir or args.out.parent / ".feature_cache") if args.cache else None,
                                                                 args.context, args.obs, args.precompute_workers, args.dense_boards,
                                                                 cells=bool(args.cell_soft_weight),
                                                                 row_observer=row_plan.observe if row_plan else None)
    load_seconds = time.time() - started
    seeds = sorted(set(seed_of)); random.Random(0).shuffle(seeds)
    val_seeds = set(seeds[: len(seeds) // 10])
    is_val = torch.tensor([s in val_seeds for s in seed_of])
    if row_plan:
        row_plan.finish(len(plant), is_val.numpy())
    print(f"{len(plant)} decisions ({int(plant.sum())} plants), {len(seeds)} seeds, "
          f"{int(is_val.sum())} validation rows", flush=True)
    memory = {"rows": len(plant), "board_bytes": boards.numel() * boards.element_size() if args.dense_boards else boards.nbytes,
              "scalar_bytes": scalars.numel() * scalars.element_size() if args.dense_boards else scalars.nbytes,
              "tensor_bytes": sum(tensor.numel() * tensor.element_size()
                                  for tensor in (plant, packet, cell, weight, values, is_val))}
    print(json.dumps({"data_memory": memory}), flush=True)

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
        weight[compared] *= ((1 + args.spread_weight * spread / spread.mean()) / (1 + args.spread_weight)).clamp(max=args.spread_cap)
        print(f"spread weights: {int(compared.sum())} compared rows, mean spread {float(spread.mean()):.4f}, "
              f"weight range {float(weight[compared].min()):.2f}-{float(weight[compared].max()):.2f}", flush=True)
    if row_plan:
        row_plan.apply(weight)
        print(json.dumps({"row_intervention":row_plan.metadata(),
                          "positive_weight_rows":int((weight>0).sum())}),flush=True)
    net = RolloutNet(args.width, args.pool, args.context, args.obs)
    if args.init:
        from pvz_rollout_policy import load_student
        initial = load_student(args.init)
        weights = initial.state_dict()
        if args.obs == "v2" and initial.obs == "v1" and args.context == initial.context:
            weights["trunk.0.weight"] = F.pad(weights["trunk.0.weight"], (0, 0, 0, 0, 0, 9))
            weights["scalar.0.weight"] = F.pad(weights["scalar.0.weight"], (0, 7))
        net.load_state_dict(weights)
    net = net.to(args.device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    waits = float(weight[plant < 0.5].sum())
    pos_weight = torch.tensor([waits / max(1.0, float(plant.sum()))], device=args.device).sqrt()

    ema = (torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema),
                                                use_buffers=True) if args.ema else None)

    def batch_loss(idx, train):
        b = boards[idx].float().to(args.device) if args.dense_boards else boards.batch(idx, args.device)
        s = scalars[idx].to(args.device) if args.dense_boards else scalars.batch(idx, args.device)
        y = plant[idx].to(args.device); pk = packet[idx].to(args.device); c = cell[idx].to(args.device)
        w = weight[idx].to(args.device)
        planted = y > 0.5
        out = (net if train or ema is None else ema.module)(b, s, pk.clamp(min=0))
        denom = w.sum().clamp_min(1e-30) if row_plan else w.sum()
        loss = (F.binary_cross_entropy_with_logits(out["plant"], y, pos_weight=pos_weight, reduction="none") * w
                ).sum() / denom
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
                soft_denom=ws.sum().clamp_min(1e-30) if row_plan else ws.sum()
                loss = loss + args.soft_weight * (kl * ws).sum() / soft_denom
        if args.cell_soft_weight and planted.any():
            cd = cands[idx].to(args.device).float()                       # (B, K, 3): packet, cell, value
            same = (cd[:, :, 0] == pk[:, None].float()) & ~torch.isnan(cd[:, :, 2]) & planted[:, None]
            rows = same.sum(1) >= 2                                       # >= 2 cells of the labelled card compared
            if rows.any():
                m = same[rows]
                logits = out["cell"][rows].gather(1, cd[rows][:, :, 1].clamp(min=0).long())
                student = torch.log_softmax(logits.masked_fill(~m, -1e9), 1)
                teacher = torch.softmax((cd[rows][:, :, 2].nan_to_num(0) / args.soft_temperature).masked_fill(~m, -1e9), 1)
                kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).masked_fill(~m, 0).sum(1)
                wc = w[rows]
                cell_denom=wc.sum().clamp_min(1e-30) if row_plan else wc.sum()
                loss = loss + args.cell_soft_weight * (kl * wc).sum() / cell_denom
        if planted.any():
            wp = w[planted]                               # all 1 unless --label-weight is used
            loss = loss + ((F.cross_entropy(out["packet"][planted], pk[planted], reduction="none") +
                            F.cross_entropy(out["cell"][planted], c[planted], reduction="none")) * wp).sum() / (wp.sum().clamp_min(1e-30) if row_plan else wp.sum())
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
    steps = args.epochs * -(-len(train_idx) // 2048)
    if args.max_steps is not None:
        if args.max_steps>steps:
            p.error("--max-steps exceeds the requested epochs")
        steps=args.max_steps
    schedule = (torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
                if args.lr_schedule == "cosine" else None)
    history = []

    def checkpoint() -> dict:
        return {"state_dict": {k: v.cpu() for k, v in (net if ema is None else ema.module).state_dict().items()}, "labels": args.labels, "width": args.width,
                "label_weight": label_weight, "pool": args.pool, "packet_balance": args.packet_balance,
                "context": args.context, "obs": args.obs, "soft_weight": args.soft_weight, "spread_weight": args.spread_weight, "spread_cap": args.spread_cap, "lr_schedule": args.lr_schedule, "ema": args.ema, "cell_soft_weight": args.cell_soft_weight,
                "history": list(history),
                **({"row_intervention":row_plan.metadata()} if row_plan else {}),
                **({"max_steps":args.max_steps,"optimizer_steps":completed_steps} if args.max_steps else {})}

    completed_steps=0
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        net.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        for start in range(0, len(perm), 2048):
            loss, _ = batch_loss(perm[start:start + 2048], True)
            opt.zero_grad(); loss.backward(); opt.step()
            if schedule is not None:
                schedule.step()
            if ema is not None:
                ema.update_parameters(net)
            completed_steps+=1
            if completed_steps>=steps:
                break
        net.eval()
        # Validate in chunks: one ~95k-row batch is a >2 GB tensor, which MPS silently gets wrong
        # (2026-10-15: garbage val metrics while the model itself was fine on the CPU).
        with torch.no_grad():
            parts = [(len(chunk), *batch_loss(chunk, False)) for chunk in val_idx.split(4096)]
        total = sum(n for n, _l, _s in parts)
        vl = sum(float(l) * n for n, l, _s in parts) / total
        vs = {k: sum(st[k] * n for n, _l, st in parts) / total for k in parts[0][2]}
        history.append({"epoch": epoch, "val_loss": float(vl), **vs, "seconds": time.perf_counter() - epoch_started})
        print(json.dumps(history[-1]), flush=True)
        if args.save_epochs and epoch > args.epochs - args.save_epochs:
            args.out.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint(), args.out / f"epoch_{epoch:02d}.pt")
        if completed_steps>=steps:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint(), args.out / "model.pt")
    (args.out / "metrics.json").write_text(json.dumps({"history": history, "seconds": time.time() - started,
                                                       "load_seconds": load_seconds, "data_memory": memory}, indent=1))
    print(f"wrote {args.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
