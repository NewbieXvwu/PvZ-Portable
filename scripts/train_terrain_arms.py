#!/usr/bin/env python3
"""task158: terrain interference arms on one stratified sample.

Arms (same rows, same epochs, same split):
  joint64        current architecture (RolloutNet, width 64)
  joint128       wider, ~3.1x parameters
  experts        five separate width-64 nets, each trained only on its terrain
  hard_moe       shared conv1+conv2; conv3 and the three heads per terrain (hard route)
  soft_moe       shared expert + 4 routed experts at mid-trunk, top-2, shared heads, load balance
  film           joint64 with a per-terrain FiLM modulation of the trunk (cheapest conditioning)

Training rows: validation seeds and whole unseen decks are excluded. Metrics are reported per
terrain on seen-deck validation rows and on unseen-deck rows (task153's unseen-deck split).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]

from pvz_rollout_policy import LAWN_COLS, PACKETS, RolloutNet  # noqa: E402
from pvz_student_observation import OBS_EXTRA  # noqa: E402
from pvz_training_rows import balance_levels  # noqa: E402
from pvz_value_net import BOARD_CHANNELS, COLS, ROWS, SCALARS  # noqa: E402
from pvz_value_net import PLANT_TYPES as PlantTypes  # noqa: E402
from train_rollout_policy import read_pack  # noqa: E402

TERRAINS = ("day", "night", "pool", "fog", "roof")
TERRAIN_COUNT = len(TERRAINS)
EXTRA = 2


class Pack:
    """The sampled pack: compact stores plus the per-row study columns."""

    def __init__(self, path: Path):
        loaded = read_pack(Path(path))
        (self.boards, self.scalars, self.plant, self.packet, self.cell, self.weight,
         self.seed_of, self.values, self.cands) = loaded[:9]
        self.packet_info = loaded[9] if len(loaded) > 9 else None
        extra = np.load(path, allow_pickle=False)
        self.terrain = np.asarray(extra["terrain"])
        self.episode = np.asarray(extra["episode"])
        self.level = np.asarray(extra["level"])
        self.deck = np.asarray(extra["deck"])
        self.holdout = np.asarray(extra["holdout_deck"]).astype(bool)
        self.deck_names = [str(name) for name in extra["deck_names"]]
        self.meta = json.loads(str(extra["meta"]))
        self.weight = self.weight.clone()
        self.rows = len(self.plant)

    def boards_of(self, index):
        return View(self.boards, np.asarray(index, dtype=np.int64))

    def scalars_of(self, index):
        return View(self.scalars, np.asarray(index, dtype=np.int64))

    def packets_of(self, index):
        return PacketView(self.packet_info, np.asarray(index, dtype=np.int64))


class View:
    def __init__(self, parent, index):
        self.parent, self.index = parent, index

    def batch(self, local, device):
        return self.parent.batch(self.index[np.asarray(local)], device)


class PacketView:
    def __init__(self, parent, index):
        self.parent, self.index = parent, index

    def __getitem__(self, local):
        return self.parent[self.index[np.asarray(local)]]


def _global(pool: str, h, s):
    if pool == "meanmax":
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3)), s], dim=1)
    return torch.cat([h.mean(dim=(2, 3)), s], dim=1)


def identity_scalars(scalars: torch.Tensor, packet_info: torch.Tensor) -> torch.Tensor:
    """RolloutNet.encode's identity-head input transform, applied verbatim in the routed arms.

    Without it the experts/hard_moe/soft_moe/film arms would read the legacy per-slot packet
    block while the joint arms read the per-kind aggregate -- an input advantage, not routing
    (found by the split-export equivalence check, task158)."""
    if packet_info is None or packet_info.ndim != 3 or packet_info.shape[1:] != (PACKETS, 4):
        raise ValueError('identity packet head requires [batch,10,4] public packet_info')
    result = scalars.clone()
    result[:, 5:113] = 0
    kinds = packet_info[:, :, 0].long()
    valid = kinds >= 0
    ready = packet_info[:, :, 1] * packet_info[:, :, 3]
    for start, values in ((5, ready), (59, packet_info[:, :, 2])):
        result[:, start:start + PlantTypes].scatter_reduce_(
            1, kinds.clamp(min=0), values.masked_fill(~valid, 0), reduce='amax',
            include_self=True)
    return result


class _Heads(nn.Module):
    """The three heads of RolloutNet, parameterised once for reuse by the routed arms."""

    def __init__(self, width, g_dim, cells, e=0):
        super().__init__()
        self.plant = nn.Linear(g_dim, 1)
        self.packet = nn.Sequential(nn.Linear(g_dim + e + 3, width), nn.ReLU(), nn.Linear(width, 1))
        self.packet_embedding = nn.Embedding(PlantTypes, e)
        self.cell = nn.Sequential(nn.Conv2d(width + e + g_dim, width, 1), nn.ReLU(),
                                  nn.Conv2d(width, 1, 1))

    def plant_logit(self, g):
        return self.plant(g).squeeze(1)

    def packet_logits(self, g, packet_info):
        kind = packet_info[:, :, 0].long()
        embedding = self.packet_embedding(kind.clamp(min=0))
        logits = self.packet(torch.cat([g[:, None, :].expand(-1, PACKETS, -1), embedding,
                                        packet_info[:, :, 1:]], dim=2)).squeeze(2)
        return logits.masked_fill(kind < 0, -1e9)

    def cell_logits(self, h, g, packet, packet_info):
        kind = packet_info[:, :, 0].long().gather(1, packet[:, None]).squeeze(1).clamp(min=0)
        e = self.packet_embedding(kind)[:, :, None, None].expand(-1, -1, ROWS, COLS)
        ctx = g[:, :, None, None].expand(-1, -1, ROWS, COLS)
        return self.cell(torch.cat([h, e, ctx], dim=1))[:, 0, :, :LAWN_COLS].flatten(1)


class HardMoE(nn.Module):
    def __init__(self, width=64, obs="v2timer", pool="meanmax", row_pool=True, terrains=TERRAIN_COUNT):
        super().__init__()
        g_dim = (3 if pool == "meanmax" else 2) * width
        self.pool, self.row_pool, self.obs = pool, row_pool, obs
        self.trunk_shared = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS + OBS_EXTRA[obs][0], width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.experts = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, (1, 5), padding=(0, 2)),
                                                    nn.ReLU()) for _ in range(terrains)])
        self.scalar = nn.Sequential(nn.Linear(SCALARS + EXTRA + OBS_EXTRA[obs][1], width), nn.ReLU())
        self.row_projection = nn.Conv2d(2 * width, width, 1) if row_pool else None
        self.heads = nn.ModuleList([_Heads(width, g_dim, width, e=width) for _ in range(terrains)])

    def forward(self, board, scalars, packet, terrain, packet_info):
        h = self.trunk_shared(board)
        s = self.scalar(identity_scalars(scalars, packet_info))
        plant = torch.empty(len(board), device=board.device)
        packet_logits = torch.empty(len(board), PACKETS, device=board.device)
        cell_logits = torch.empty(len(board), ROWS * LAWN_COLS, device=board.device)
        for index in torch.unique(terrain).tolist():
            select = terrain == index
            head = self.heads[index]
            part = self.experts[index](h[select])
            if self.row_pool:
                rows = torch.cat([part.mean(dim=3, keepdim=True), part.amax(dim=3, keepdim=True)], dim=1)
                part = F.relu(part + self.row_projection(rows))
            g = _global(self.pool, part, s[select])
            plant[select] = head.plant_logit(g)
            packet_logits[select] = head.packet_logits(g, packet_info[select])
            cell_logits[select] = head.cell_logits(part, g, packet[select], packet_info[select])
        return {"plant": plant, "packet": packet_logits, "cell": cell_logits}


class GatedMoE(nn.Module):
    """Shared expert + 4 routed experts at the trunk's third layer, top-2, shared heads."""

    def __init__(self, width=64, obs="v2timer", pool="meanmax", row_pool=True, routed=4):
        super().__init__()
        g_dim = (3 if pool == "meanmax" else 2) * width
        self.pool, self.row_pool, self.routed = pool, row_pool, routed
        self.trunk_shared = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS + OBS_EXTRA[obs][0], width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU())
        self.experts = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, (1, 5), padding=(0, 2)),
                                                    nn.ReLU()) for _ in range(1 + routed)])
        self.scalar = nn.Sequential(nn.Linear(SCALARS + EXTRA + OBS_EXTRA[obs][1], width), nn.ReLU())
        self.row_projection = nn.Conv2d(2 * width, width, 1) if row_pool else None
        self.router = nn.Linear(g_dim, routed)
        self.heads = _Heads(width, g_dim, width, e=width)
        self.last_gates = None

    def forward(self, board, scalars, packet, terrain, packet_info):
        h = self.trunk_shared(board)
        s = self.scalar(identity_scalars(scalars, packet_info))
        context = _global(self.pool, h, s)
        logits = self.router(context)
        top = logits.topk(2, dim=1).indices
        mask = torch.zeros_like(logits).scatter(1, top, 1.0)
        weights = torch.softmax(logits, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        part = self.experts[0](h)
        for index in range(self.routed):
            part = part + weights[:, index, None, None, None] * self.experts[index + 1](h)
        if self.row_pool:
            rows = torch.cat([part.mean(dim=3, keepdim=True), part.amax(dim=3, keepdim=True)], dim=1)
            part = F.relu(part + self.row_projection(rows))
        g = _global(self.pool, part, s)
        self.last_gates = logits.detach()
        return {"plant": self.heads.plant_logit(g),
                "packet": self.heads.packet_logits(g, packet_info),
                "cell": self.heads.cell_logits(part, g, packet, packet_info)}


class FiLMNet(nn.Module):
    """RolloutNet trunk with per-terrain scale/shift after the second convolution."""

    def __init__(self, width=64, obs="v2timer", pool="meanmax", row_pool=True, terrains=TERRAIN_COUNT):
        super().__init__()
        g_dim = (3 if pool == "meanmax" else 2) * width
        self.pool, self.row_pool = pool, row_pool
        self.trunk = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS + OBS_EXTRA[obs][0], width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
            nn.Conv2d(width, width, (1, 5), padding=(0, 2)), nn.ReLU())
        self.film = nn.Parameter(torch.zeros(2, terrains, width))
        self.scalar = nn.Sequential(nn.Linear(SCALARS + EXTRA + OBS_EXTRA[obs][1], width), nn.ReLU())
        self.row_projection = nn.Conv2d(2 * width, width, 1) if row_pool else None
        self.heads = _Heads(width, g_dim, width, e=width)

    def forward(self, board, scalars, packet, terrain, packet_info):
        h = self.trunk[:4](board)
        scale, shift = self.film[0][terrain], self.film[1][terrain]
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.trunk[4:](h)
        s = self.scalar(identity_scalars(scalars, packet_info))
        if self.row_pool:
            rows = torch.cat([h.mean(dim=3, keepdim=True), h.amax(dim=3, keepdim=True)], dim=1)
            h = F.relu(h + self.row_projection(rows))
        g = _global(self.pool, h, s)
        return {"plant": self.heads.plant_logit(g),
                "packet": self.heads.packet_logits(g, packet_info),
                "cell": self.heads.cell_logits(h, g, packet, packet_info)}


def build(arm: str, width: int, seed: int):
    torch.manual_seed(seed)
    if arm == "joint64":
        return RolloutNet(width, "meanmax", False, "v2timer", True, packet_head="identity")
    if arm == "joint128":
        return RolloutNet(128, "meanmax", False, "v2timer", True, packet_head="identity")
    if arm == "hard_moe":
        return HardMoE(width)
    if arm == "soft_moe":
        return GatedMoE(width)
    if arm == "film":
        return FiLMNet(width)
    if arm == "experts":
        return [RolloutNet(width, "meanmax", False, "v2timer", True, packet_head="identity")
                for _ in range(TERRAIN_COUNT)]
    raise SystemExit(f"unknown arm {arm}")


class Batch:
    """The row side of one arm: global indices plus compact-store views and dense columns."""

    def __init__(self, pack: Pack, index: np.ndarray):
        self.pack = pack
        self.index = np.asarray(index, dtype=np.int64)
        self.boards = pack.boards_of(self.index)
        self.scalars = pack.scalars_of(self.index)
        self.packet_info = pack.packets_of(self.index)

    def tensors(self, local, device):
        glob = self.index[np.asarray(local)]
        return (torch.from_numpy(glob).to(device),
                torch.from_numpy(np.asarray(self.pack.plant[glob], np.float32)).to(device),
                torch.from_numpy(np.asarray(self.pack.packet[glob], np.int64)).to(device),
                torch.from_numpy(np.asarray(self.pack.cell[glob], np.int64)).to(device),
                torch.from_numpy(np.asarray(self.pack.weight[glob], np.float32)).to(device),
                torch.from_numpy(np.asarray(self.pack.terrain[glob], np.int64)).to(device))


def forward_model(model, b, s, pk, terrain, info):
    if isinstance(model, RolloutNet):
        return model(b, s, pk, packet_info=info)
    return model(b, s, pk, terrain, info)


def loss_step(model, batch: Batch, local, pos_weight, args):
    device = args.device
    glob, y, pk, c, w, terrain = batch.tensors(local, device)
    b = batch.boards.batch(local, device)
    s = batch.scalars.batch(local, device)
    info = batch.packet_info[local].to(device)
    out = forward_model(model, b, s, pk.clamp(min=0), terrain, info)
    planted = y > 0.5
    denom = w.sum().clamp_min(1e-30)
    loss = (F.binary_cross_entropy_with_logits(out["plant"], y, pos_weight=pos_weight,
                                               reduction="none") * w).sum() / denom
    if planted.any():
        cd = batch.pack.cands[batch.index[np.asarray(local)]].to(device).float()
        same = (cd[:, :, 0] == pk[:, None].float()) & ~torch.isnan(cd[:, :, 2]) & planted[:, None]
        rows = same.sum(1) >= 2
        if rows.any():
            m = same[rows]
            logits = out["cell"][rows].gather(1, cd[rows][:, :, 1].nan_to_num(0).clamp(min=0).long())
            student = torch.log_softmax(logits.masked_fill(~m, -1e9), 1)
            teacher = torch.softmax((cd[rows][:, :, 2].nan_to_num(0) / args.soft_temperature)
                                    .masked_fill(~m, -1e9), 1)
            kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).masked_fill(~m, 0).sum(1)
            wc = w[rows]
            loss = loss + args.cell_soft_weight * (kl * wc).sum() / wc.sum().clamp_min(1e-30)
        wp = w[planted]
        loss = loss + ((F.cross_entropy(out["packet"][planted], pk[planted], reduction="none") +
                        F.cross_entropy(out["cell"][planted], c[planted], reduction="none")) * wp
                       ).sum() / wp.sum().clamp_min(1e-30)
    aux = 0.0
    if isinstance(model, GatedMoE):
        probs = torch.softmax(model.last_gates, dim=1)
        top = model.last_gates.topk(2, dim=1).indices
        mask = torch.zeros_like(model.last_gates).scatter(1, top, 1.0)
        f = mask.mean(0)
        p = probs.mean(0)
        aux = model.routed * (f * p).sum()
        loss = loss + 0.01 * aux
    return loss, out, aux


@torch.no_grad()
def evaluate(model, batch: Batch, args, chunk=4096):
    model.eval()
    device = args.device
    plant_logit, packet_pred, cell_pred, gates = [], [], [], []
    for start in range(0, len(batch.index), chunk):
        local = np.arange(start, min(start + chunk, len(batch.index)))
        glob, y, pk, c, w, terrain = batch.tensors(local, device)
        b = batch.boards.batch(local, device)
        s = batch.scalars.batch(local, device)
        info = batch.packet_info[local].to(device)
        out = forward_model(model, b, s, pk.clamp(min=0), terrain, info)
        plant_logit.append(out["plant"].float().cpu())
        packet_pred.append(out["packet"].argmax(1).cpu())
        cell_pred.append(out["cell"].argmax(1).cpu())
        if isinstance(model, GatedMoE):
            gates.append(model.last_gates.cpu())
    y = batch.pack.plant[batch.index]
    truth_packet = batch.pack.packet[batch.index].long()
    truth_cell = batch.pack.cell[batch.index].long()
    return dict(logit=torch.cat(plant_logit).numpy(), plant=y.numpy().astype(bool),
                packet_pred=torch.cat(packet_pred).numpy(), packet=truth_packet.numpy(),
                cell_pred=torch.cat(cell_pred).numpy(), cell=truth_cell.numpy(),
                gates=None if not gates else torch.cat(gates).numpy(),
                terrain=batch.pack.terrain[batch.index])


def roc_auc(score, label):
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    positives, negatives = int(label.sum()), int((~label).sum())
    if not positives or not negatives:
        return float("nan")
    uniq, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    average = np.cumsum(counts) - counts + (counts + 1) / 2
    ranks = average[inverse]
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def metrics(result):
    prediction = result["logit"] > 0
    truth = result["plant"]
    planted = truth
    waiting = ~truth
    full = np.where(planted,
                    prediction & (result["packet_pred"] == result["packet"]) &
                    (result["cell_pred"] == result["cell"]),
                    ~prediction)
    return dict(rows=int(len(truth)), plant_rate=float(planted.mean()),
                action_acc=float(full.mean()), auc=roc_auc(result["logit"], truth),
                false_plant=float(prediction[waiting].mean()) if waiting.any() else float("nan"),
                missed_plant=float((~prediction)[planted].mean()) if planted.any() else float("nan"),
                packet_acc=float((result["packet_pred"][planted] == result["packet"][planted]).mean())
                if planted.any() else float("nan"),
                cell_acc=float((result["cell_pred"][planted] == result["cell"][planted]).mean())
                if planted.any() else float("nan"))


def routing_report(gates, terrain):
    """Is the learned router just a terrain classifier?"""
    if gates is None:
        return None
    best = gates.argmax(1)
    table = {}
    purity = {}
    for index, name in enumerate(TERRAINS):
        mask = terrain == index
        if not mask.any():
            continue
        counts = np.bincount(best[mask], minlength=gates.shape[1])
        table[name] = [int(count) for count in counts]
        purity[name] = float(counts.max() / counts.sum())
    entropies = []
    for row in table.values():
        p = np.asarray(row, dtype=np.float64)
        p = p[p > 0] / p.sum()
        entropies.append(float(-(p * np.log(p)).sum() / math.log(gates.shape[1])))
    return dict(expert_by_terrain=table, dominant_share=purity,
                normalized_entropy=float(np.mean(entropies)) if entropies else None,
                distinct_dominant=len({int(np.argmax(row)) for row in table.values()}))


def prepare(pack: Pack, args):
    """Production-style row weights (spread, level balance) and the fixed split."""
    rng = np.random.default_rng(args.split_seed)
    seeds = np.asarray(sorted(set(pack.seed_of.tolist())))
    rng.shuffle(seeds)
    val_seeds = set(seeds[: max(1, int(len(seeds) * args.val_fraction))].tolist())
    is_val = np.isin(pack.seed_of, list(val_seeds))
    weight = pack.weight.numpy().copy()
    if args.spread_weight:
        compared_rows, spread = pack.values.spread()
        compared = np.asarray(compared_rows)
        if len(compared):
            factor = ((1 + args.spread_weight * spread / spread.mean()) /
                      (1 + args.spread_weight)).clip(max=args.spread_cap)
            weight[compared] *= factor
    level_multipliers = {}
    if args.level_balance:
        training = ~is_val & ~pack.holdout
        levels = {}
        for level in np.unique(pack.level):
            mask = (pack.level == level) & training
            rows, mass = int(mask.sum()), float(weight[mask].sum())
            if rows and mass > 0:
                levels[int(level)] = dict(rows=rows, original_weight=mass)
        if levels:
            summary = balance_levels(levels, args.level_balance, args.level_balance_power,
                                     args.level_balance_max)
            level_multipliers = {level: record["multiplier"] for level, record in levels.items()}
            for level, multiplier in level_multipliers.items():
                mask = (pack.level == level) & training
                weight[mask] *= multiplier
            print(json.dumps({"level_balance": {k: round(v, 3) for k, v in level_multipliers.items()},
                              "allocated_ratio": round(summary["allocated_weight_ratio"], 4)}), flush=True)
    pack.weight = torch.from_numpy(weight.astype(np.float32))
    return is_val


def train(arm: str, seed: int, pack: Pack, is_val: np.ndarray, args):
    device = args.device
    train_mask = (~is_val) & (~pack.holdout)
    val_seen = is_val & (~pack.holdout)
    unseen = pack.holdout
    counts = {name: dict(train=int((train_mask & (pack.terrain == i)).sum()),
                         val_seen=int((val_seen & (pack.terrain == i)).sum()),
                         unseen=int((unseen & (pack.terrain == i)).sum()))
              for i, name in enumerate(TERRAINS)}
    print(json.dumps({"row_counts": counts}), flush=True)
    if any(counts[name]["unseen"] == 0 for name in TERRAINS):
        print("warning: a terrain has no unseen-deck rows", flush=True)

    models = build(arm, args.width, seed)
    if arm == "experts":
        return train_experts(models, seed, pack, train_mask, val_seen, unseen, args)
    model = models.to(device)
    parameters = sum(p.numel() for p in model.parameters())
    print(json.dumps({"arm": arm, "seed": seed, "parameters": parameters}), flush=True)
    batch = Batch(pack, np.flatnonzero(train_mask))
    plant_flags = pack.plant.numpy()[batch.index] > 0.5
    waits = float(pack.weight.numpy()[batch.index][~plant_flags].sum())
    pos_weight = torch.tensor([waits / max(1.0, float(plant_flags.sum()))], device=device).sqrt()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * -(-len(batch.index) // args.batch_size)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))
    history, completed = [], 0
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        rng = np.random.default_rng(seed * 1000 + epoch)
        order = rng.permutation(len(batch.index))
        running = 0.0
        for start in range(0, len(order), args.batch_size):
            local = order[start:start + args.batch_size]
            loss, _out, _aux = loss_step(model, batch, local, pos_weight, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            running += float(loss.detach()) * len(local)
            completed += 1
        evaluation = evaluate_model(model, pack, val_seen, args)
        seen = {name: metrics(evaluation[name]) for name in TERRAINS if evaluation[name]}
        overall = metrics(evaluation["all"])
        history.append(dict(epoch=epoch, train_loss=running / len(batch.index),
                            val_loss=None, action_acc=overall["action_acc"],
                            auc=overall["auc"],
                            terrain_acc={name: seen[name]["action_acc"] for name in TERRAINS},
                            seconds=round(time.perf_counter() - started, 1)))
        print(json.dumps(history[-1]), flush=True)
    final = evaluate_model(model, pack, val_seen, args)
    unseen_results = evaluate_model(model, pack, unseen, args)
    report = dict(arm=arm, seed=seed, parameters=parameters, history=history,
                  seen={name: metrics(final[name]) for name in TERRAINS if final[name]},
                  seen_all=metrics(final["all"]),
                  unseen={name: metrics(unseen_results[name]) for name in TERRAINS if unseen_results[name]},
                  unseen_all=metrics(unseen_results["all"]),
                  routing=routing_report(final["all"].get("gates"), final["all"].get("terrain"))
                  if "gates" in final["all"] else None,
                  routing_unseen=routing_report(unseen_results["all"].get("gates"),
                                                unseen_results["all"].get("terrain")))
    save(report, model, args, seed)
    return report


def slice_result(result, mask):
    return {key: (value[mask] if isinstance(value, np.ndarray) else value)
            for key, value in result.items()}


def evaluate_model(model, pack, mask, args):
    rows = np.flatnonzero(mask)
    if not len(rows):
        return dict(all=None, **{name: None for name in TERRAINS})
    evaluated = evaluate(model, Batch(pack, rows), args)
    result = {"all": evaluated}
    for index, name in enumerate(TERRAINS):
        part = evaluated["terrain"] == index
        result[name] = slice_result(evaluated, part) if part.any() else None
    return result


def train_experts(models, seed, pack, train_mask, val_seen, unseen, args):
    device = args.device
    wanted = [name for name in TERRAINS if name in getattr(args, "terrains", TERRAINS)]
    tag = getattr(args, "tag", "")
    reports = []
    for index, name in enumerate(TERRAINS):
        if name not in wanted:
            continue
        model = models[index].to(device)
        parameters = sum(p.numel() for p in model.parameters())
        train_rows = np.flatnonzero(train_mask & (pack.terrain == index))
        if not len(train_rows):
            continue
        batch = Batch(pack, train_rows)
        planted = pack.plant.numpy()[train_rows] > 0.5
        waits = float(pack.weight.numpy()[train_rows][~planted].sum())
        pos_weight = torch.tensor([waits / max(1.0, float(planted.sum()))], device=device).sqrt()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        steps = args.epochs * -(-len(batch.index) // args.batch_size)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)))
        history = []
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            order = np.random.default_rng(seed * 1000 + epoch).permutation(len(batch.index))
            running = 0.0
            for start in range(0, len(order), args.batch_size):
                local = order[start:start + args.batch_size]
                loss, _out, _aux = loss_step(model, batch, local, pos_weight, args)
                optimizer.zero_grad(); loss.backward(); optimizer.step(); schedule.step()
                running += float(loss.detach()) * len(local)
            seen_rows = np.flatnonzero(val_seen & (pack.terrain == index))
            seen = metrics(evaluate(model, Batch(pack, seen_rows), args)) if len(seen_rows) else None
            history.append(dict(epoch=epoch, train_loss=running / len(batch.index),
                                action_acc=None if seen is None else seen["action_acc"],
                                seconds=round(time.perf_counter() - started, 1)))
            print(json.dumps({"terrain": name, **history[-1]}), flush=True)
        seen_rows = np.flatnonzero(val_seen & (pack.terrain == index))
        unseen_rows = np.flatnonzero(unseen & (pack.terrain == index))
        report = dict(arm="experts", seed=seed, terrain=name, parameters=parameters, history=history,
                      seen=metrics(evaluate(model, Batch(pack, seen_rows), args)) if len(seen_rows) else None,
                      unseen=metrics(evaluate(model, Batch(pack, unseen_rows), args)) if len(unseen_rows) else None)
        reports.append(report)
        save(report, model, args, seed, suffix=tag + name)
    combined = dict(arm="experts" + (f"_{tag}" if tag else ""), seed=seed,
                    parameters=sum(report["parameters"] for report in reports),
                    epochs=args.epochs,
                    seen={report["terrain"]: report["seen"] for report in reports},
                    unseen={report["terrain"]: report["unseen"] for report in reports})
    (args.out / f"experts{('_' + tag) if tag else ''}_s{seed}.json").write_text(
        json.dumps(combined, indent=1) + "\n")
    return combined


def save(report, model, args, seed, suffix=""):
    args.out.mkdir(parents=True, exist_ok=True)
    name = f"{report['arm']}_s{seed}" + (f"_{suffix}" if suffix else "")
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config": dict(arm=report["arm"], seed=seed, width=getattr(args, "width", 64))},
               args.out / f"{name}.pt")
    (args.out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", required=True,
                        choices=["joint64", "joint128", "experts", "hard_moe", "soft_moe", "film"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--cell-soft-weight", type=float, default=3.0)
    parser.add_argument("--soft-temperature", type=float, default=0.05)
    parser.add_argument("--spread-weight", type=float, default=8.0)
    parser.add_argument("--spread-cap", type=float, default=5.0)
    parser.add_argument("--level-balance", type=float, default=0.25)
    parser.add_argument("--level-balance-power", type=float, default=0.5)
    parser.add_argument("--level-balance-max", type=float, default=20.0)
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--max-rows", type=int, help="smoke: subsample training rows")
    parser.add_argument("--terrains", default=",".join(TERRAINS),
                        help="experts only: terrains to train (the equal-steps control trains a subset)")
    parser.add_argument("--tag", default="",
                        help="experts only: output name suffix, so a control run does not overwrite the arm")
    args = parser.parse_args()
    args.terrains = tuple(name for name in args.terrains.split(",") if name)

    started = time.time()
    pack = Pack(args.data)
    if args.max_rows:
        keep = np.zeros(pack.rows, bool)
        keep[np.random.default_rng(0).choice(pack.rows, min(args.max_rows, pack.rows), replace=False)] = True
        pack.holdout = pack.holdout | ~keep
        args.epochs = min(args.epochs, 2)
    is_val = prepare(pack, args)
    report = train(args.arm, args.seed, pack, is_val, args)
    print(json.dumps({"done": args.arm, "seed": args.seed, "seconds": round(time.time() - started, 1)}), flush=True)


if __name__ == "__main__":
    main()
