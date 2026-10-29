#!/usr/bin/env python3
"""task164: 离线测数据组装 vs GPU 步吞吐（只读生产 segment，不写生产目录）。

只挑与"最新一份 segment"同格式的那一组（obs 版本变了，混读会被 read_segments 拒绝）。

用法：
  python3 scripts/task164_bench_load.py --segments 40 --batches 60 --mode both
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_rollout_policy as T  # noqa: E402

SEGMENTS = Path.home() / "pvz-logs" / "curriculum_state.pipeline" / "frames" / "packs" / "segments"
OBS_BY_CHANNELS = {101: "v2", 117: "v2timer", 105: "v2sleep"}


def signature(path: Path):
    with np.load(path, allow_pickle=False) as z:
        shape = tuple(int(n) for n in z["board_shape"])
        scalars = tuple(int(n) for n in z["scalar_shape"])
        files = tuple(sorted(z.files))
    return shape, scalars, files


def pick_group(segments: int) -> tuple[list[Path], str]:
    """Take the newest segment format, then the newest N files that match it."""
    paths = sorted(SEGMENTS.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no segments under {SEGMENTS}")
    want = signature(paths[-1])
    chosen = [p for p in reversed(paths) if signature(p) == want][:segments]
    return sorted(chosen), OBS_BY_CHANNELS[want[0][0]]


def gpu_utilization(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
            samples.append(float(out[0]))
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        stop.wait(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", type=int, default=40)
    parser.add_argument("--batches", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mode", choices=("assemble", "train", "both"), default="both")
    parser.add_argument("--sync-every", type=int, default=1, help="每 N 个 batch 同步一次 loss（生产是 1）")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    paths, obs = pick_group(args.segments)
    started = time.perf_counter()
    loaded = T.read_segments(paths)
    load_seconds = time.perf_counter() - started
    boards, scalars, plant, packet, cell, weight, seed_of, values, cands = loaded[:9]
    packet_info = loaded[9] if len(loaded) > 9 else None
    rows = len(plant)
    device = torch.device(args.device)
    print(json.dumps({"obs": obs, "segments": len(paths), "rows": rows,
                      "load_seconds": round(load_seconds, 1),
                      "rows_per_second_load": round(rows / max(load_seconds, 1e-9))}), flush=True)

    net = T.RolloutNet(64, "meanmax", False, obs, True, aux=False, packet_head="identity").to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    waits = float(weight[plant < 0.5].sum())
    pos_weight = torch.tensor([waits / max(1.0, float(plant.sum()))], device=device).sqrt()

    def assemble(idx):
        b = boards.batch(idx, device)
        s = scalars.batch(idx, device)
        y = plant[idx].to(device).float()
        pk = packet[idx].to(device).long()
        c = cell[idx].to(device).long()
        w = weight[idx].to(device)
        info = packet_info[idx].to(device) if packet_info is not None else None
        cd = cands[idx].to(device).float()
        return b, s, y, pk, c, w, info, cd

    def step(b, s, y, pk, c, w, info, cd, train=True):
        # 与 train_rollout_policy.batch_loss 同构（cell_soft_weight=3.0, soft_temperature=0.05）
        planted = y > 0.5
        out = net(b, s, pk.clamp(min=0), packet_info=info)
        denom = w.sum()
        loss = (F.binary_cross_entropy_with_logits(out["plant"], y, pos_weight=pos_weight, reduction="none") * w).sum() / denom
        if planted.any():
            same = (cd[:, :, 0] == pk[:, None].float()) & ~torch.isnan(cd[:, :, 2]) & planted[:, None]
            rows_mask = same.sum(1) >= 2
            if rows_mask.any():
                m = same[rows_mask]
                logits = out["cell"][rows_mask].gather(1, cd[rows_mask][:, :, 1].nan_to_num(0).clamp(min=0).long())
                student = torch.log_softmax(logits.masked_fill(~m, -1e9), 1)
                teacher = torch.softmax((cd[rows_mask][:, :, 2].nan_to_num(0) / 0.05).masked_fill(~m, -1e9), 1)
                kl = (teacher * (torch.log(teacher.clamp(min=1e-12)) - student)).masked_fill(~m, 0).sum(1)
                loss = loss + 3.0 * (kl * w[rows_mask]).sum() / w[rows_mask].sum()
            wp = w[planted]
            loss = loss + ((F.cross_entropy(out["packet"][planted], pk[planted], reduction="none") +
                            F.cross_entropy(out["cell"][planted], c[planted], reduction="none")) * wp).sum() / wp.sum()
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        return loss

    order = torch.randperm(rows)
    limit = min(rows, args.batches * args.batch_size)
    batches = [order[i:i + args.batch_size] for i in range(0, limit, args.batch_size)]
    if not batches:
        raise SystemExit("no batches")

    for idx in batches[:2]:                      # 预热（cuda context + cudnn autotune）
        step(*assemble(idx), train=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    stop = threading.Event()
    samples: list[float] = []
    thread = threading.Thread(target=gpu_utilization, args=(stop, samples), daemon=True)
    thread.start()

    assemble_times, step_times, total_times = [], [], []
    for idx in batches:
        t0 = time.perf_counter()
        parts = assemble(idx)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if args.mode != "assemble":
            loss = step(*parts, train=True)
            if args.sync_every and (len(total_times) + 1) % args.sync_every == 0:
                float(loss.detach())
                if device.type == "cuda":
                    torch.cuda.synchronize()   # 生产每 batch 都同步一次；--sync-every 0 只看 CPU 发射速率
        t2 = time.perf_counter()
        assemble_times.append(t1 - t0)
        step_times.append(t2 - t1)
        total_times.append(t2 - t0)

    if device.type == "cuda":
        torch.cuda.synchronize()
    stop.set(); thread.join(timeout=2)

    def stats(xs):
        return {"mean_ms": round(1000 * statistics.mean(xs), 2), "median_ms": round(1000 * statistics.median(xs), 2)}

    total_rows = sum(len(b) for b in batches)
    result = {
        "obs": obs,
        "mode": args.mode,
        "threads": args.threads,
        "batch_size": args.batch_size,
        "batches": len(batches),
        "rows": total_rows,
        "rows_per_second_end_to_end": round(total_rows / sum(total_times), 1),
        "assemble": stats(assemble_times),
        "step": stats(step_times),
        "total": stats(total_times),
        "assemble_share": round(sum(assemble_times) / sum(total_times), 3),
        "gpu_util_mean": round(statistics.mean(samples), 1) if samples else None,
        "gpu_util_samples": len(samples),
        "net_params": sum(p.numel() for p in net.parameters()),
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
