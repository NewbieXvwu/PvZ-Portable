#!/usr/bin/env python3
"""task164: 从生产日志拆分每轮耗时（只读）。

用法：python3 scripts/task164_round_timings.py [--root ~/pvz-logs/curriculum_state.pipeline]
输出：stdout 的 markdown 表 + 汇总（--json <path> 存原始数据）。

时间口径（全部来自文件时间戳，不靠推算）：
  采集 = curriculum.log 里该轮第一次出现 → trained.json 的 pack_started
  打包 = pack_started → train_started
  训练 = train_started → train_finished（4 个候选串行，含每候选的加载）
  挑选 = selection_started → selection_finished
  评估 = curriculum.log 的 "background round N evaluate done: Xs"
  轮次 = 该轮第一次出现 → "student activated at group boundary: round N"
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_first_json(path: Path):
    """pack.log / selection.log are JSONL: the first line carries the summary."""
    try:
        first = path.read_text(errors="replace").splitlines()[0]
        return json.loads(first)
    except (OSError, ValueError, IndexError):
        return None


def epoch_seconds(seed_dir: Path) -> list[float]:
    """train.log 每行一个 epoch 的 JSON；返回 seconds 列表。"""
    log = seed_dir / "train.log"
    if not log.is_file():
        return []
    out = []
    for line in log.read_text(errors="replace").splitlines():
        line = line.strip()
        if line.startswith('{"epoch"'):
            try:
                out.append(float(json.loads(line)["seconds"]))
            except (ValueError, KeyError):
                pass
    return out


def round_row(root: Path) -> dict | None:
    trained = load_json(root / "trained.json")
    if not trained:
        return None
    pack = load_first_json(root / "pack.log") or {}
    row = {
        "round": int(trained.get("round", -1)),
        "pack_seconds": None,
        "pack_rows": pack.get("rows"),
        "pack_built": pack.get("built"),
        "pack_reused": pack.get("reused"),
        "pack_workers": pack.get("workers"),
        "candidates": len(trained.get("candidates") or []),
        "families": sorted({c.get("recipe_family") for c in (trained.get("candidates") or []) if c.get("recipe_family")}),
        "candidate_seconds": [],
        "load_seconds": [],
        "epoch_seconds": [],
        "epoch_count": [],
        "rows": None,
        "pack_started": trained.get("pack_started"),
        "train_started": trained.get("train_started"),
        "train_finished": trained.get("train_finished"),
        "selection_started": trained.get("selection_started"),
        "selection_finished": trained.get("selection_finished"),
        "train_seconds": None,
        "selection_seconds": None,
    }
    if pack.get("seconds") is not None:
        row["pack_seconds"] = float(pack["seconds"])
    for candidate in trained.get("candidates") or []:
        if candidate.get("train_started") and candidate.get("train_finished"):
            row["candidate_seconds"].append(float(candidate["train_finished"]) - float(candidate["train_started"]))
    if trained.get("train_started") and trained.get("train_finished"):
        row["train_seconds"] = float(trained["train_finished"]) - float(trained["train_started"])
    if trained.get("selection_started") and trained.get("selection_finished"):
        row["selection_seconds"] = float(trained["selection_finished"]) - float(trained["selection_started"])
    for seed_dir in sorted((root / "seeds").glob("seed_*")):
        metrics = load_json(seed_dir / "student" / "metrics.json")
        epochs = epoch_seconds(seed_dir)
        if metrics:
            if metrics.get("load_seconds") is not None:
                row["load_seconds"].append(float(metrics["load_seconds"]))
            if metrics.get("data_memory", {}).get("rows"):
                row["rows"] = int(metrics["data_memory"]["rows"])
            if not epochs and metrics.get("history"):
                epochs = [float(h["seconds"]) for h in metrics["history"]]
        if epochs:
            row["epoch_seconds"].extend(epochs)
            row["epoch_count"].append(len(epochs))
    return row


def log_events(log_path: Path) -> dict[int, dict]:
    """每轮：首次出现时间、evaluate done 秒数、激活时间。"""
    events: dict[int, dict] = {}
    if not log_path.is_file():
        return events
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith("20"):
            continue
        stamp, _, rest = line.partition(" ")
        match = re.match(r"round (\d+) ", rest)
        if match:
            entry = events.setdefault(int(match.group(1)), {})
            entry.setdefault("first", stamp)
            entry["last"] = stamp
            continue
        match = re.match(r"background round (\d+) evaluate done: ([\d.]+)s", rest)
        if match:
            events.setdefault(int(match.group(1)), {})["evaluate"] = float(match.group(2))
            continue
        match = re.match(r"student activated at group boundary: round (\d+)", rest)
        if match:
            events.setdefault(int(match.group(1)), {})["activated"] = stamp
    return events


def iso(seconds) -> str | None:
    if not seconds:
        return None
    return datetime.fromtimestamp(float(seconds)).isoformat()


def seconds_between(a, b) -> float | None:
    if not a or not b:
        return None
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except ValueError:
        return None


def fmt(value, digits=0) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.home() / "pvz-logs" / "curriculum_state.pipeline"))
    parser.add_argument("--log", default=str(Path.home() / "pvz-logs" / "curriculum.log"))
    parser.add_argument("--json")
    args = parser.parse_args()
    root = Path(args.root)
    events = log_events(Path(args.log))

    rows = [row for row in (round_row(path) for path in sorted(root.glob("round_*"))) if row]
    for row in rows:
        event = events.get(row["round"], {})
        row["collection_seconds"] = seconds_between(event.get("first"), iso(row["pack_started"]))
        row["pack_wall"] = seconds_between(iso(row["pack_started"]), iso(row["train_started"]))
        row["evaluate_seconds"] = event.get("evaluate")
        row["cycle_seconds"] = seconds_between(event.get("first"), event.get("activated"))

    print("| 轮 | 候选 | 行数 | 采集 s | 打包 s | 训练 s | 挑选 s | 评估 s | 轮次 s | 训练占比 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        share = (row["train_seconds"] / row["cycle_seconds"]) if row["train_seconds"] and row["cycle_seconds"] else None
        print("| {round} | {candidates} | {rows} | {coll} | {pack} | {train} | {sel} | {eval} | {cycle} | {share} |".format(
            round=row["round"],
            candidates=row["candidates"],
            rows=f"{row['rows'] / 1e6:.1f}M" if row["rows"] else "-",
            coll=fmt(row["collection_seconds"]),
            pack=fmt(row["pack_wall"]),
            train=fmt(row["train_seconds"]),
            sel=fmt(row["selection_seconds"]),
            eval=fmt(row["evaluate_seconds"]),
            cycle=fmt(row["cycle_seconds"]),
            share=f"{share:.0%}" if share else "-",
        ))

    complete = [r for r in rows if r["train_seconds"]]
    print()
    print(f"轮数 {len(rows)}，有 trained.json 的 {len(complete)}")
    for label, key in (("采集", "collection_seconds"), ("打包墙钟", "pack_wall"), ("训练", "train_seconds"),
                       ("挑选", "selection_seconds"), ("评估", "evaluate_seconds"), ("轮次", "cycle_seconds")):
        values = [r[key] for r in complete if r.get(key)]
        if values:
            print(f"{label}: 中位 {statistics.median(values):.0f} s, 最大 {max(values):.0f} s, n={len(values)}")
    epochs = [s for r in complete for s in r["epoch_seconds"]]
    if epochs:
        print(f"单 epoch: 中位 {statistics.median(epochs):.0f} s, 最大 {max(epochs):.0f} s, n={len(epochs)}")
    loads = [s for r in complete for s in r["load_seconds"]]
    if loads:
        print(f"加载/候选: 中位 {statistics.median(loads):.0f} s, 最大 {max(loads):.0f} s, n={len(loads)}")

    unfinished = [path for path in sorted(root.glob("round_*"))
                  if not (path / "trained.json").is_file() and any(path.glob("seeds/seed_*/train.log"))]
    if unfinished:
        print()
        print("训练过但没写完 trained.json 的轮次（白训）：")
        for path in unfinished:
            counts = [len(epoch_seconds(seed)) for seed in sorted(path.glob("seeds/seed_*"))]
            print(f"  {path.name}: 各候选已完成 epoch {counts}")

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
