#!/usr/bin/env python3
"""把 `pvz_research.py` 的检查点保留策略应用到**已有的** run 上。

为什么需要它：策略是在 `pvz_research.save()` 里执行的，所以它只对**之后**产生的
检查点生效。在策略加上之前跑完的 run 会一直留着每个 update 一份的中间快照
（2026-10-01 实测：`reward_r0_seed1_v2` 堆了 54 个 `trained`，2.3 GB）。

这里刻意**复用** `pvz_research.prune_trained_checkpoints()`，而不是在别处重写一遍
删除逻辑 —— 两份实现必然漂移。曾经就有一个 shell 版本只保留 1 个 `trained`
（与策略的 8 不一致），而且不保护 `resume.json` 指向的文件。

用法：
    # 空跑（默认）：看每个 run 会删什么，不删任何东西
    ~/.venvs/ml/bin/python scripts/prune_research_checkpoints.py

    # 只看一个 run
    ~/.venvs/ml/bin/python scripts/prune_research_checkpoints.py --run reward_r0_seed1_v2

    # 真正执行
    ~/.venvs/ml/bin/python scripts/prune_research_checkpoints.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from pvz_research import TRAINED_CHECKPOINT_KEEP, prune_trained_checkpoints  # noqa: E402

RESEARCH_DIR = ROOT / "artifacts" / "research"


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def protected_paths(run_dir: Path) -> set[Path]:
    """绝不能被删的检查点：`training_state.json` 与 `resume.json` 指到的文件。

    `resume.json` 是权威的续跑判据 —— 实测四个 `reward_r*_v2` 的它都指向
    `boundary` 而不是最新的 `trained`，所以"只留最新一个 trained"这种写法
    会把它删掉。这里把两个文件里所有以 `.pt` 结尾的字符串值都收进来。
    """
    protected: set[Path] = set()
    for name in ("training_state.json", "resume.json"):
        try:
            payload = json.loads((run_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for value in _walk_strings(payload):
            if value.endswith(".pt"):
                protected.add((run_dir / value).resolve())
    return protected


def _walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_strings(value)


def plan(run_dir: Path, keep: int) -> tuple[list[Path], list[Path], set[Path]]:
    run_1 = run_dir / "runs" / "run_1"
    snapshots = sorted(run_1.glob("update_*_trained_*.pt")) if run_1.is_dir() else []
    protected = protected_paths(run_dir)
    survivors = ({path.resolve() for path in snapshots[-keep:]} if keep else set()) | protected
    doomed = [path for path in snapshots if path.resolve() not in survivors]
    return snapshots, doomed, protected


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", help="只处理这个 run（默认全部）")
    parser.add_argument("--keep", type=int, default=TRAINED_CHECKPOINT_KEEP,
                        help=f"保留最新的几个 trained（默认 {TRAINED_CHECKPOINT_KEEP}，"
                             "与 pvz_research.py 一致）")
    parser.add_argument("--apply", action="store_true", help="真正删除；不加则只空跑")
    args = parser.parse_args()

    if args.keep < 0:
        parser.error("--keep 不能为负")
    if args.run:
        runs = [RESEARCH_DIR / args.run]
        if not runs[0].is_dir():
            sys.exit(f"找不到 run：{runs[0]}")
    else:
        runs = sorted(p for p in RESEARCH_DIR.iterdir() if p.is_dir()) if RESEARCH_DIR.is_dir() else []

    print(f"保留策略: 最新 {args.keep} 个 trained + 所有被指针引用的检查点")
    print(f"模式    : {'*** 实际删除 ***' if args.apply else '空跑（不删任何东西）'}")
    print()

    total_files = 0
    total_bytes = 0
    for run_dir in runs:
        snapshots, doomed, protected = plan(run_dir, args.keep)
        if not snapshots:
            continue
        size = sum(path.stat().st_size for path in doomed)
        total_files += len(doomed)
        total_bytes += size
        kept = sorted(path.name for path in snapshots if path.resolve() not in {p.resolve() for p in doomed})
        print(f"{run_dir.name}")
        print(f"  trained {len(snapshots)} 个 → 删 {len(doomed)} 个（{_human(size)}）")
        print(f"  保留: {', '.join(kept) if kept else '（无）'}")
        if protected:
            print(f"  指针引用（永不删）: {', '.join(sorted(p.name for p in protected))}")

        # 硬约束：resume.json 指向的文件一旦落在删除列表里就中止。
        resume_target = None
        try:
            resume = json.loads((run_dir / "resume.json").read_text(encoding="utf-8"))
            target = resume.get("checkpoint")
            if isinstance(target, str):
                resume_target = (run_dir / target).resolve()
        except (OSError, ValueError):
            pass
        if resume_target is not None and resume_target in {p.resolve() for p in doomed}:
            sys.exit(f"  !! {run_dir.name}: resume.json 指向的文件在删除列表里，中止")

        if args.apply:
            removed = prune_trained_checkpoints(run_dir / "runs" / "run_1", args.keep, protected)
            print(f"  已删除 {len(removed)} 个")
            if resume_target is not None:
                print(f"  resume.json 指向的文件仍存在: {resume_target.is_file()}")
        print()

    print(f"合计：{total_files} 个文件 / {_human(total_bytes)}")
    if not args.apply:
        print("空跑结束，未删除。加 --apply 才真正执行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
