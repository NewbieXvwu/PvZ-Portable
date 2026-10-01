#!/usr/bin/env python3
"""用 Hugging Face Hub 在两台机器之间同步训练检查点与结论。

为什么不用 git：单个检查点约 43 MB，一个 run 会产出几十个；把它们提交进 git
会让仓库在几天内涨到 GB 级（2026-10-01 审计实测：`artifacts/` 被推了 2.0 GB，
而 `.gitignore` 认可的交付集只有 9.4 MB）。HF Hub 专为模型产物设计，支持断点续传、
SHA 校验和私有仓库，从任何机器一行拉取，不依赖临时 SSH 隧道。

分工：
  * **git**     —— 代码、配置、结论层（KB 级 json）。
  * **HF Hub**  —— 检查点（`.pt`）、评估分片（`.npz`）等大文件。

用法：
    export HF_TOKEN=hf_xxx
    export PVZ_HF_REPO=<你的用户名>/pvz-agent-artifacts

    python3 scripts/hf_sync.py ls                       # 看远端有哪些 run
    python3 scripts/hf_sync.py push reward_r0_seed0_v2  # 上传（只传最新检查点 + 结论层）
    python3 scripts/hf_sync.py pull reward_r0_seed0_v2  # 下载到本地
    python3 scripts/hf_sync.py pull-all                 # 下载全部

    # 想看会上传什么，先空跑：
    python3 scripts/hf_sync.py push reward_r0_seed0_v2 --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / "artifacts" / "research"

# 结论层：这些文件小（KB 级）且是真正的结论，永远随 run 一起同步。
EVIDENCE_GLOBS = [
    "learning_curve.json",
    "training_state.json",
    "experiment_config.json",
    "provenance.json",
    "resume.json",
    "evaluations/*.json.gz",
]

# 检查点：只同步最新一个（续跑用）。历史中间快照没有任何引用链指向。
CHECKPOINT_GLOB = "runs/run_1/*.pt"


def _require_hf():
    if importlib.util.find_spec("huggingface_hub") is None:
        sys.exit(
            "缺少 huggingface_hub。安装：\n"
            "  <venv>/bin/pip install 'huggingface_hub>=0.23'"
        )
    from huggingface_hub import HfApi, snapshot_download
    return HfApi, snapshot_download


def _repo_id(explicit: str | None) -> str:
    repo = explicit or os.environ.get("PVZ_HF_REPO", "")
    if not repo:
        sys.exit("未指定仓库。设置 PVZ_HF_REPO=<用户名>/<仓库名>，或传 --repo。")
    return repo


def _latest_checkpoint(run_dir: Path) -> Path | None:
    """最新的一个检查点。

    优先取 `evaluated`（评估节点，分析脚本按它取模型），否则退回最新的任意检查点。
    两者都按文件名排序 —— 文件名里带 update 序号与纳秒时间戳，字典序即时间序。
    """
    run_1 = run_dir / "runs" / "run_1"
    if not run_1.is_dir():
        return None
    evaluated = sorted(run_1.glob("update_*_evaluated_*.pt"))
    if evaluated:
        return evaluated[-1]
    everything = sorted(run_1.glob("*.pt"))
    return everything[-1] if everything else None


def _collect(run_dir: Path) -> tuple[list[Path], list[Path]]:
    """返回 (检查点文件, 结论层文件)。"""
    checkpoints: list[Path] = []
    latest = _latest_checkpoint(run_dir)
    if latest is not None:
        checkpoints.append(latest)
    evidence: list[Path] = []
    for pattern in EVIDENCE_GLOBS:
        evidence.extend(sorted(run_dir.glob(pattern)))
    return checkpoints, evidence


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def cmd_ls(args) -> int:
    HfApi, _ = _require_hf()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = _repo_id(args.repo)
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as exc:  # 仓库不存在或没有权限
        print(f"无法列出 {repo}: {exc}")
        return 1
    runs: dict[str, int] = {}
    for name in files:
        head = name.split("/", 1)[0]
        runs[head] = runs.get(head, 0) + 1
    if not runs:
        print(f"{repo} 是空的。")
        return 0
    print(f"{repo} 上有 {len(runs)} 个 run：")
    for run, count in sorted(runs.items()):
        print(f"  {run:<32} {count} 个文件")
    return 0


def cmd_push(args) -> int:
    HfApi, _ = _require_hf()
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = _repo_id(args.repo)
    run_dir = (RESEARCH_DIR / args.run).resolve()
    if not run_dir.is_dir():
        sys.exit(f"找不到 run 目录：{run_dir}")

    checkpoints, evidence = _collect(run_dir)
    if not checkpoints and not evidence:
        sys.exit(f"{args.run} 里没有可同步的文件。")

    print(f"上传 {args.run} → {repo}")
    print(f"  检查点 {len(checkpoints)} 个：")
    for path in checkpoints:
        print(f"    {path.relative_to(run_dir)}  {_human(path.stat().st_size)}")
    print(f"  结论层 {len(evidence)} 个（合计 "
          f"{_human(sum(p.stat().st_size for p in evidence))}）")

    if args.dry_run:
        print("\n空跑结束，未上传。")
        return 0

    api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)

    manifest = {
        "run": args.run,
        "checkpoint": str(checkpoints[-1].relative_to(run_dir)) if checkpoints else None,
        "evidence_files": sorted(str(p.relative_to(run_dir)) for p in evidence),
        "note": "只包含最新检查点与结论层；历史中间检查点按 2026-10-01 审计结论不保留。",
    }
    staging = run_dir / ".hf_staging"
    if staging.exists():
        # 上次崩在收尾会留下它。目录是脚本自己的临时产物，直接清掉重来，
        # 否则会卡在一个需要人工介入的状态里。
        print(f"清理上次残留的暂存目录 {staging}")
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        for path in checkpoints:
            target = staging / path.relative_to(run_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
        for path in evidence:
            target = staging / path.relative_to(run_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(path)
        (staging / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        api.upload_folder(
            repo_id=repo,
            repo_type="model",
            folder_path=str(staging),
            path_in_repo=args.run,
            commit_message=f"sync {args.run}: 最新检查点 + 结论层",
        )
    finally:
        # 暂存目录里既有指向源文件的符号链接，也有脚本自己写的 MANIFEST.json。
        # 早期实现只解链接、再逐个 rmdir，会漏掉那个普通文件并抛 ENOTEMPTY
        # （2026-10-01 在台式机上实测，上传本身成功、只有收尾崩了）。
        # shutil.rmtree 对符号链接只删链接本身、不跟进目标，源文件安全。
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\n完成。下载：python3 scripts/hf_sync.py pull {args.run}")
    return 0


def cmd_pull(args) -> int:
    _, snapshot_download = _require_hf()
    repo = _repo_id(args.repo)
    dest = Path(args.dest).expanduser().resolve() if args.dest else RESEARCH_DIR
    dest.mkdir(parents=True, exist_ok=True)
    print(f"下载 {repo}/{args.run} → {dest}")
    snapshot_download(
        repo_id=repo,
        repo_type="model",
        allow_patterns=[f"{args.run}/*"],
        local_dir=str(dest),
    )
    print("完成。")
    return 0


def cmd_pull_all(args) -> int:
    _, snapshot_download = _require_hf()
    repo = _repo_id(args.repo)
    dest = Path(args.dest).expanduser().resolve() if args.dest else RESEARCH_DIR
    dest.mkdir(parents=True, exist_ok=True)
    print(f"下载 {repo} 全部内容 → {dest}")
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(dest))
    print("完成。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="HF 仓库 id，默认读 $PVZ_HF_REPO")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ls", help="列出远端有哪些 run").set_defaults(func=cmd_ls)

    push = sub.add_parser("push", help="上传一个 run 的最新检查点与结论层")
    push.add_argument("run")
    push.add_argument("--dry-run", action="store_true", help="只显示会上传什么")
    push.set_defaults(func=cmd_push)

    pull = sub.add_parser("pull", help="下载一个 run")
    pull.add_argument("run")
    pull.add_argument("--dest", help="下载目录，默认 artifacts/research")
    pull.set_defaults(func=cmd_pull)

    pull_all = sub.add_parser("pull-all", help="下载全部")
    pull_all.add_argument("--dest", help="下载目录，默认 artifacts/research")
    pull_all.set_defaults(func=cmd_pull_all)

    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN") and args.command != "ls":
        print("提示：未设置 HF_TOKEN，将依赖已登录的凭据。", file=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
