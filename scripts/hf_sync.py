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
    python3 scripts/hf_sync.py push reward_r0_seed0_v2  # 上传检查点集 + 结论层
    python3 scripts/hf_sync.py pull reward_r0_seed0_v2  # 下载到本地
    python3 scripts/hf_sync.py pull-all                 # 下载全部

    # 想看会上传什么，先空跑：
    python3 scripts/hf_sync.py push reward_r0_seed0_v2 --dry-run

上传范围（每个 run 约 286 MB）：
  * 检查点 —— `evaluated` / `initial` / `boundary` / `resumed` 全部，
    加最新一个 `trained`，**再加 `resume.json` 指向的那个文件**。
    实测 `reward_r*_v2` 的 `resume.json` 指向 `boundary`，所以"只传最新
    `evaluated`"是接不上的。历史 `trained` 中间快照不带。
  * 结论层 —— `learning_curve.json` / `training_state.json` /
    `experiment_config.json` / `provenance.json` / `resume.json` /
    `evaluations/*.json.gz`。
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
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

# 检查点命名：`update_<序号>_<阶段>_<纳秒时间戳>.pt`。字典序即时间序。
CHECKPOINT_RE = re.compile(r"^update_\d+_(?P<phase>[a-z]+)_\d+\.pt$")

# 已知阶段：`initial` 起点 / `evaluated` 分析脚本取模型的节点 / `boundary` 冻结边界 /
# `resumed` 续跑锚点 —— 这几个**全部**带走。
# `trained` 每个 update 都存一份，只带最新的一个。其余是纯历史，没有任何引用链指向
# （2026-10-01 审计：单个 run 曾堆 70+ 个 trained，3.1 GB）。
#
# 注意这里是**白名单式的黑名单**：只有确认冗余的 `trained` 会被丢掉，
# 阶段名认不出来的一律保留。宁可多传，不要悄悄丢证据。
CHECKPOINT_KNOWN_PHASES = ("initial", "evaluated", "boundary", "resumed")
CHECKPOINT_TRAINED_KEEP = 1
TRAINED_PROBE_WINDOW = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_resume(run_dir: Path) -> None:
    pointer = run_dir / "resume.json"
    if not pointer.exists():
        return
    payload = json.loads(pointer.read_text())
    target = _resume_target(run_dir)
    resolved = _resolve_checkpoint(run_dir, target) if target else None
    if resolved is None or not resolved.is_relative_to(run_dir.resolve()):
        raise ValueError(f"{pointer}: resume target missing or outside this snapshot")
    if payload.get("sha256") != _sha256(resolved):
        raise ValueError(f"{pointer}: resume checkpoint SHA256 mismatch or missing")


@contextlib.contextmanager
def _source_locks(source: Path):
    """Read-lock existing run locks; an active writer is never interrupted."""
    with contextlib.ExitStack() as stack:
        for path in sorted(source.rglob(".execution.lock")):
            if ".hf_staging" in path.parts:
                continue
            stream = stack.enter_context(path.open("rb"))
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"active run cannot be snapshotted: {path.parent}") from None
        yield


def _validate_archive(source: Path) -> None:
    manifest = source / "archive_manifest.json"
    if not manifest.exists():
        return
    files = json.loads(manifest.read_text())["files"]
    if not isinstance(files, dict):
        raise ValueError("unsupported archive inventory schema")
    for name, expected in files.items():
        path = source / name
        if (not path.resolve().is_relative_to(source.resolve()) or not path.is_file()
                or path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]):
            raise ValueError(f"archive reference chain broken: {name}")


def _upload_snapshot(api, repo: str, source: Path, key: str,
                     files: dict[str, Path], manifest: dict, *, revision: str | None = None) -> None:
    """Copy mutable metadata/logs; link immutable checkpoints and rollout shards."""
    if "MANIFEST.json" in files:
        raise ValueError("source MANIFEST.json would collide with the HF delivery manifest")
    with tempfile.TemporaryDirectory(prefix="pvz-hf-snapshot-") as directory:
        staging = Path(directory)
        records = {}
        for name, path in sorted(files.items()):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("invalid snapshot relative path")
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in (".pt", ".npz"):
                target.symlink_to(path.resolve())
            else:
                shutil.copy2(path, target)
            records[name] = {"bytes": target.stat().st_size, "sha256": _sha256(target)}
        manifest.update(schema_version=2, files=records)
        if revision is not None:
            manifest['hf_revision'] = revision
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        api.create_repo(repo_id=repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(staging),
                          path_in_repo=key, commit_message=f"sync {key}: hashed evidence snapshot",
                          **({'revision':revision} if revision is not None else {}))
        # A lock-free binary edited during upload must not be reported as verified.
        for name, path in files.items():
            if path.suffix in (".pt", ".npz") and _sha256(path) != records[name]["sha256"]:
                raise ValueError(f"source binary changed during upload: {name}")


def cmd_push_evidence(args) -> int:
    source = args.source.resolve()
    if not source.is_dir() or not any((source / name).is_file() for name in
                                      ("archive_manifest.json", "report.json", "training_state.json")):
        raise ValueError("push-evidence requires a completed research/engineering evidence tree")
    repo = args.repo or os.environ.get("PVZ_HF_REPO")
    if not repo and not args.dry_run:
        repo = _repo_id(None)
    with _source_locks(source):
        _validate_archive(source)
        for pointer in source.rglob("resume.json"):
            _validate_resume(pointer.parent)
        files = {str(p.relative_to(source)): p for p in source.rglob("*")
                 if p.is_file() and ".hf_staging" not in p.relative_to(source).parts}
        if any(not p.resolve().is_relative_to(source) for p in files.values()):
            raise ValueError("evidence tree contains a file link outside the selected source")
        total = sum(p.stat().st_size for p in files.values())
        print(f"证据 {source} → {repo or '(未配置)'} / {args.name}: {len(files)} files, {_human(total)}")
        if args.dry_run:
            print("引用链核验通过；空跑结束，未上传。")
            return 0
        HfApi, _ = _require_hf()
        _upload_snapshot(HfApi(token=os.environ.get("HF_TOKEN")), repo, source, args.name, files,
                         {"source_directory": str(source), "kind": "complete_evidence_tree"},
                         revision=getattr(args, 'revision', None))
    return 0


def cmd_verify(args) -> int:
    source = args.source.resolve()
    manifest = json.loads((source / "MANIFEST.json").read_text())
    if manifest.get("schema_version") != 2:
        raise ValueError("verification requires the hashed schema-2 HF manifest")
    for name, expected in manifest["files"].items():
        path = source / name
        if (not path.resolve().is_relative_to(source) or not path.is_file()
                or path.stat().st_size != expected["bytes"] or _sha256(path) != expected["sha256"]):
            raise ValueError(f"HF snapshot integrity failure: {name}")
    for pointer in source.rglob("resume.json"):
        _validate_resume(pointer.parent)
    _validate_archive(source)
    print(f"SHA256、体积及续跑指针核验通过: {len(manifest['files'])} files")
    return 0


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


def _phase(path: Path) -> str | None:
    match = CHECKPOINT_RE.match(path.name)
    return match.group("phase") if match else None


def _resume_target(run_dir: Path) -> str | None:
    """`resume.json` 的 `checkpoint` 字段 —— 定义"要接着训练必须加载哪个文件"。

    这是唯一权威的续跑判据（见 REPO_BLOAT_AUDIT_20261001.md）。2026-10-01 实测
    `reward_r*_v2` 四个 run 的它都指向 `boundary`，而不是最新的 `evaluated` ——
    早先只传最新 `evaluated` 的版本，下载回来是**接不上的**。
    """
    try:
        payload = json.loads((run_dir / "resume.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    target = payload.get("checkpoint")
    return target if isinstance(target, str) else None


def _resolve_checkpoint(run_dir: Path, target: str) -> Path | None:
    """把 `resume.json` 里的 `checkpoint` 值解析成真实路径。

    实测值是相对 run 目录的（`runs/run_1/update_...pt`）。也容忍只写文件名、
    或写成相对仓库根的路径 —— 解析不出来时返回 None，由调用方给警告，
    **不要假装成功**。
    """
    for candidate in (run_dir / target,
                      run_dir / "runs" / "run_1" / target,
                      ROOT / target):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _collect(run_dir: Path) -> tuple[list[Path], list[Path], list[str]]:
    """返回 (检查点, 结论层, 警告)。

    检查点范围 = 除历史 `trained` 快照外的全部（见 CHECKPOINT_KNOWN_PHASES 的说明），
    **再强制并入 `resume.json` 指向的那个文件**。最后一步是硬要求：漏掉它，
    这个通道就白建了 —— 实测 `reward_r*_v2` 的 `resume.json` 都指向 `boundary`，
    而早先"只传最新 evaluated"的版本下载回来是接不上的。
    """
    warnings: list[str] = []
    checkpoints: list[Path] = []
    run_1 = run_dir / "runs" / "run_1"
    if run_1.is_dir():
        by_phase: dict[str, list[Path]] = {}
        for path in sorted(run_1.glob("*.pt")):
            by_phase.setdefault(_phase(path) or "unrecognised", []).append(path)
        for phase, paths in sorted(by_phase.items()):
            if phase == "trained":
                checkpoints.extend(paths[-CHECKPOINT_TRAINED_KEEP:])
            else:
                checkpoints.extend(paths)

    target = _resume_target(run_dir)
    if target:
        resolved = _resolve_checkpoint(run_dir, target)
        if resolved is None:
            warnings.append(f"⚠️ resume.json 指向 {target}，但该文件不存在 —— 这份归档接不上")
        elif resolved not in {path.resolve() for path in checkpoints}:
            checkpoints.append(resolved)
            warnings.append(f"resume.json 指向 {target}，已并入上传集")

    evidence: list[Path] = []
    for pattern in EVIDENCE_GLOBS:
        evidence.extend(sorted(run_dir.glob(pattern)))
    return checkpoints, evidence, warnings


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
        files = api.list_repo_files(repo_id=repo, repo_type="model",
                                   **({'revision':args.revision} if getattr(args, 'revision', None) else {}))
    except Exception as exc:  # 仓库不存在或没有权限
        print(f"无法列出 {repo}: {exc}")
        return 1
    runs: dict[str, int] = {}
    root_files: list[str] = []
    for name in files:
        if "/" in name:
            head = name.split("/", 1)[0]
            runs[head] = runs.get(head, 0) + 1
        else:
            # 仓库根下的文件（.gitattributes 等）不是 run，别混进 run 列表里。
            root_files.append(name)
    if not runs and not root_files:
        print(f"{repo} 是空的。")
        return 0
    if runs:
        print(f"{repo} 上有 {len(runs)} 个 run：")
        for run, count in sorted(runs.items()):
            print(f"  {run:<32} {count} 个文件")
    if root_files:
        print(f"  仓库根文件：{', '.join(sorted(root_files))}")
    return 0


def cmd_push(args) -> int:
    run_dir = (RESEARCH_DIR / args.run).resolve()
    with _source_locks(run_dir):
        return _cmd_push_locked(args, run_dir)


def _cmd_push_locked(args, run_dir: Path) -> int:
    api = None
    if not args.dry_run:
        HfApi, _ = _require_hf()
        api = HfApi(token=os.environ.get("HF_TOKEN"))
    repo = args.repo or os.environ.get("PVZ_HF_REPO")
    if not repo and not args.dry_run:
        repo = _repo_id(None)
    if not run_dir.is_dir():
        sys.exit(f"找不到 run 目录：{run_dir}")

    checkpoints, evidence, warnings = _collect(run_dir)
    _validate_resume(run_dir)
    if args.include_trained_window:
        checkpoints = sorted(set(checkpoints) | set(sorted(
            (run_dir / "runs/run_1").glob("update_*_trained_*.pt"))[-TRAINED_PROBE_WINDOW:]))
    rollouts = sorted(run_dir.glob("runs/run_1/.seed_jobs/**/*.npz")) if args.include_rollouts else []
    for path in args.log:
        if not path.is_file():
            raise FileNotFoundError(f"selected log missing: {path}")
    if not checkpoints and not evidence:
        sys.exit(f"{args.run} 里没有可同步的文件。")

    total = sum(p.stat().st_size for p in checkpoints)
    print(f"上传 {args.run} → {repo or '(未配置)'}")
    print(f"  检查点 {len(checkpoints)} 个（合计 {_human(total)}）：")
    for path in checkpoints:
        print(f"    {path.relative_to(run_dir)}  {_human(path.stat().st_size)}")
    print(f"  结论层 {len(evidence)} 个（合计 "
          f"{_human(sum(p.stat().st_size for p in evidence))}）")
    for warning in warnings:
        print(f"  {warning}")
    print(f"  原始分片 {len(rollouts)} 个，外部日志 {len(args.log)} 个")

    if args.dry_run:
        print("\n空跑结束，未上传。")
        return 0

    resume_target = _resume_target(run_dir)
    manifest = {
        "run": args.run,
        "resume_checkpoint": resume_target,
        "checkpoints": sorted(str(p.relative_to(run_dir)) for p in checkpoints),
        "evidence_files": sorted(str(p.relative_to(run_dir)) for p in evidence),
        "rollout_files": sorted(str(p.relative_to(run_dir)) for p in rollouts),
        "trained_window_included": args.include_trained_window,
        "note": ("含全部 evaluated/initial/boundary/resumed 与最新一个 trained，"
                 "外加 resume.json 指向的检查点；历史 trained 中间快照按 2026-10-01 审计结论不保留。"),
    }
    stale_staging = run_dir / ".hf_staging"
    if stale_staging.is_symlink():
        raise ValueError("old HF staging directory is a symlink; source not modified")
    if stale_staging.exists():
        # 上次崩在收尾会留下它。目录是脚本自己的临时产物，直接清掉重来，
        # 否则会卡在一个需要人工介入的状态里。
        print(f"清理上次残留的暂存目录 {stale_staging}")
        shutil.rmtree(stale_staging)
    files = {str(p.relative_to(run_dir)): p for p in checkpoints + evidence + rollouts}
    for path in args.log:
        name = f"logs/{path.name}"
        if name in files:
            raise ValueError("duplicate external log name")
        files[name] = path.resolve()
    _upload_snapshot(api, repo, run_dir, args.run, files, manifest,
                     revision=getattr(args, 'revision', None))

    revision_arg = f" --revision {args.revision}" if getattr(args, 'revision', None) else ''
    print(f"\n完成。下载：python3 scripts/hf_sync.py{revision_arg} pull {args.run}")
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
        **({'revision':args.revision} if getattr(args, 'revision', None) else {}),
    )
    print("完成。")
    return 0


def cmd_pull_all(args) -> int:
    _, snapshot_download = _require_hf()
    repo = _repo_id(args.repo)
    dest = Path(args.dest).expanduser().resolve() if args.dest else RESEARCH_DIR
    dest.mkdir(parents=True, exist_ok=True)
    print(f"下载 {repo} 全部内容 → {dest}")
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(dest),
                      **({'revision':args.revision} if getattr(args, 'revision', None) else {}))
    print("完成。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", help="HF 仓库 id，默认读 $PVZ_HF_REPO")
    parser.add_argument("--revision", help="已有HF分支/读取版本；省略保持main。证据分支必须另行从空基础提交创建，工具不删除远端文件")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ls", help="列出远端有哪些 run").set_defaults(func=cmd_ls)

    push = sub.add_parser("push", help="上传一个 run 的最新检查点与结论层")
    push.add_argument("run")
    push.add_argument("--dry-run", action="store_true", help="只显示会上传什么")
    push.add_argument("--include-rollouts", action="store_true", help="额外上传原始NPZ分片及失败分片")
    push.add_argument("--include-trained-window", action="store_true", help="额外上传最后8个trained，供末段诊断")
    push.add_argument("--log", type=Path, action="append", default=[], help="额外上传指定日志，可重复")
    push.set_defaults(func=cmd_push)

    evidence = sub.add_parser("push-evidence", help="上传整棵已有归档或中断试验现场，核验引用链")
    evidence.add_argument("source", type=Path)
    evidence.add_argument("--name", required=True, help="HF仓库内的证据目录名")
    evidence.add_argument("--dry-run", action="store_true")
    evidence.set_defaults(func=cmd_push_evidence)

    verify = sub.add_parser("verify", help="离线核验下载的schema-2 HF快照及续跑指针")
    verify.add_argument("source", type=Path)
    verify.set_defaults(func=cmd_verify)

    pull = sub.add_parser("pull", help="下载一个 run")
    pull.add_argument("run")
    pull.add_argument("--dest", help="下载目录，默认 artifacts/research")
    pull.set_defaults(func=cmd_pull)

    pull_all = sub.add_parser("pull-all", help="下载全部")
    pull_all.add_argument("--dest", help="下载目录，默认 artifacts/research")
    pull_all.set_defaults(func=cmd_pull_all)

    args = parser.parse_args()
    if (not os.environ.get("HF_TOKEN") and args.command in ("push", "push-evidence", "pull", "pull-all")
            and not getattr(args, "dry_run", False)):
        print("提示：未设置 HF_TOKEN，将依赖已登录的凭据。", file=sys.stderr)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
