"""用给定引擎重放 episode_archive_v1 存档，逐操作比对状态。

    # 单个存档：和存档里记的 tick / ok / state_hash 比
    python3 scripts/llmplay/verify_archive.py --archive DIR --executable BIN

    # 两个引擎对同一份存档：比 tick / ok / 精简观测摘要（state_hash 只做记录）
    python3 scripts/llmplay/verify_archive.py --archive DIR --executable BIN \
        --baseline-exe BASE --against baseline

为什么默认不比 state_hash：引擎的状态哈希把粒子抖动这类**进程内地址派生**的量也
算进去（task63 的 FIELD_SHAKE），同一份源码换一种链接方式就会变。task70 的逐位门禁
因此只比动作序列与可观测状态。这里的"可观测状态"取精简观测的完整 JSON 摘要 ——
棋盘、植物血量、僵尸坐标/血量/冰冻、卡槽冷却、阳光、波次、割草机、占位物全在里面。

退出码 0 = 全部一致。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_event_env import EventWaitEnv  # noqa: E402
from human_demo import reset_archive  # noqa: E402

DEFAULT_RESOURCES = Path("/home/newbiexvwu/.cache/pvz-research-resources")


def _digest(observation: dict) -> str:
    blob = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def replay_states(archive: Path, executable: Path, resource_dir: Path) -> list:
    """按存档的操作序列重放，返回每一步的 (tick, end_tick, ok, state_hash, 观测摘要)。"""
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    out = []
    with EventWaitEnv(resource_dir, executable=executable) as env:
        obs = reset_archive(env, meta)
        env.set_lean_observation(True)
        obs = env.observe()
        with (archive / "ticks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                before = int(obs["tick"])
                obs, _, _, _, info = env.step(entry["action"])
                out.append((before, int(obs["tick"]), bool(info["ok"]),
                            env._last_response.get("state_hash"), _digest(obs)))
    return out


def compare(name: str, left: list, right: list, *, check_hash: bool) -> tuple:
    """返回 (失败列表, 说明)。tick/ok/观测摘要必须一致；hash 只在 check_hash 时算失败。"""
    problems, notes = [], []
    if len(left) != len(right):
        problems.append("%s: 操作数不同 %d vs %d" % (name, len(left), len(right)))
    hash_diffs = 0
    for index, (a, b) in enumerate(zip(left, right), 1):
        if a[:3] != b[:3]:
            problems.append("%s: 第 %d 步 tick/ok 不同 %s vs %s" % (name, index, a[:3], b[:3]))
            break
        if a[4] != b[4]:
            problems.append("%s: 第 %d 步观测不同（摘要 %s vs %s）"
                            % (name, index, a[4][:16], b[4][:16]))
            break
        if a[3] != b[3]:
            hash_diffs += 1
            if check_hash:
                problems.append("%s: 第 %d 步 state_hash 不同 %s vs %s"
                                % (name, index, str(a[3])[:16], str(b[3])[:16]))
                break
    if hash_diffs and not check_hash:
        notes.append("%s: state_hash 有 %d 步不同（粒子/地址派生量，按 task70 口径不算失败）"
                     % (name, hash_diffs))
    return problems, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline-exe", type=Path)
    parser.add_argument("--against", choices=("recorded", "baseline"), default="recorded",
                        help="recorded=和存档里记的比；baseline=和 --baseline-exe 的重放结果比")
    parser.add_argument("--strict-hash", action="store_true",
                        help="连 state_hash 也要一致（默认只在 against=recorded 时检查）")
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCES)
    args = parser.parse_args()
    if args.against == "baseline" and not args.baseline_exe:
        parser.error("--against baseline 需要 --baseline-exe")
    failures = 0
    report = []
    for archive in args.archive:
        recorded = []
        with (archive / "ticks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                recorded.append((int(entry["tick"]), int(entry["end_tick"]),
                                 bool(entry["ok"]), entry.get("state_hash")))
        got = replay_states(archive, args.executable, args.resource_dir)
        if args.against == "recorded":
            problems, notes = [], []
            if len(recorded) != len(got):
                problems.append("%s: 操作数不同 %d vs %d"
                                % (archive.name, len(recorded), len(got)))
            for index, (a, b) in enumerate(zip(recorded, got), 1):
                if a[:3] != b[:3]:
                    problems.append("%s: 第 %d 步 tick/ok 不同 %s vs %s"
                                    % (archive.name, index, a[:3], b[:3]))
                    break
                if a[3] != b[3]:
                    problems.append("%s: 第 %d 步 state_hash 不同 %s vs %s"
                                    % (archive.name, index, str(a[3])[:16], str(b[3])[:16]))
                    break
        else:
            base = replay_states(archive, args.baseline_exe, args.resource_dir)
            problems, notes = compare("%s vs baseline" % archive.name, base, got,
                                      check_hash=args.strict_hash)
        report.append(dict(archive=str(archive), operations=len(got),
                           problems=problems, notes=notes))
        failures += bool(problems)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
