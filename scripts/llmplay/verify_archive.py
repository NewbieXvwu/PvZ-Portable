"""用给定引擎重放 episode_archive_v1 存档，逐操作比对 tick / ok / state_hash。

    # 单个存档：用 BIN 重放，和存档里记的哈希逐条比对
    python3 scripts/llmplay/verify_archive.py --archive DIR --executable BIN

    # 两个引擎对同一份存档：先用 BASELINE 重放，再和 BIN 的结果逐条比对
    python3 scripts/llmplay/verify_archive.py --archive DIR --executable BIN --baseline-exe BASE

退出码 0 = 全部一致。不一致时打印第一处差异的序号与两条哈希。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from pvz_event_env import EventWaitEnv  # noqa: E402
from human_demo import reset_archive  # noqa: E402

DEFAULT_RESOURCES = Path("/home/newbiexvwu/.cache/pvz-research-resources")


def replay_hashes(archive: Path, executable: Path, resource_dir: Path) -> list:
    """按存档的操作序列重放，返回每一步 (tick, end_tick, ok, state_hash)。"""
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    out = []
    with EventWaitEnv(resource_dir, executable=executable) as env:
        obs = reset_archive(env, meta)
        with (archive / "ticks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                before = int(obs["tick"])
                obs, _, _, _, info = env.step(entry["action"])
                out.append((before, int(obs["tick"]), bool(info["ok"]),
                            env._last_response.get("state_hash")))
    return out


def compare(name: str, left: list, right: list, *, check_hash: bool = True) -> list:
    problems = []
    if len(left) != len(right):
        problems.append("%s: 操作数不同 %d vs %d" % (name, len(left), len(right)))
    for index, (a, b) in enumerate(zip(left, right), 1):
        if a[:3] != b[:3]:
            problems.append("%s: 第 %d 步 tick/ok 不同 %s vs %s" % (name, index, a[:3], b[:3]))
            break
        if check_hash and a[3] != b[3]:
            problems.append("%s: 第 %d 步 state_hash 不同 %s vs %s"
                            % (name, index, str(a[3])[:16], str(b[3])[:16]))
            break
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, action="append", required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--baseline-exe", type=Path)
    parser.add_argument("--resource-dir", type=Path, default=DEFAULT_RESOURCES)
    args = parser.parse_args()
    failures = 0
    report = []
    for archive in args.archive:
        recorded = []
        with (archive / "ticks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                entry = json.loads(line)
                recorded.append((int(entry["tick"]), int(entry["end_tick"]),
                                 bool(entry["ok"]), entry.get("state_hash")))
        got = replay_hashes(archive, args.executable, args.resource_dir)
        problems = compare("%s vs 存档" % archive.name, recorded, got)
        if args.baseline_exe:
            base = replay_hashes(archive, args.baseline_exe, args.resource_dir)
            problems += compare("%s vs baseline" % archive.name, base, got)
        report.append(dict(archive=str(archive), operations=len(got), problems=problems))
        failures += bool(problems)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
