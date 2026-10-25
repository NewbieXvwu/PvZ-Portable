#!/usr/bin/env python3
"""探路包自检：用指定引擎重放 4 局人类示范，打印一段可以直接复制回去的报告。

    python3 selfcheck.py                # 自动选引擎（编译出来的优先，否则用预编的）
    python3 selfcheck.py --engine PATH  # 指定引擎

判定标准（主控 17:45 定，10-25 18:0x 校正）：**存档里记录的每一步都复现一致
（存档记 ok=true 的操作必须被接受，记 ok=false 的必须同样被拒绝）、整局仍然通关**。
人类示范里有少量暂停时重复点种被引擎拒绝的操作，存档按原样记了 ok=false，
所以不能要求"全部被接受"。state_hash 只作诊断打印，不作为失败条件 ——
不同构建的粒子/地址派生量会让哈希不同。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMOS = ROOT / "demos"
RESOURCES = ROOT / "res"
PREBUILT = ROOT / "build_prebuilt" / "pvz-portable"
BUILT = ROOT / "build" / "pvz-portable"
REPORT = ROOT / "build_report.json"


class EngineError(RuntimeError):
    pass


class Engine:
    """最小的引擎协议客户端（只依赖标准库）。

    RESET_V2 的参数顺序照抄 python/pvz_env.py 的 PvZEnv.reset —— 两边必须逐字一致，
    否则重放的不是同一局。
    """

    def __init__(self, executable: Path, resource_dir: Path):
        self.savedir = tempfile.TemporaryDirectory(prefix="pvz-probe-")
        env = {**os.environ, "PVZ_NO_POPUP": "1",
               "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1")}
        self.process = subprocess.Popen(
            [str(executable), "-env", "-resdir", str(resource_dir),
             "-savedir", self.savedir.name],
            cwd=str(executable.parent), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
        self.ready = self._read()

    def _read(self) -> dict:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if line.startswith("PVZENV "):
                return json.loads(line[len("PVZENV "):])
        raise EngineError("引擎在握手前退出（returncode=%s）" % self.process.poll())

    def command(self, text: str) -> dict:
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(text + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise EngineError("引擎已退出：%s" % error)
        return self._read()

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=10)
        except Exception:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
        self.savedir.cleanup()


def reset_command(meta: dict) -> str:
    task = meta["environment_task"]
    profile = task["profile"]
    deck = meta["task"]["deck"]
    cards = []
    for card in deck:
        if isinstance(card, int):
            cards.append(str(card))
        else:
            cards.append("%s:%s" % (card[0], card[1]))
    deck_text = ",".join(cards) or "-"
    upgrades = ",".join(str(v) for v in profile["owned_upgrade_plants"]) or "-"
    forced = ",".join(str(v) for v in task["forced_seeds"]) or "-"
    preplanted = ",".join("%s:%s:%s" % tuple(p) for p in task["preplanted"]) or "-"
    command = ("RESET_V2 %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s"
               % (task["level"], task["seed"], task["playthrough"],
                  profile["seed_slot_count"], int(profile["imitater_owned"]),
                  int(profile["first_aid_owned"]), int(profile["pool_cleaner_owned"]),
                  int(profile["roof_cleaner_owned"]), profile["rake_charges"],
                  upgrades, forced, deck_text,
                  ("%g" % task["zombie_count_multiplier"]),
                  task["wave_cap"] or 0, preplanted))
    if task.get("conveyor_start") == "original":
        command += " conveyor_original"
    if task.get("fog_observation"):
        command += " fog_%s" % task["fog_observation"]
    return command


def action_command(action: dict) -> str:
    kind = action.get("type")
    if kind == "plant":
        return "PLANT %s %s %s" % (action["packet"], action["col"], action["row"])
    if kind == "shovel":
        return "SHOVEL %s %s" % (action["col"], action["row"])
    if kind == "wait":
        return "WAIT %s" % action["ticks"]
    if kind == "collect":
        return "COLLECT %s %s" % (action["x"], action["y"])
    raise EngineError("未知动作类型 %r" % kind)


def replay(engine: Engine, archive: Path) -> dict:
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    response = engine.command(reset_command(meta))
    if not response.get("ok") or response.get("observation") is None:
        return dict(name=archive.name, ok=False, stage="reset", detail=str(response)[:300])
    observation = response["observation"]
    # state_hash 只对**同一个二进制**可比（官方 verify 在二进制 SHA 不同时直接跳过哈希比较），
    # 所以这里只在开局取一次作诊断，不参与判定，也不逐操作取（那会让自检慢一倍）。
    initial_hash = engine.command("STATE_HASH").get("state_hash")
    expected_initial = meta.get("initial_state_hash")
    initial_hash_ok = None if not expected_initial else initial_hash == expected_initial
    operations = 0
    first_bad = None
    rejected = 0          # 存档里本来就记录为 ok=false 的操作（暂停时重复点种，被引擎拒绝）
    rejected_matched = 0
    hashes_differ = 0
    hashed = 0
    first_hash = None
    # frames.jsonl 是"人类做决定时"的数值观测，每个 frame 的 operation 是它**之前**
    # 那一步在 ticks.jsonl 里的 0 基下标。所以第 i 个操作（1 基）结束后的局面应当等于
    # operation == i 的那个 frame。只在有 frame 的操作上核对，比只看 tick 更能证明模拟一致。
    frames = {}
    if (archive / "frames.jsonl").is_file():
        with (archive / "frames.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                frame = json.loads(line)
                frames[int(frame.get("operation", -1))] = frame
    frame_checks = 0
    frame_mismatch = None
    started = time.time()
    with (archive / "ticks.jsonl").open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            entry = json.loads(line)
            operations = index
            try:
                response = engine.command(action_command(entry["action"]))
            except EngineError as error:
                return dict(name=archive.name, ok=False, stage="engine-exit",
                            operations=operations, first_bad=index, detail=str(error)[:300])
            observation = response.get("observation") or {}
            expected_ok = bool(entry.get("ok", True))
            if not expected_ok:
                rejected += 1
                rejected_matched += 0 if response.get("ok") else 1
            frame = frames.get(index)
            if frame is not None:
                frame_checks += 1
                got = (observation.get("sun"), observation.get("wave"),
                       len(observation.get("plants") or []), len(observation.get("zombies") or []),
                       observation.get("result"))
                want = (frame.get("sun"), frame.get("wave"), len(frame.get("plants") or []),
                        len(frame.get("zombies") or []), frame.get("result"))
                if got != want and frame_mismatch is None:
                    frame_mismatch = dict(operation=index, tick=observation.get("tick"),
                                          got=got, expected=want, action=entry["action"])
            if first_bad is None and (bool(response.get("ok")) != expected_ok
                                      or int(observation.get("tick", -1)) != int(entry["end_tick"])):
                first_bad = dict(operation=index, tick=observation.get("tick"),
                                 expected=entry["end_tick"], ok=response.get("ok"),
                                 expected_ok=expected_ok, action=entry["action"])
    elapsed = time.time() - started
    outcome = meta.get("outcome") or {}
    result = int(observation.get("result") or 0)
    won = result == 1
    expected_won = bool(outcome.get("won"))
    tick_ok = int(observation.get("tick", -1)) == int(outcome.get("final_tick", -2))
    return dict(name=archive.name, ok=first_bad is None and won == expected_won and tick_ok,
                operations=operations, first_bad=first_bad, result=result,
                expected_result=outcome.get("result"), final_tick=observation.get("tick"),
                expected_tick=outcome.get("final_tick"), seconds=round(elapsed, 1),
                rejected=rejected, rejected_matched=rejected_matched,
                initial_hash=initial_hash, initial_hash_ok=initial_hash_ok,
                frame_checks=frame_checks, frame_mismatch=frame_mismatch)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def glibc_version() -> str:
    try:
        output = subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout
        return output.splitlines()[0] if output else "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--resources", type=Path, default=RESOURCES)
    args = parser.parse_args()

    if args.engine:
        engines = [args.engine]
    else:
        engines = []
        if BUILT.is_file():
            engines.append(BUILT)
        if PREBUILT.is_file():
            engines.append(PREBUILT)
    if not engines:
        print("SELFCHECK FAILED: 找不到引擎（build/ 与 build_prebuilt/ 都没有）")
        return 1
    if not (args.resources / "main.pak").is_file():
        print("SELFCHECK FAILED: 缺少 res/main.pak")
        return 1
    archives = sorted(path for path in DEMOS.iterdir() if (path / "meta.json").is_file())
    if not archives:
        print("SELFCHECK FAILED: demos/ 里没有存档")
        return 1

    build = {}
    if REPORT.is_file():
        for line in REPORT.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                build[key.strip()] = value.strip()

    lines = ["=== llmplay probe report (copy everything below) ==="]
    lines.append("time:      %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("os:        %s" % platform.platform())
    lines.append("glibc:     %s" % glibc_version())
    lines.append("cpu:       %s" % cpu_model())
    lines.append("python:    %s" % sys.version.split()[0])
    if build:
        lines.append("compiler:  %s" % build.get("cxx", "?"))
        lines.append("cmake:     %s" % build.get("cmake", "?"))
        lines.append("build:     %s %s" % (build.get("build", "?"),
                                           build.get("build_stage", "")).strip())
    else:
        lines.append("build:     not attempted (没有 build_report.json)")

    failures = 0
    for engine in engines:
        lines.append("engine:    %s sha256=%s" % (engine.relative_to(ROOT), sha256(engine)[:16]))
        try:
            client = Engine(engine, args.resources)
        except Exception as error:
            lines.append("  HANDSHAKE FAILED: %s" % error)
            failures += 1
            continue
        try:
            for archive in archives:
                outcome = replay(client, archive)
                if outcome["ok"]:
                    lines.append("  %-32s PASS ops=%d result=%s tick=%s (%ss)"
                                 % (archive.name, outcome["operations"], outcome["result"],
                                    outcome["final_tick"], outcome["seconds"]))
                else:
                    failures += 1
                    lines.append("  %-32s FAIL %s" % (archive.name,
                                                      json.dumps({k: v for k, v in outcome.items()
                                                                  if k != "name"},
                                                                 ensure_ascii=False)[:400]))
                if "rejected" in outcome:
                    lines.append("  %-32s      存档内被拒绝的操作 %d/%d 复现一致"
                                 % ("", outcome["rejected_matched"], outcome["rejected"]))
                    if outcome.get("frame_checks"):
                        lines.append("  %-32s      逐 tick 观测核对 %d 次：%s"
                                     % ("", outcome["frame_checks"],
                                        "全部一致" if not outcome.get("frame_mismatch")
                                        else "首个不一致 " + json.dumps(outcome["frame_mismatch"],
                                                                       ensure_ascii=False)[:300]))
                    lines.append("  %-32s      开局 state_hash=%s（存档记录 %s，仅诊断：哈希随构建变化）"
                                 % ("", (outcome.get("initial_hash") or "?")[:16],
                                    "一致" if outcome.get("initial_hash_ok") else "不同"))
        finally:
            client.close()
    lines.append("verdict:   %s" % ("ALL PASS" if failures == 0 else "%d FAILURE(S)" % failures))
    lines.append("=== end ===")
    print("\n".join(lines))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
