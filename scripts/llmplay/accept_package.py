#!/usr/bin/env python3
"""交付前验收一个已打好的 llmplay 包（主机侧，x86-64 Linux，每个包跑一次，几分钟）。

不进包、不在 GPT 沙箱里跑（AGENTS §6.5）。它只测沙箱里会遇到、单元测试覆盖不到的东西：
包内引擎能不能在 Linux 上直接跑、三类题从实际解出的包里能不能开局 / 推进 / 导出、
进程被回收后 save→load 续打、多进程同时领题不重复、连续两次导出进度不回退、
导出的 drill 存档能被导入器复验。

    python3 scripts/llmplay/accept_package.py --package <解开的 llmplay_pkg> \
        --resource-dir ~/.cache/pvz-research-resources [--cache64 <已有 cache64 目录>] --work <临时目录>

11-05 起每个包交付前必跑：238 包把 Mac arm64 引擎、回收进程判定错误、并发进度回退、
起点无目标几个问题一起交给了用户。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def worker(mode: str, payload: dict) -> dict:
    """在解开的包里执行（cwd = 包目录，LLMPLAY_OUT 由父进程设置）。"""
    sys.path.insert(0, os.getcwd())
    import llmplay
    from llmplay import Game, InvalidDrillError
    if mode == "claim":
        item = llmplay.next_task()["item"]
        return dict(layer=item.get("layer"), key=[item.get("task_id"), item.get("seed"),
                                                   item.get("future_seed")])
    if mode == "drill_start":
        while True:
            item = llmplay.next_task(track=None)["item"]
            if item.get("layer") != "drill":
                raise RuntimeError("队首不是 drill：%s" % item.get("layer"))
            try:
                game = Game.load_archive(item["path"], strategy=item["knowledge"])
                game.reseed(item["future_seed"])
                game.observe()
                break
            except InvalidDrillError:
                continue
        game.wait_until(game.tick + 300)
        game.save(payload["save"])
        game.close()
        return dict(item=dict(path=item["path"], drill_id=item.get("drill_id"),
                              horizon=item["horizon"]))
    if mode == "drill_finish":          # 新进程：模拟沙箱回收了上一个进程
        game = Game.load(payload["save"])
        game.wait_until(lambda g: g.drill_result()["reached"], max_ticks=int(payload["horizon"]) + 600)
        result = game.drill_result()
        path = game.export_archive(llmplay.out_dir() / ("drill_" + str(payload["drill_id"])))
        game.close()
        return dict(result={k: result[k] for k in ("success", "reached", "tick_delta", "mower_used")},
                    archive=str(path))
    if mode == "scenario":
        catalog = json.loads(Path("configs/llmplay_scenarios.json").read_text())["items"][0]
        game = Game.load_archive(catalog["archive"])
        game.reseed(catalog["future_seeds"][0])
        row = 2
        game.spawn(16, row=row, x=760)
        game.declare_scenario("气球：验收", horizon=600, targets=[game.last_spawn_id])
        game.wait_until(lambda g: g.tick - g.scenario_meta["root_tick"] >= 600 or g.obs.get("terminal"),
                        max_ticks=1200)
        try:
            path = str(game.export_archive(llmplay.out_dir() / "scenario_accept"))
        except ValueError as error:     # “什么都不做”也成功会被拒，这也是接口按设计工作
            path = "rejected: %s" % error
        game.close()
        return dict(export=path)
    if mode == "resume":
        catalog = json.loads(Path("configs/llmplay_resume.json").read_text())
        items = catalog["items"] if isinstance(catalog, dict) else catalog
        entry = items[0]
        game = Game.load_archive(entry.get("archive") or entry["path"])
        game.wait_until(game.tick + 300)
        tick = game.tick
        game.close()
        return dict(tick=tick)
    if mode == "progress":
        document = llmplay.progress()
        return dict(archives=len(document.get("archives") or []))
    if mode == "export":
        return dict(bundles=[str(path) for path in llmplay.export_bundle()])
    raise ValueError(mode)


def run(package: Path, env: dict, mode: str, payload: dict | None = None) -> dict:
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", mode,
               "--payload", json.dumps(payload or {})]
    completed = subprocess.run(command, cwd=package, env=env, capture_output=True, text=True,
                               timeout=900)
    if completed.returncode:
        raise RuntimeError("%s 失败：\n%s" % (mode, completed.stderr[-3000:]))
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker")
    parser.add_argument("--payload", default="{}")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--resource-dir", type=Path)
    parser.add_argument("--cache64", type=Path)
    parser.add_argument("--work", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, json.loads(args.payload)), ensure_ascii=False))
        return 0
    import types
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "python")]
    stub = types.ModuleType("llmplay")      # 同 build_package：只导模块，不触发包的开局初始化
    stub.__path__ = [str(ROOT / "scripts" / "llmplay")]
    sys.modules["llmplay"] = stub
    from llmplay.build_package import check_engine
    package, work = args.package.resolve(), args.work.resolve()
    report = {}
    check_engine(package / "build" / "pvz-portable")
    report["engine"] = "static x86-64 ELF"
    res = package / "res"
    if not (res / "main.pak").is_file():     # 资源不能走 HF：用本机已有资源拼出 res/
        res.mkdir(exist_ok=True)
        for name in ("main.pak", "properties"):
            if not (res / name).exists():
                (res / name).symlink_to(args.resource_dir / name)
        if args.cache64 and not (res / "cache64").exists():
            shutil.copytree(args.cache64, res / "cache64")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    env = dict(os.environ, LLMPLAY_OUT=str(work / "out"), TMPDIR=str(work), PVZ_NO_POPUP="1",
               PYTHONPATH="")

    claims = [subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", "claim"],
                               cwd=package, env=env, stdout=subprocess.PIPE, text=True)
              for _ in range(3)]
    keys = [json.loads(proc.communicate(timeout=600)[0].strip().splitlines()[-1])["key"] for proc in claims]
    if len({json.dumps(key) for key in keys}) != len(keys):
        raise SystemExit("并发领题重复：%s" % keys)
    report["concurrent_claims"] = keys

    started = run(package, env, "drill_start", dict(save=str(work / "drill.save")))
    finished = run(package, env, "drill_finish", dict(save=str(work / "drill.save"),
                                                      **started["item"]))
    report["drill"] = finished["result"]

    from import_llm_run import verify_short
    copy = work / "verify" / Path(finished["archive"]).name
    shutil.copytree(finished["archive"], copy)
    verified = verify_short(copy, package / "res", package / "build" / "pvz-portable")
    report["importer_verify"] = verified["result"]

    report["scenario"] = run(package, env, "scenario")["export"]
    report["resume_tick"] = run(package, env, "resume")["tick"]

    before = run(package, env, "progress")["archives"]
    run(package, env, "export")
    run(package, env, "export")
    after = run(package, env, "progress")["archives"]
    if after < before:
        raise SystemExit("两次导出后进度回退：%d → %d" % (before, after))
    report["progress_archives"] = [before, after]
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
