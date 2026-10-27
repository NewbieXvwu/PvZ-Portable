"""离线自检：不依赖仓库任何东西，只用包里自带的东西跑一遍。

    python3 selfcheck.py

成功时最后一行是 SELFCHECK OK，失败会抛异常并以非零码退出。
"""
from __future__ import annotations

import json
import os
import platform
import ast
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def network_imports(files):
    forbidden = {"socket", "urllib", "requests", "http", "httpx", "aiohttp", "ftplib",
                 "smtplib", "paramiko", "websockets", "ssl"}
    hits = []
    for path in files:
        for node in ast.walk(ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                     [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                if name.split(".", 1)[0] in forbidden:
                    hits.append("%s:%s: %s" % (path, node.lineno, name))
    return hits


def main() -> int:
    # 在导入接口或启动 ELF 前报告环境要求；本文件本身可由较旧的 Python 解析。
    if sys.version_info < (3, 10):
        print("SELFCHECK FAILED: 需要 Python >= 3.10，请换用符合要求的本地 Python 环境。")
        return 1
    if platform.system() != "Linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        print("SELFCHECK FAILED: 包内引擎需要 x86_64 Linux；请换用符合要求的运行环境。")
        return 1
    # 先扫描，再导入接口；不执行被检查的源码。
    try:
        # 只检查随包接口源码，不检查用户放在包旁/根目录的额外脚本。
        files = [ROOT / "selfcheck.py"] if (ROOT / "selfcheck.py").is_file() else []
        for directory in ("llmplay", "python", "scripts", "examples"):
            files.extend((ROOT / directory).rglob("*.py"))
        hits = network_imports(files)
        assert files and not hits, "接口含联网模块 import: " + str(hits)
        print("接口联网 import 扫描：0 命中")
    except (OSError, ValueError, SyntaxError, AssertionError) as error:
        print("SELFCHECK FAILED: %s" % error)
        return 1
    out = Path(tempfile.mkdtemp(prefix="llmplay-selfcheck-"))
    os.environ["LLMPLAY_OUT"] = str(out / "delivery")
    os.environ["PVZ_LLMPLAY_SESSION"] = str(out / "budget.json")
    try:
        from llmplay import Game, list_tasks, resume, progress, next_task, out_dir
        from llmplay import budget
        from llmplay.delivery import verify_archive
        from llmplay.game import load_tasks
    except (OSError, ValueError, AssertionError) as error:
        print("SELFCHECK FAILED: %s" % error)
        shutil.rmtree(out, ignore_errors=True)
        return 1
    exe = ROOT / "build" / "pvz-portable"
    pak = ROOT / "res" / "main.pak"
    if not exe.is_file() or not pak.is_file():
        print("SELFCHECK FAILED: 缺少引擎或资源，请使用配对的两个包。")
        shutil.rmtree(out, ignore_errors=True)
        return 1
    exe.chmod(exe.stat().st_mode | 0o111)
    budget.reset_session(out / "budget.json")
    tasks = list_tasks()
    print("任务数：%d" % len(tasks))
    if not tasks:
        print("SELFCHECK FAILED: 任务清单是空的")
        return 1

    task_id = "demo_roof_41_doom_m200" if "demo_roof_41_doom_m200" in tasks else sorted(tasks)[0]
    try:
        with Game(task_id, seed=load_tasks()[task_id]["seeds"][0]) as game:
            print("开局：%s" % task_id)
            print(game.observe())
            legal = game.legal()
            print("合法种植组合：%d 个卡槽" % len(legal["plants"]))
            # 200 步：能动就种，不能动就等 60 tick。
            for _ in range(200):
                if game.terminal:
                    break
                legal = game.legal()
                if legal["plants"] and legal["plants"][0]["cells"]:
                    first = legal["plants"][0]
                    row, col = first["cells"][0]
                    if game.plant(first["packet"], row, col):
                        continue
                game.wait(60)
            print(game.summary())

            # task81d：稳健性接口。只验形状与可复现，不要求这一小局能赢。
            def policy(trial):
                trial.wait(30)

            first = game.robustness(policy, k=2, checkpoints=[0], max_ticks=300)
            assert (out / "delivery" / "out.zip").is_file(), "robustness 没有自动交付"
            assert progress()["archives"][-1]["robustness"] == first
            second = game.robustness(policy, k=2, checkpoints=[0], max_ticks=300)
            assert first == second, "robustness() 同一打法同一种子不可复现"
            assert first["checkpoints"]["0"]["reached"] is True
            print("稳健性：%s" % json.dumps(first, sort_keys=True))

            archive = game.export_archive(out / "archive")
            print("存档：%s（%d 个操作）" % (archive, len(game.ops)))
        meta = json.loads((out / "archive" / "meta.json").read_text(encoding="utf-8"))
        assert meta["schema"] == "episode_archive_v1"
        assert meta["label_source"] == "llm_agent"
        assert meta["operations"] > 0
        assert meta["robustness"]["checkpoints"]["0"]["reached"] is True
        assert meta["manual_patch"] is False
        run_id = progress()["run_id"]
        assert meta["run_id"] == run_id
        assert meta["environment_fingerprint"]["cpu_model"]
        assert meta["environment_fingerprint"]["cpu_count"]
        assert meta["environment_fingerprint"]["python_version"]
        assert (out / "delivery" / "out.zip").is_file()
        assert len(progress()["archives"]) >= 1, "robustness/export 没有自动交付"
        verify_archive(out / "archive")
        # 默认按旧progress从zip直接恢复；引擎重放已在上面的verify_archive执行。
        previous_zip = out / "delivery" / "out.zip"
        os.environ["LLMPLAY_OUT"] = str(out / "resumed")
        resume(previous_zip)
        assert progress()["run_id"] == run_id, "续跑没有沿用 run_id"
        assert all(entry["valid"] for entry in progress()["archives"])
        with Game.load_archive(out / "archive") as restored:
            restored._start()
            assert restored.tick == meta["outcome"]["final_tick"]
        # 损坏一条操作结束 tick，识别为作废，不能跳过对应任务/种子。
        damaged = out / "damaged"
        shutil.copytree(out / "archive", damaged)
        tape = (damaged / "ticks.jsonl").read_text().splitlines()
        entry = json.loads(tape[0])
        entry["end_tick"] += 1
        tape[0] = json.dumps(entry)
        (damaged / "ticks.jsonl").write_text("\n".join(tape) + "\n")
        os.environ["LLMPLAY_OUT"] = str(out / "invalid-resume")
        resume(damaged, verify="full")
        assert progress()["archives"][0]["valid"] is False
        assert next_task() is not None
        print("持续交付 / 断点续跑 / 损坏存档识别：OK")
        print("SELFCHECK OK")
        return 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
