"""离线自检：不依赖仓库任何东西，只用包里自带的东西跑一遍。

    python3 selfcheck.py

成功时最后一行是 SELFCHECK OK，失败会抛异常并以非零码退出。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from llmplay import Game, list_tasks, resume, progress, next_task  # noqa: E402
from llmplay import budget  # noqa: E402
from llmplay.delivery import verify_archive  # noqa: E402


def main() -> int:
    problems = []
    exe = ROOT / "build" / "pvz-portable"
    pak = ROOT / "res" / "main.pak"
    if not exe.is_file():
        problems.append("缺少引擎 build/pvz-portable")
    if not pak.is_file():
        problems.append("缺少资源 res/main.pak")
    if problems:
        print("SELFCHECK FAILED: " + "; ".join(problems))
        return 1
    # 解压工具不一定保留可执行位（Python 的 zipfile 就不保留）。
    try:
        exe.chmod(exe.stat().st_mode | 0o111)
    except OSError:
        pass

    # 自检使用完全独立的预算/交付目录，不能污染或受用户已完成任务影响。
    out = Path(tempfile.mkdtemp(prefix="llmplay-selfcheck-"))
    os.environ["PVZ_LLMPLAY_OUT"] = str(out / "delivery")
    budget.reset_session(out / "budget.json")
    tasks = list_tasks()
    print("任务数：%d" % len(tasks))
    if not tasks:
        print("SELFCHECK FAILED: 任务清单是空的")
        return 1

    task_id = "demo_roof_41_doom_m200" if "demo_roof_41_doom_m200" in tasks else sorted(tasks)[0]
    try:
        with Game(task_id) as game:
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
        assert meta["label_source"] == "llm_gpt_pro"
        assert meta["operations"] > 0
        assert meta["robustness"]["checkpoints"]["0"]["reached"] is True
        assert meta["manual_patch"] is False
        assert (out / "delivery" / "out.zip").is_file()
        assert len(progress()["archives"]) >= 1, "robustness/export 没有自动交付"
        verify_archive(out / "archive")
        # 从 zip 恢复到另一目录，必须实际经包内引擎核验。
        previous_zip = out / "delivery" / "out.zip"
        os.environ["PVZ_LLMPLAY_OUT"] = str(out / "resumed")
        resume(previous_zip)
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
        os.environ["PVZ_LLMPLAY_OUT"] = str(out / "invalid-resume")
        resume(damaged)
        assert progress()["archives"][0]["valid"] is False
        assert next_task() is not None
        print("持续交付 / 断点续跑 / 损坏存档识别：OK")
        print("SELFCHECK OK")
        return 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
