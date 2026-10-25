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

from llmplay import Game, list_tasks  # noqa: E402


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

    tasks = list_tasks()
    print("任务数：%d" % len(tasks))
    if not tasks:
        print("SELFCHECK FAILED: 任务清单是空的")
        return 1

    task_id = "smoke_roof_41" if "smoke_roof_41" in tasks else sorted(tasks)[0]
    out = Path(tempfile.mkdtemp(prefix="llmplay-selfcheck-"))
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
            archive = game.export_archive(out / "archive")
            print("存档：%s（%d 个操作）" % (archive, len(game.ops)))
        meta = json.loads((out / "archive" / "meta.json").read_text(encoding="utf-8"))
        assert meta["schema"] == "episode_archive_v1"
        assert meta["label_source"] == "llm_gpt_pro"
        assert meta["operations"] > 0
        print("SELFCHECK OK")
        return 0
    finally:
        shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
