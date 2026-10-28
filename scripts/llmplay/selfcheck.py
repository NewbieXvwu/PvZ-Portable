"""离线自检：不依赖仓库任何东西，只用包里自带的东西跑一遍。

    python3 selfcheck.py

顺序是硬的：**先查环境、再扫联网 import、最后才导入接口并起引擎**。
成功时最后一行是 SELFCHECK OK；失败打印 SELFCHECK FAILED 并以非零码退出。

task139 的冒烟内容：A 层建局 → 放弃 → 同种子换卡重开 → 导出存档，
再验证「跳层」会被拒绝（此时 C 层还不允许建局）。
"""
from __future__ import annotations

import ast
import json
import os
import platform
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
    try:
        # 只检查随包接口源码，不检查用户放在包旁的额外脚本。
        files = [ROOT / "selfcheck.py"] if (ROOT / "selfcheck.py").is_file() else []
        for directory in ("llmplay", "python", "scripts", "examples"):
            files.extend((ROOT / directory).rglob("*.py"))
        hits = network_imports(files)
        assert files and not hits, "接口含联网模块 import: " + str(hits)
        print("接口联网 import 扫描：0 命中")
    except (OSError, ValueError, SyntaxError, AssertionError) as error:
        print("SELFCHECK FAILED: %s" % error)
        return 1

    out = Path(tempfile.mkdtemp(prefix="llmplay3-selfcheck-"))
    os.environ["LLMPLAY_OUT"] = str(out / "delivery")
    os.environ["PVZ_LLMPLAY_SESSION"] = str(out / "budget.json")
    try:
        return _smoke(out)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _smoke(out: Path) -> int:
    try:
        from llmplay import Game, next_task, progress
        from llmplay.delivery import verify_archive
        from llmplay.game import load_tasks
    except (OSError, ValueError, AssertionError) as error:
        print("SELFCHECK FAILED: %s" % error)
        return 1
    executable = ROOT / "build" / "pvz-portable"
    if not executable.is_file() or not (ROOT / "res" / "main.pak").is_file():
        print("SELFCHECK FAILED: 缺少引擎或资源，请使用配对的两个包。")
        return 1

    tasks = load_tasks()
    print("任务数：%d（A 层 21、C 层 50、E 层 42）" % len(tasks))
    item = next_task()
    print("当前该做：%s" % json.dumps(item, ensure_ascii=False))
    task_id, seed = item["task_id"], item["seed"]
    reference = (tasks[task_id].get("reference_deck") or [])
    deck = list(reference) if len(reference) >= 6 else [1, 33, 39, 44, 15, 3]
    if task_id.startswith("llmplay3_c_"):
        deck = list(tasks[task_id]["deck"])

    # 跳层必须被拒绝：此时 C 层的指定卡组任务一律不许建局。
    challenge = next(task for task in tasks.values()
                     if task.get("layer") == "C" and int(task.get("batch") or 1) == 1)
    try:
        Game(challenge["task_id"], seed=challenge["seeds"][0], deck=challenge["deck"])
        print("SELFCHECK FAILED: 跳层没有被拒绝")
        return 1
    except ValueError as error:
        print("跳层拒绝正常：%s" % str(error)[:80])

    with Game(task_id, seed=seed, deck=deck, reason="自检：先试参考卡组", strategy="参考卡组") as first:
        print(first.observe().splitlines()[0])
        first.deck_review = "自检局：只验证接口，不评价强弱"
        first.abandon(reason="自检：同种子换卡重开")
        first.save(out / "abandoned.json")
    with Game(task_id, seed=seed, deck=deck, reason="自检：换卡重开", strategy="参考卡组") as second:
        second.wait(30)
        second.deck_review = "自检局：重开后仍能建局"
        archive = second.export_archive(out / "archive")
    meta = json.loads((archive / "meta.json").read_text(encoding="utf-8"))
    document = progress()
    assert meta["schema"] == "episode_archive_v1" and meta["label_source"] == "llm_agent"
    assert meta["cycle"] == document["cycle"] == 1
    assert len(document["archives"]) == 2, document["archives"]
    assert document["archives"][0]["abandoned"]
    assert not document["complete"] and document["layer"] == "A"
    assert (out / "delivery" / "out.zip").is_file()
    verify_archive(archive, executable=executable)
    print(json.dumps(dict(task=task_id, seed=seed, cycle=document["cycle"], layer=document["layer"],
                          archives=len(document["archives"]), verified=True), ensure_ascii=False))
    print("SELFCHECK OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
