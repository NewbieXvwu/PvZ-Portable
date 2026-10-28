"""LLM 打全程的沙箱接口（task139：循环式、逐层下传）。

用法（在包根目录下）：

    import llmplay
    print(llmplay.next_task())        # 领一项：item 已为你占位（多子代理并发不会撞项）
    g = llmplay.Game(task_id, seed=..., deck=[...], reason="...", strategy="...")
    print(g.observe())
    g.deck_review = "这套卡组前期省，后期缺群伤"   # A/D 层导出前必填
    g.export_archive(llmplay.out_dir() / "x")
"""
from .game import Game, load_tasks, task_spec, ROOT
from .delivery import (resume, progress, next_task, ensure_run_id, diagnose,
                       output_dir as out_dir)
from .runtime import paths
from .resources import install as _install_resources, app_data as _app_data

_resources = _install_resources(ROOT)
if _resources is not None:
    _app_data(ROOT / "build" / "pvz-portable", _resources)
    ensure_run_id()

__all__ = ["Game", "load_tasks", "task_spec", "ROOT", "list_tasks", "resume", "progress",
           "next_task", "diagnose", "out_dir", "paths"]


def list_tasks() -> dict:
    """本圈还没做完的任务 id -> 关卡/种子/倍率/当前进度，方便挑要打哪一关。

    不在本圈批次里的 C 层任务不会出现；已达标的任务也不会出现。
    """
    document = progress()
    return {task_id: dict(level=task["level"], seeds=document["tasks"][task_id]["available_seeds"],
                          multiplier=task.get("zombie_count_multiplier"),
                          deck=task.get("deck"), reference_deck=task.get("reference_deck"),
                          layer=document["tasks"][task_id].get("layer"),
                          cycle=document["cycle"],
                          description=task.get("description"), unlock_field=task.get("unlock_field"),
                          progress=document["tasks"].get(task_id))
            for task_id, task in load_tasks().items()
            if task_id in document["tasks"] and document["tasks"][task_id].get("active", True)
            and not document["tasks"][task_id]["complete"]}
