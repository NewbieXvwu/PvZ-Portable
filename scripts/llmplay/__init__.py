"""LLM 打全程的沙箱接口（task186：扩卡组专用任务）。

用法（在包根目录下）：

    import llmplay
    print(llmplay.next_task(shard=0, shards=3))  # item 已占位
    print(llmplay.roster(task_id, seed=...))  # 先查威胁，再利用先验知识比较不同克制方案
    g = llmplay.Game(task_id, seed=..., deck=[...], reason="针对…用…；经济/主力/防守/应急/地形卡…")
    print(g.observe())
    g.export_archive(llmplay.out_dir() / "x")
"""
from .game import Game, load_tasks, task_spec, ROOT
from .delivery import (resume, carry, progress, next_task, ensure_run_id, diagnose,
                       output_dir as out_dir)
from .runtime import paths
from .decks import configure_shard
from .roster import roster
from .resources import install as _install_resources, app_data as _app_data

_resources = _install_resources(ROOT)
if _resources is not None:
    _app_data(ROOT / "build" / "pvz-portable", _resources)
# 包内带 carry/progress.json 时，第一次用就接上前次进度（out 里已有进度则什么都不做）。
carry()
ensure_run_id()

__all__ = ["Game", "load_tasks", "task_spec", "ROOT", "list_tasks", "resume", "carry",
           "progress", "next_task", "diagnose", "out_dir", "paths", "configure_shard", "roster"]


def list_tasks() -> dict:
    """本片可继续扩卡组的任务 id -> 关卡/种子/倍率/当前进度。"""
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
