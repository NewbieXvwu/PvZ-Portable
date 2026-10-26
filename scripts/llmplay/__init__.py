"""LLM 打全程的沙箱接口。

用法（在包根目录下）：

    from llmplay import Game, list_tasks

    print(list_tasks())
    g = Game("demo_roof_49_doom_m200")
    print(g.observe())
    print(g.legal())
    g.plant(0, 0, 0)
    g.wait(300)
    print(g.summary())
    g.export_archive("/mnt/data/out/l49_seed20261049")
"""
from .game import Game, load_tasks, task_spec, ROOT
from .delivery import resume, progress, next_task

__all__ = ["Game", "load_tasks", "task_spec", "ROOT", "list_tasks", "resume", "progress", "next_task"]


def list_tasks() -> dict:
    """任务 id -> 关卡/种子/倍率，方便挑要打哪一关。"""
    document = progress()
    return {task_id: dict(level=task["level"], seeds=document["tasks"][task_id]["available_seeds"],
                          multiplier=task.get("zombie_count_multiplier"),
                          deck=task.get("deck"), progress=document["tasks"].get(task_id))
            for task_id, task in load_tasks().items()
            if task_id in document["tasks"] and not document["tasks"][task_id]["complete"]}
