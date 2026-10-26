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

__all__ = ["Game", "load_tasks", "task_spec", "ROOT", "list_tasks"]


def list_tasks() -> dict:
    """任务 id -> 关卡/种子/倍率，方便挑要打哪一关。"""
    return {task_id: dict(level=task["level"], seeds=task.get("seeds"),
                          multiplier=task.get("zombie_count_multiplier"),
                          deck=task.get("deck"))
            for task_id, task in load_tasks().items()}
