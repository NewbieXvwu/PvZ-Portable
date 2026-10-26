"""LLM 打全程的沙箱接口。

用法（在包根目录下）：

    from llmplay import Game, list_tasks, out_dir

    print(list_tasks())
    g = Game("demo_roof_49_doom_m200")
    print(g.observe())
    print(g.legal())
    g.plant(0, 0, 0)
    g.wait(300)
    print(g.summary())
    g.export_archive(out_dir() / "l49_seed_example")
"""
from .game import Game, load_tasks, task_spec, ROOT
from .delivery import resume, progress, next_task, ensure_run_id, output_dir as out_dir
from .runtime import paths
from .resources import install as _install_resources, app_data as _app_data

_resources = _install_resources(ROOT)
if _resources is not None:
    _app_data(ROOT / "build" / "pvz-portable", _resources)
    ensure_run_id()

__all__ = ["Game", "load_tasks", "task_spec", "ROOT", "list_tasks", "resume", "progress", "next_task", "out_dir", "paths"]


def list_tasks() -> dict:
    """任务 id -> 关卡/种子/倍率，方便挑要打哪一关。"""
    document = progress()
    return {task_id: dict(level=task["level"], seeds=document["tasks"][task_id]["available_seeds"],
                          multiplier=task.get("zombie_count_multiplier"),
                          deck=task.get("deck"), progress=document["tasks"].get(task_id))
            for task_id, task in load_tasks().items()
            if task_id in document["tasks"] and not document["tasks"][task_id]["complete"]}
