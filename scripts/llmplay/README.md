# llmplay：让模型一轮接一轮地自己打

上手：解压 `llmplay_pkg.7z.001` 等全部分卷到同一目录（资源包 `llmplay_res.zip` 放在旁边，
首次 import 自动安装），然后进入 `llmplay_pkg/`：

    python3 examples/play.py                # 领一项、打一局、导出存档

接口只有一层：

    import llmplay
    from llmplay import Game, next_task, progress, resume, diagnose, out_dir, paths

    todo = next_task()      # 领一项：todo["item"] 已为你占位，永远不会返回 None
    item = todo["item"]
    g = Game(item["task_id"], seed=item["seed"], deck=item.get("deck") or [1,33,39,44,15,3],
             reason="双投手", strategy=item.get("strategy") or "投手压制")
    print(g.observe())      # 局面
    g.deck_review = "前期省，后期缺群伤"      # A/D 层导出前必填
    g.export_archive(out_dir() / "l41_seed_1")

**这个任务没有终点**：一圈（A 探索 → B 稳健 → D 打法 → E 加压，与 C 挑战轨并行）做完自动进下一圈，
`progress()["complete"]` 永远是 `False`。停止只由用户在对话里说停。

多个子代理可以同时调用 `next_task()`：每次返回一项还没被领走的任务，占位记录在 `out/claims.json`，
进程没了 15 分钟自动释放；全部被领走时 `item.action == "busy"`。C 轨专职子代理用
`next_task(track="C")`（C 优先，做完接主线）。

纪律全是当场计数的规则：队列指定的关卡与种子、A 层新卡组差 ≥3 张、D 层打法不重名、
同种子同卡组放弃 ≥2 次不计数、C 轨每圈必做量。没有事后重放、没有摘要比对、没有启动自检。
细则见 PROMPT.md。

导出存档或完成稳健性检查时，接口自动更新 `out_dir()` 中的 `progress.json`、`report.md`、
`budget.json` 和 `out.zip`。原子发布失败时保留上一份。测试可用 `LLMPLAY_OUT` 指定交付目录。

本轮 113 个任务：A 层 21 关（屋顶 L41–L49 + 12 个白天/夜间/泳池/雾夜关）、
C 层 50 个指定卡组挑战（task129 里老师 ≤2/8 的 46 套 + 4 个留出卡组，按关卡分三批循环）、
E 层 42 个加压任务（每关 1.5 / 2.0 倍）。每个任务 160 个种子，每圈用 40 个的窗口。

需要 Python >= 3.10、x86_64 Linux。
