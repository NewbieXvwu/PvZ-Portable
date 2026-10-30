# llmplay：让模型一轮接一轮地自己打

上手：解压 `llmplay_pkg.7z.001` 等全部分卷到同一目录（资源包 `llmplay_res.zip` 放在旁边，
首次 import 自动安装），然后进入 `llmplay_pkg/`：

    python3 examples/play.py                # 领一项、打一局、导出存档

接口只有一层：

    import llmplay
    from llmplay import Game, next_task, progress, resume, carry, diagnose, out_dir, paths

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

第 1 圈保持原任务和已完成记录。当前第 2 圈保留已完成的旧 21 关、直接追加 7 关；
进度为 A28/28、B28/28、C18/18、D/E0/28，下一项从 D 层开始。
第 3 圈起主线 A/B/D/E 共 22 关：屋顶 L41–L49、Boss L50，
泳池 L22/26/27/28/29，雾 L32/33/34/36/38/39/40；原 L2/6/8/12/16/18 保留前两圈成绩。
L50 稳健检查点用 `tick:6000`，其余用第 10 波。C 轨保持三批循环，第 2 批 18 项；
task177 双臂零胜项及全部门禁/挑选留出签名已换成 GPT Pro 胜局卡组（同关优先，否则同地形），
换卡组使用 `llmplay5_c_*` 新族。旧族只供首圈历史计数，不再派发。每任务仍 160 个种子，每圈用 40 个窗口。
工作量仍按 ≥10 小时设计；当前第 2 圈继续完成 D/E 层，做完自动进入第 3 圈继续。

包内 `carry/progress.json` 是上一次会话的进度记录：首次 `import llmplay` 自动接上
（已完成的层、用过的卡组与种子、B 层下传的卡组都从它重算；不重放、不复验、不读旧存档）。
out 里已经有进度时不动它。

E 层每一卡组、倍数组两胜，或连续 5 个新种子全败后用
`diagnose(task_id, deck=item["deck"], waves=[实际波次], missing="缺什么")` 完成。
旧 2.0 倍两胜照算完成，连续五败只补诊断，其余降到 1.25 倍并抵扣已有胜局；败局不跨倍数拼五连败。

需要 Python >= 3.10、x86_64 Linux。
