# llmplay：让模型自己打一局

一分钟上手：

    python3 selfcheck.py                 # 离线自检，最后一行应为 SELFCHECK OK
    python3 examples/play.py demo_roof_49_doom_m200

接口只有一层：

    from llmplay import Game, resume, next_task, progress
    # 续跑时先 resume("/mnt/data/previous_out.zip")
    print(next_task())      # 自动跳过已达标任务，全部完成才返回 None
    g = Game("demo_roof_49_doom_m200")
    print(g.observe())      # 局面
    print(g.legal())        # 合法动作
    g.plant(0, 0, 0)        # 卡槽 0，种在第 0 行第 0 列
    g.wait(300)
    print(g.summary())
    g.export_archive("/mnt/data/out/l49_seed20261049")
    g.robustness(policy, k=8, checkpoints=[0, 10, 20, 30])   # 换 8 个新未来还能赢几个

导出存档或完成稳健性检查时，接口自动更新 `/mnt/data/out/progress.json`、`report.md`、
`budget.json` 和 `out.zip`。原子发布失败时保留上一份。没有时间上限，五层每任务至少
5 个种子通关、其中至少 3 个稳健才停止。测试可用 `PVZ_LLMPLAY_OUT` 指定交付目录。

细节见 PROMPT.md。一局 = (任务, 操作序列)；rewind 就是从开局重放。
