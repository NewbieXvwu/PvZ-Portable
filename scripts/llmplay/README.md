# llmplay：让模型自己打一局

一分钟上手：

    python3 selfcheck.py                 # 离线自检，最后一行应为 SELFCHECK OK
    python3 examples/play.py demo_roof_49_doom_m200

接口只有一层：

    from llmplay import Game
    g = Game("demo_roof_49_doom_m200")
    print(g.observe())      # 局面
    print(g.legal())        # 合法动作
    g.plant(0, 0, 0)        # 卡槽 0，种在第 0 行第 0 列
    g.wait(300)
    print(g.summary())
    g.export_archive("/mnt/data/out/l49_seed20261049")
    g.robustness(policy, k=8, checkpoints=[0, 10, 20, 30])   # 换 8 个新未来还能赢几个

细节见 PROMPT.md。一局 = (任务, 操作序列)；rewind 就是从开局重放。
