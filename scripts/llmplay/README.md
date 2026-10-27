# llmplay：让模型自己打一局

上手：先上传 `llmplay_pkg.zip` 和 `llmplay_res.zip` 到同一目录，只解压接口包。
资源 zip 留在 `llmplay_pkg/` 旁边，首次 import 自动安装游戏资源和预编译缓存；随后进入
`llmplay_pkg/` 执行：

    python3 selfcheck.py                 # 离线自检，最后一行应为 SELFCHECK OK
    python3 examples/play.py llmplay2_f_l42_m100

接口只有一层：

    from llmplay import Game, resume, next_task, progress, out_dir, paths
    paths()
    # 续跑时先 resume(out_dir().parent / "previous_out.zip")
    print(next_task())      # 自动跳过已达标任务，全部完成才返回 None
    g = Game("llmplay2_f_l42_m100", deck=[1,33,32,34,2,3,17], reason="双投手", strategy="双投手")
    print(g.observe())      # 局面
    print(g.legal())        # 合法动作
    g.plant(0, 0, 0)        # 卡槽 0，种在第 0 行第 0 列
    g.wait(300)
    print(g.summary())
    g.export_archive(out_dir() / "l49_seed_example")
    g.robustness(policy, k=8, checkpoints=[10])   # 换 8 个新未来还能赢几个

导出存档或完成稳健性检查时，接口自动更新 `out_dir()` 中的 `progress.json`、`report.md`、
`budget.json` 和 `out.zip`。原子发布失败时保留上一份。没有时间上限，五层每任务至少
5 个种子通关、其中至少 3 个稳健才停止。测试可用 `LLMPLAY_OUT` 指定交付目录。

细节见 PROMPT.md。一局 = (任务, 操作序列)；rewind 就是从开局重放。

每次新运行的随机 run_id 决定任务内种子顺序；resume 保留它。需要 Python >= 3.10、x86_64 Linux；先跑 selfcheck 按报错处理。

本轮按 F→H，共19关；每关三套不同流派、两两换至少3张，各赢≥2种子且全关≥6胜，≥3种子稳健。
参考卡组见 list_tasks()；g.deck_review 填优缺点评价，g.abandon(reason="...") 自动存档并释放引擎，可同种子换卡重开。
