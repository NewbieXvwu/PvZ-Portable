# task133 交付（2026-10-28）

工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/llmplay2`，分支 `llmplay2`。
从 origin/pvz-env 979b784b 创建，合入 origin/llmplay 的现行自动交付和直接续跑接口，
并带入主控指定 CQ 的 gate v5 面板；不动 CQ、deploy14i、等待器和生产状态。

按修订1登记 F7/H12 共19个全新关卡族，全部1.0倍，G取消。
模型通过 `Game(task_id, deck=[...], reason="...", strategy="...")` 选卡，
`game.deck_review` 写该套卡组优缺点，`game.abandon(reason="...")` 自动交付并释放引擎。
同一种子可以换卡重开。日志、存档、分支和稳健复现均保留该局卡组和理由。
每次开局有 attempt_id，重复导出不会增加失败次数。

卡组按集合统计，换序不算新套；每关三套各赢两个种子、全关至少六个不同种子胜，
其中三个种子达到第10波8个未来赢≥5。两两至少更换三张，核心流派由模型声明，
主控根据卡组和优缺点评价复核其真实差异。无v5黑白名单。

两处口径按现有任务与修订解释：沿用二周目，普通植物全解锁、升级植物已购买；
首周目L2只有两张普通卡，与6–10张和多流派要求冲突。L45按修订1改成自选卡种卡模式，
原 `demo_roof_45_orig_m100` 原生传送带任务完整保留，新族不覆盖旧族。
已将这两点告知用户，未将其表述为用户另行确认。

Boss旧接口改以 `tick:6000` 为默认稳健检查点，避免wave恒0导致永远无法达标；
F/H包本身不含Boss。一个小测试验证8未来判定与原卡组传递；相关4个纯逻辑小测试通过。
唯一真实游戏冒烟为L42、seed133000000，自选卡建局→放弃→换卡重开→存档，
输出SELFCHECK OK。先查free -g，全程ulimit -v 8000000，未做通关测试或旧存档重放。

上传下列两个文件（同目录，资源zip保持原样，解压7z的001分卷）：

- `/home/newbiexvwu/pvz-llmplay2/llmplay_pkg.7z.001`：5,282,902 bytes；SHA256 `22b9c4e273a27626a3dcca7c1fbafe91ced6f43b93b2d76aaf1645f93bd66542`。
- `/home/newbiexvwu/pvz-llmplay2/llmplay_res.zip`：原资源包，SHA256 `0005bb8e31ad2e2ca7649bc0b6128211ff13c8f1c333c2163984302d7d65d94e`。若会话已有该资源包可复用。

接口压缩后不足20MB分卷阈值，因此仅001一卷，低于100MB上限。包和资源不进git。
复用原静态引擎a84f10cf与资源契约，不重建缓存、不全关复验。
