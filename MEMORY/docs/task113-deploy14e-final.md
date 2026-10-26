# task113：deploy14e 定版（看门狗 + 门禁 v4 Holm 校正 + 两个根因修复）

主控决定（用户 10-26 授权主控对整个项目、包括门禁有完全裁量权）。

## 背景

- 第 20 轮门禁：总体 +0.2pp [−1.6, +2.0]，pool +5.5pp；day −3.6pp，单侧 McNemar p = 0.027，被判“显著区退步”而拒绝。
- 门禁对 6 个区逐个检验，每个都用 α = 0.05，没有做多重比较校正。在 Holm 校正下，最小 p 值要 < 0.05 / 6 ≈ 0.0083 才算显著，0.027 不算。已连续 9 轮被拒。

## 要做的（分支 deploy14e）

1. 把 deploy14e（task109 的 b7120205 / 60d8b233，基于 cefb9989）变基或合并到 **origin/deploy14d 最新 cbd96daf**。cbd96daf 是去掉 7 个 llmplay 层任务继承的 `demo_start` 的热补。
2. **门禁 v4**（`python/pvz_promotion.py`）：
   - 新增 `GATE_VERSION = 4`。面板、种子池、配对数、总体非劣 3pp 规则都与 v3 相同；
   - 区否决改为对各区单侧 McNemar p 做 **Holm–Bonferroni 逐步校正**，族错误率 0.05；
   - `policy(4)` 的 `zone_veto` 文字如实写明校正方法；
   - 不改 v1–v3 的 policy 文本，因为契约里有它们的哈希；
   - 测试：用第 20 轮的实际数字（6 区，day p = 0.027，其余见 `round_000020/evaluations/*/decision.json`）验证 v4 不否决；构造一个 p < 0.0083 的区验证 v4 照样否决；
   - DESIGN 登记 v4 及理由。
3. **单格配置错误不再让 qc 停**：
   - 规划起点格时，如果 `demo_start` 存档的种子不在该任务的 `seeds` 里，或者存档不存在，就把这格记为“配置无效”并退休、写日志，不派组；
   - 加测试。task109 的隔离若已覆盖 search_play rc=2，就补一个针对这种情况的测试。
4. **根因**：查 `scripts/llmplay/make_llmplay_tasks.py`（pvz-env 与 llmplay 分支），它生成 C 层等派生任务时复制了源族的 `demo_start`。改成不复制，并修到 pvz-env。
5. 如果分支 `demo2545`（task112：L25 / L45 示范起点格）已推送完成，一并合入；没有就不等。
6. 更新 `scripts/qc_deploy14e.sh`（台式机复制到 `~/pvz-logs/qc_deploy14e.sh`）：
   - REV、CODE_FILES、基线 SHA 按 cq 现状重算；
   - 不加回填；
   - 门禁版本切换写清楚，切换要在轮边界。
   - **不要执行**，主控执行。

## 约束

- 只跑相关测试。
- 最多 2 个游戏进程。
- 遵守 AGENTS.md §6.5。
- 推送 deploy14e，结果写进本文件末尾（合并时同步到 pvz-env 的同名文件）。


## 结果（2026-10-26，task113）

- deploy14e 已合并 origin/deploy14d 的 cbd96daf（合并 1fc6a1b7）及已推送的 demo2545/13a005c2（合并 4aab1c96）。L25/L45 两格随之进入清单；七条错误继承 demo_start 的热补保留。
- 核心实现 3573ea37：GATE_VERSION=4、独立 v4-task113 面板标识；任务、1920 配对、种子池和总体非劣 3pp 与 v3 相同。六区单侧 exact McNemar 使用 Holm–Bonferroni，族错误率 0.05；policy(1..3) 和旧决策路径不改，证书继续按签发版本核对。DESIGN §14.37 已登记理由。
- 第20轮实际 decision.json（b4e85c41a983bbe8121a）的六区计数已写入数值回归：day (both_win,gain,harm,both_lose)=(336,16,30,2)，原始 p=0.027038015929434778，v4 校正 p=0.16222809557660867；总体仍 +0.208pp，CI [−1.605,+2.022]pp，v4 不否决，v3 仍否决。8 次净 harm 的构造 p=1/256<0.05/6 仍否决，另覆盖逐步阈值及首次不显著后停止拒绝。未重跑第20轮、未回填历史决策。
- 示范格加载不再因缺存档全局退出；规划只读小型 meta.json，缺存档、未 verified、示范 seed 不在任务 seeds 时退休该格，持久化 cell_errors 并记录日志，不派组，其余格继续。未来 RESEED 块不需属于示范任务 seeds。测试覆盖缺存档/错种子/未验证、退休不复活、种子耗尽仍可扩块续跑。
- 生成器根因补丁 e87d7f18：派生任务剔除源 seeds 与 demo_start，不改源任务、卡组、地形。已同步 llmplay（28e01d19）；pvz-env 通过基于远端最新版本的独立检出接入同一补丁（6e1a4813），避免触碰本机旧 pvz-env 检出的既有删除。
- 仅运行相关门禁、控制器、pipeline、多种子、看门狗和生成器测试：153 passed、1 skipped、4 subtests passed（32.66s）。跳过的是本工作树不含未入 git 的用户示范档。随后新增的显式 v3 与轮边界保护用例 3 项通过；Holm 停止规则的补充用例通过。初次使用系统 Python 的两项 torch 依赖失败，改用项目 ml 环境后通过。
- scripts/qc_deploy14e.sh 固定运行时代码 REV=3573ea37，CODE_FILES 七项和基线 SHA 按 cq 现有文件重算，引擎和两套源面板 SHA 保留并核实。无回填步骤；代码覆盖后先以显式 --gate-version 3 完成遗留半轮及后台排空，再在轮边界以缺省 v4 启动。控制器另设门禁版本切换保护，半轮或旧后台未排空时拒绝切换。首个新组成功后启动 qc-watchdog。脚本复制到 ~/pvz-logs/qc_deploy14e.sh；只做 shell 语法和 start_qc 提取检查，未执行部署或启动看门狗。
- 本任务未启动游戏/训练进程，未写生产 state、offset、种子账本或 teacher_rates。deploy14e 与生成器补丁按任务要求推送；本结果同步至 MEMORY/docs/task113-deploy14e-final.md 与 pvz-env 同名任务书。
