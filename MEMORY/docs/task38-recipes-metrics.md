# task38（主控 2026-10-23 08:20 下发给台式机 Codex 第三会话 codex3）：训练配方文件化 + 统一运行指标日志

先读 `AGENTS.md`、`MEMORY/docs/hotpath-audit.md` §6（开关清单）、`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22/10-23 的几条、`scripts/train_rollout_policy.py`、`scripts/unified_panel.py`、`scripts/curriculum_queue.py`。

## 背景
1. **开关膨胀**：scripts 里 732 个命令行参数（`train_rollout_policy` 34 个）。每次训练靠手拼长命令行（`scripts/run_dw4_study.py`、`run_identity_validation.py` 等里各写一份），配方无法一眼对比，也留不下“这一版到底用了什么”。
2. **看不见运行状态**：10-22 晚 dw5i 训练只有 8% 时间在跑（磁盘狂读、GPU 0%），过了几小时才被代理事后发现；WSL 被内存挤死前也没有任何记录。

## 要做的（工作树 `git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/recipe -b recipe origin/pvz-env`）
1. **配方文件**：`recipes/<名>.json`（进 git，KB 级），字段 = 训练需要的全部设置（数据来源清单 / 行权重 / 模型结构 / 优化器 / epoch / 种子 / 验证集划分 / 输出位置）；
   `train_rollout_policy.py --recipe recipes/x.json` 读它，**命令行只保留 `--recipe` 与少数运维参数**（输出目录、设备、线程数）。其余训练参数从命令行删除（按 task33 §6 的批准清单与例外；配方里出现的字段才是唯一来源）。
   把 dw4i、dw5i、identity_dw3_s1 的实际配置反写成三个配方（从它们的 `config.json` / 运行脚本取，逐字段核对），验证用配方复训的前 200 步损失与原训练一致到 1e-6（同数据同种子）。
   `scripts/run_*study*.py` 等手拼命令的地方改为引用配方；面板脚本同理把评估设置写成 `recipes/panel_v2.json` 之类。配方文件的 schema 写在 `python/pvz_recipe.py`（校验 + 默认值 + 版本号），未知字段报错。
2. **统一运行指标日志**：`python/pvz_metrics.py`——一个后台线程每 30 s 往 `~/pvz-logs/metrics/<进程名>-<pid>.jsonl` 追加一行：时间、本进程与子进程 RSS、系统 MemAvailable、`/proc/pressure/memory|io`、本进程累计读写字节（`/proc/self/io`）、
   CPU 利用率、GPU 利用率与显存（有 `nvidia-smi` / NVML 才记）、以及调用方传入的进度计数（训练：step / 行每秒；采集：局数 / 决策每秒）。训练、采集 worker 汇总、课程控制器、面板都接上。
   再写 `scripts/metrics_report.py`：读这些 jsonl，给出“训练有效时间占比（GPU 利用率 > 50% 的时间比例）/ 磁盘读写 / 内存最低点”等摘要；**训练有效时间占比 < 50% 持续 10 分钟就在日志里写 WARN**（主控巡检时 grep）。
   指标日志按天轮转、保留 7 天。
3. 报告 `MEMORY/docs/recipes-metrics.md`：配方 schema、迁移前后命令对照、指标字段说明、示例摘要。

## 规则
- 不动 tmux 队列与别的代理会话；不改别的工作树与主工作区；最多 2 个你自己的游戏进程，训练对拍在 GPU 空闲时做（`pgrep -f "train_rollout_policy|train_value"` 为空）。
- 分支 `recipe` 提交，可以 push；不推 pvz-env、不合并。不加新开关（`--recipe` 是替换几十个参数的唯一入口）；不写字面密钥；不 `git add -f`；大文件不进 git。只用绝对路径。
- 每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续。

## 结果
- 2026-10-23：新增严格版本化配方 schema，反写 dw4i（170 标签）、dw5i（281）、identity_dw3_s1（82）；训练 CLI 收敛为 recipe/out/device/threads/precompute-workers，保存 resolved recipe；schema 4 项与 identity 训练 5 项测试通过。200 步历史对拍及调用方迁移待做。
- 2026-10-23：指标接入学生/价值训练、采集 worker 共享进度汇总、课程控制器和统一面板；单 worker 真实冒烟 1 局完成，1086 个决策与指标计数一致；metrics 4 项、课程控制器 11 项测试通过。
- 2026-10-23：完成独立运行指标模块与 metrics_report（30 秒采样、进程树 RSS/CPU、PSI、IO、可选 GPU、进度速率、10 分钟 WARN、按日轮转/7 天保留）；4 项测试通过，尚待各入口接入。

## 问题
- 本分支缺 hotpath-audit.md；已 fetch 并完整读取 origin/audit 对应报告（含 §6），依据 task33 批准清单删除训练 EMA/全候选软目标入口，保留 cell soft temperature、width 与旧模型元数据。退役前标记 retire-recipe-training-flags-20261023。
