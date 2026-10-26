# task109：把 task98（qc 单点异常不停机 + 看门狗）移植到 deploy14d 之上

背景：
- task98 的成果在分支 `qcwatch`，但那条分支还混着 task100 的提交，并且基于旧代码。
- 生产现在走 deploy14d（`origin/deploy14d` = cefb9989），包含 deploy14c 多种子、task108 轮日志修复、task107 传送带示范格。

要做的：
1. 从 `origin/deploy14d` 开新分支 `deploy14e`。只移植 task98 本身的改动，不带 task100 的提交：
   - `curriculum_queue.py` 的异常分类：单组或单任务失败时，隔离该格并继续，不让整个 qc 退出；
   - `scripts/qc_watchdog.sh`；
   - 相应的测试。
   先读 task98 任务书（`MEMORY/docs/task98-qc-resilience.md`）和 qcwatch 上的结果节，弄清哪些提交属于 task98。
2. 冲突以 deploy14d 的现行行为为准：多种子、completed_groups、起点格 archive 识别、种子耗尽退休都不能回退。
3. 只跑 qc 相关测试：`test_curriculum_queue.py`、`test_multiseed_production.py`、`test_curriculum_pipeline.py`，以及 task98 新增的测试。
4. 写 `qc_deploy14e.sh` 草稿：结构照台式机 `~/pvz-logs/qc_deploy14d.sh`，用 ssh 只读查看。只替换代码，不要回填步骤。看门狗的启动方式写清楚。**不要执行。**
5. 推送 `deploy14e`，结果写进本文件末尾。

约束：
- Mac 本机已冻结，不跑游戏或训练。
- 台式机只读。
- 遵守 AGENTS.md §6.5。
