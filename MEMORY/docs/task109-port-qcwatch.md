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

## 结果（2026-10-26）

- `deploy14e` 直接基于 `origin/deploy14d` 的 `cefb9989`，未合并 `pvz-env`。已阅读 task98 任务书与 qcwatch 结果，task98 前期审计为 `b25bb289`，实际代码和测试来自 `36f327f6`、`18326ea6`，未引入 task100 提交。
- 实现提交 `b7120205c88b8e02fafafaf3d55f5ca5a9990f00` 已推送到 `origin/deploy14e`。单格、采集组和后台快照分别隔离，临时 IO 重试一次，状态无法保存时暂停新采集。保留 deploy14d 的多种子训练/挑选、completed_groups 逐组保存和重启去重、archive 起点识别，以及种子耗尽退休。失败组从打包与挑选证据中排除，采集失败不更新胜率窗口。
- 仅运行 `python3 -m pytest -q python/test_curriculum_queue.py python/test_multiseed_production.py python/test_curriculum_pipeline.py python/test_qc_resilience.py python/test_qc_watchdog.py`：118 项通过、3 项跳过、4 个子断言通过，耗时 9.23 秒。跳过项分别需要未入 git 的示范存档、Linux 的 flock 命令与台式机本地部署脚本。
- 部署草稿为 `scripts/qc_deploy14e.sh`，结构取自通过 SSH 只读获取的 `/home/newbiexvwu/pvz-logs/qc_deploy14d.sh`，固定运行时代码提交为 `b7120205`。仅覆盖 `scripts/curriculum_queue.py`、`python/pvz_selection_refresh.py`、`scripts/qc_watchdog.sh`，保留原引擎和两套面板，回滚只恢复代码。已移除所有回填步骤。
- 草稿在首个新采集组成功后，由 `start_watchdog` 自动创建 `qc-watchdog` 会话，执行 `bash /home/newbiexvwu/PvZ-search/.claude/worktrees/cq/scripts/qc_watchdog.sh`，输出追加到 `/home/newbiexvwu/pvz-logs/qc_watchdog.out`。看门狗每 120 秒巡检，30 分钟无新行告警，死亡会话使用最新部署脚本提取出的 start_qc 重启，30 分钟两次失败后熔断，遇停止标志或部署进程时避让。
- 草稿和看门狗已通过 `bash -n`，也已从草稿提取 start_qc 并检查其语法。未执行部署草稿，未启动看门狗，未修改台式机状态，本机未运行游戏或训练。结果同步保存在本工作树同名任务书中。
