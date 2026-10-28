# task140：修掉 `test_persistent_state_error_*` 的无限写日志 / 内存暴涨

主控 2026-10-28 定。分支 `fixtest`，以生产控制器所在分支 `origin/deckcells` 为底。

## 事故

- 10-27 12:09，台式机：`pytest python/test_qc_resilience.py` 涨到 17 GB，生产 qc 被 OOM。
- 10-28 08:40 前后，Mac：两个 pytest 进程同时整套跑 `test_curriculum_queue.py`、`test_curriculum_pipeline.py`、`test_qc_resilience.py`。`pytest-150/151/152/test_persistent_state_error_bl0/curriculum.log` 分别写到 9.3、8.3、2.0 GB，Mac 磁盘写满、内存耗尽，系统冻结所有应用。

## 要做的

1. 找出 `test_persistent_state_error_*` 为什么不停写日志：测试本身的模拟状态错误是否导致控制器在重试循环里每轮写一行、永不退出？这条路径会不会在**生产**中出现（状态文件持久化失败时日志无限增长）？
2. 修复：
   - 测试必须有界，加迭代 / 时间上限，失败要快速失败；
   - 若生产代码在持久化错误时会无限重试写日志，改成退避 + 限次，最后明确报错退出，交给看门狗。
3. 在 `python/conftest.py`（没有就新建）给这三个文件加一个会话级保护：单测试超时（例如 120 秒）和 tmp 目录写入上限。实现要轻，不能拖慢正常测试（AGENTS §6.5）。
4. 只用 `-k` 跑这一个用例来验证，加 `timeout 300` 和 `ulimit -f 2000000`。遵守 AGENTS §6.5 第 8 条。
5. 生产代码若有改动，写 `scripts/qc_deploy14n.sh`（只替换相关文件，start_qc 照抄台式机最新部署脚本），不执行。结论写在本文件末尾，推送分支。
