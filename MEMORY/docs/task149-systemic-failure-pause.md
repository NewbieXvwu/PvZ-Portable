# task149：游戏进程全线崩溃时，采集队列暂停，不再空转烧掉整轮

主控 2026-10-29 定。分支 `sysfail`，以 `origin/prodint` 为底（生产代码 deploy14p = `4beab441`）。

## 事故

2026-10-28 23:52–23:57，台式机更新显卡驱动替换了系统库。之后每个游戏进程一启动就报 `Inconsistency detected by ld.so: dl-setup_hash.c ... Assertion failed`。`curriculum_queue.py` 把每组都记成 `group skipped: incomplete/invalid game results: 0/24`，5 分钟内把第 37 轮 38 组全部“跑完”。结果是：
- 记了 `last_round`；
- 写了 `round 37 done ... 768 games`（实际 0 局有效）；
- 把一个没有新数据的训练任务排进了后台队列。

同一时间，后台的 r34 选拔与 r35 打包也因同一原因失败。

## 要做的

1. 识别“系统性失败”：例如连续 N 组（N=3）都是 0 有效局，或者失败用时远短于正常组。出现时队列停在组边界，不再继续空转。
   - 写一行清楚的日志；
   - 每隔几分钟用一组很小的探测（1 局）重试，恢复后自动继续；
   - 不把这些组算进 `last_round`，也不算进本轮完成数。
2. 整轮 0 有效局时，不入队后台训练。
3. 后台任务（选拔、打包、训练）因同类原因失败时，不消耗 task140 的有限重试次数，等探测恢复后再重试。
4. 测试：只用 `-k` 跑相关用例。Mac 上每次运行都加 `timeout 300` 和 `ulimit -f 2000000`，同一时间只开一个 pytest（AGENTS §6.5 第 8 条）。**禁止整套运行** `test_curriculum_queue.py`、`test_curriculum_pipeline.py`、`test_qc_resilience.py`。
5. 结论写在本文件末尾，推送 `sysfail`。部署由主控决定。

## 约束

- 不部署、不碰台式机生产。
- 不加自校验。
