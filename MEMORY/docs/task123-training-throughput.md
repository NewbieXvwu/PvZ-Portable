# task123：学生训练吞吐——GPU 利用率 1%

主控 2026-10-27 派给 Mac codex-plus2b。分支 `trainspeed`，基于 `origin/rebalance2`。

## 现象

- 第 21 轮 K=2：02:07 排队，06:45 两个学生都还没训完（`~/pvz-logs/curriculum_state.pipeline/round_000021/seeds/*`）。第 22 轮采集 4.5 小时已经采完，**训练现在比采集慢，成了关键路径**。
- 两个 `train_rollout_policy.py` 进程各约 24% CPU、RSS 3.3–3.6 GB；`nvidia-smi` GPU 利用率 1%、显存 7.8 GB。配方 `threads: 1`、`precompute_workers: 8`、`device: cuda`、12 epoch、约 1700 万行。
- 采集同时占着 7–16 个 CPU 进程。

## 要做的

1. 在台式机上 profile 一个正在跑的或一个缩小版训练（只读生产文件，不动 qc 和正在训练的进程），找出时间花在哪里：数据装载 / 解压 / 特征 / 权重 / Python 循环 / GPU 前后向。
2. 给出并实现提速方案，目标是单个学生训练墙钟减半以上，且训练结果在固定种子下与现在一致（或说明为什么不能一致、差多少）。可考虑：数据常驻显存或内存映射、批量增大、把行权重 / 采样搬到 GPU、减少每 epoch 的 Python 开销、K=2 两个学生共享一次装载。
3. 不改训练目标、配方语义或模型结构。
4. 只跑相关测试与一次缩小版对照（AGENTS §6.5）。上线由主控安排；推送，结果写在本文件末尾。
