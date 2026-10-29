# task164：训练拖住轮次——拆分耗时、找加载瓶颈、给提速方案

> 2026-10-29。工作树 `~/PvZ-search/.claude/worktrees/trainspeed`（分支 `trainspeed`，基于 `origin/prodint16`）。
> 只读生产数据，不部署、不改生产、不碰 qc / 看门狗 / `stop_curr`，不 `pkill -f`。

## 背景

task160-D 的生产画像（2026-10-29）：

- 训练中位 **3.8 h/轮**，采集 1–2.4 h；45 轮里只有 14 轮训练完成。
- 显卡只忙 **62%**，看起来像数据加载 / CPU 侧瓶颈。
- 每轮 4 个候选、12 epoch、2264 万行（r36），单候选墙钟 3484 s。

已知相关事实（都是别人测过的，先引用，不重复劳动）：

- task61 把 batch 的 plane 散射从"每个身份一个 kernel"改成一次融合 gather+scatter，训练步快 12.6×（`scripts/train_rollout_policy.py:200` 的注释记了 218 ms → 几 ms）。
- task33/32 把棋盘改成 CSR 紧凑存储、合并映射复用（`origin/trainspeed` 上 task123 的"共享学生训练合并映射"就是这条线的产物）。
- task15 证明过：数据量变大时步数预算会改变结论（+4.35pp），所以"少训几轮"不是免费的。

## 要回答的问题

1. 每轮的时间到底花在哪：打包 / 合并、每个候选的训练（加载 + 每个 epoch + 验证）、挑选 / 门禁评估、候选数。
2. 训练器加载路径有没有明显问题：解压、拷贝、单线程、每 epoch 重读、软标签展开、逐 batch 的 CPU 侧 gather / H2D。
3. 在生产不训练时，用小子集离线测数据加载吞吐与 GPU 步吞吐，量化"CPU 组装 vs GPU 计算"的比例。
4. 按收益排序列出提速方案，每项写明是否改变训练结果（逐位一致 / 统计等价 / 会变）。

## 步骤

1. **任务书**（本文件）先提交。
2. **拆轮次耗时**：读 `~/pvz-logs/curriculum.log` 与 `curriculum_state.pipeline/round_*/{pack.log,packed.json,trained.json,seeds/*/train.log,student/metrics.json,evaluation.log,selection.log}`，写 `scripts/task164_round_timings.py` 汇总：
   - 每轮 pack 秒数、packed rows、built / reused 段数；
   - 每候选 load_seconds、每 epoch seconds、验证占比；
   - 训练总时长 vs 挑选 / 门禁时长 vs 采集时长；
   - 候选数、是否 A/B 双配方。
3. **读加载与训练路径**：`scripts/train_rollout_policy.py`（`BoardStore.batch`、`read_segments`、`batch_loss`、验证循环）、`python/pvz_training_pack.py`、`python/pvz_recipe.py`。列出每一处可疑开销，标出代码行。
4. **离线小规模实测**（台式机，生产不训练时；`systemd-run --user` + `MemoryMax=4G` + `Nice=10`，单进程）：
   - 用 8–16 个 segment（约 1–2 GB）构造与生产同构的 store，测：加载吞吐（行/秒）、每 batch 组装耗时（CPU）、前向+反向耗时（GPU）、端到端 rows/秒；
   - 用 `nvidia-smi` 采样 GPU 利用率；
   - 与生产实测（2264 万行 / 每 epoch 约 270 s ≈ 84k 行/秒）对照。
5. **方案表**：按收益排序，写清每项的预计倍数、改动范围、对结果的影响（逐位一致 / 统计等价 / 会变）、风险。
6. 结论追加本文件，提交推送 `trainspeed`，然后停下等主控。

## 约束

- 台式机是生产机：先 `free -g`、`nvidia-smi`；只在**生产不训练**时跑实验；`systemd-run --user --unit=... -p MemoryMax=4G -p Nice=10`；同时最多 1 个自己的进程。
- 不碰 qc、看门狗、`stop_curr`、生产状态文件；不 `pkill -f` 任何含 `curriculum_queue` 的模式。
- 测试只跑 `-k`，加 `ulimit -v 8000000`、`timeout 300`；不整套跑 `test_curriculum_queue.py` / `test_curriculum_pipeline.py` / `test_qc_resilience.py`。
- 只写报告与小实验脚本；不改生产训练器、不改默认值、不部署。
- 检查点、npz、log 不进 git；不写字面密钥。

## 结果

（待填）
