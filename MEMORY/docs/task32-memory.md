# task32（主控 2026-10-22 22:45 下发给 Mac 上的 CodeBuddy 会话 cb-mem）：压内存——训练、采集、价值网络

先读仓库根目录 `AGENTS.md`（优先级最高），再读 `MEMORY/notes/desktop.md`（WSL 只有 24 GB→28 GB、内存打满时 wsl.exe 连不上的事故）、
`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 的几条、`MEMORY/docs/unified-panel.md` 第 52 行起的训练资源表。

## 背景
台式机 WSL（24 GB）今晚 21:18 起被内存挤满，`wsl.exe` 再也连不上（里面的任务还在跑，但无法管理，只能重启）。同时在跑的是：
学生训练 dw5i（`scripts/train_rollout_policy.py`，dw4i 时 1364 万行、子树 RSS 求和峰约 15 GB、最低可用约 4 GB；dw5i 行数更多）、
课程采集 8 个游戏进程（每个 = Python `search_play` 约 90 MB + 引擎 `pvz-portable` 约 220 MB，Mac 实测）、价值网络补采 / 训练（`scripts/train_value_net.py`、10-15 一次 16.6 GB 被 OOM 杀）。
数据只会越来越多（课程、夜间、泳池、换卡组），不压内存下一次合训就进不去。用户认为很多东西还能压。

## 要做的
工作区（Mac）：主控已建 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/mem`（分支 `mem`，基于 pvz-env）。Mac 有 48 GB，正跑着 10 个采集进程（别停它们）；
数据可用 `/Users/newbiexvwu/PvZAgent/artifacts/hard_mac/frames`（只读）。Python 用系统 `python3`（有 torch）。不需要 GPU。

1. **量清楚钱花在哪**（先量后改）：
   - 学生训练：`train_rollout_policy.py` 的数据装载（`python/pvz_frame_store.py`、`python/pvz_training_rows.py`，含特征缓存、CSR 棋盘、标签、软目标、`packet_info`、验证集）——
     用 Mac 上一个子集（如 20–50 万行）测**每行每个组成部分的字节数**和峰值（`tracemalloc` / `resource.getrusage` / 逐数组 `nbytes`），外推到 1500 万行。
   - 采集：一个 `search_play` worker（Python + 引擎）的常驻构成：torch 导入、学生 / 价值网络权重、观测缓存、快照；引擎侧快照与推演缓冲。
   - 价值网络训练：`train_value_net.py` / `train_value_branches.py` 的装载方式。
   结果写成表（组成部分 → 字节 / 行 → 1500 万行时 GB）。
2. **压**（每一项都要证明**结果不变**）：例如更紧的 dtype（能装下就用 uint8 / int16 / float16 存、计算时再转）、把只读大数组放进 `np.memmap`（磁盘缓存文件，按需换页）而不是常驻、
   去掉重复副本（训练集 / 验证集切片时别复制、标签与特征别各存一份）、分块装载、采集 worker 不必要的 torch 线程池 / 缓存。
   **硬门槛**：同一数据同一种子，改前改后训练出的模型权重**逐位相同**（或若因浮点顺序不能逐位，前 200 步损失逐步一致到 1e-6，并说明原因）；采集同种子对局 games.jsonl 逐局相同。
   做不到逐位相同的改法放进“问题”，由主控决定。
3. 所有改动做成**默认关闭的开关**或证明逐位相同后直接生效（逐位相同的可以直接改默认，写清证据）；在 Mac 上给出改前 / 改后的峰值内存与耗时对比表。
4. 报告写 `MEMORY/docs/memory-budget.md`：现状账本、每项改动省多少、改后 1500 万行训练 + 16 采集 + 价值网络在 28 GB 里怎么排。

## 规则
- 不动 Mac 上正在跑的采集（`mac_dagger_*`、`search_play`）与别的会话；同时最多 2 个你自己的游戏进程；不改主工作区与别的工作树。
- 分支 `mem` 提交，可以 `git push origin mem`；不推 pvz-env、不合并。不写手工规则；不写字面密钥；不 `git add -f`；`.pt/.npz/.log`/缓存文件不进 git。
- 只用绝对路径。每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续，不要停下来等。

## 结果

## 问题
