# task16（主控 2026-10-20 下发给 Mac CodeBuddy）：学生网络的“按行汇总”与辅助预测头

先读仓库根目录 `AGENTS.md`（优先级最高），再读本文。

## 背景
学生 `RolloutNet`（`python/pvz_rollout_policy.py`，宽 64、`--pool meanmax`、obs v2，共 14.7 万参数）的主干是
3×3、3×3、1×5 三层卷积，每格只能直接看到同行左右各 4 列；更远的信息只能经过整盘的 mean/max 得到，而整盘池化
不分行。所以“第 0 列的格子”分不清第 8 列的僵尸在自己这一行还是别的行。两个候选改动都不是手写规则，都是网络结构 / 训练目标：

- **A 按行汇总**：主干输出 `h [B,W,6,10]` 后，按行取列方向的 mean 和 max 得 `[B,2W,6,1]`，经一个可学 1×1 卷积（2W→W）
  投影，广播回 10 列，与 h 相加后过 ReLU，得到 h'。之后整盘池化、出卡头、选格头全部改用 h'。新增约 8k 参数。
- **B 辅助预测头（只在训练时用）**：从 h' 接一个 1×1 卷积到每行一个 logit（按列取 max 或 mean 汇总成行），预测
  “这一行在接下来 K 个决策点内是否有植物被吃掉 / 被毁 / 有割草机启动”。标签从已录档案的后续帧里直接取（模拟器给出的结果，
  不是人写的规则）。损失 = BCE × `--aux-weight`。推理与 C++ 完全不用这个头，导出时去掉。

## 要做的（在 worktree 里做，只做本地提交，不 push，不合并）
- 工作区：`cd /Users/newbiexvwu/PvZAgent && git worktree add .claude/worktrees/rowpool -b rowpool pvz-env`。
  只在这个 worktree 改代码；单独编译：`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`。
  **主仓库 `/Users/newbiexvwu/PvZAgent` 的工作区与 `build/` 不许碰**（Mac 正式采集在用）。同时最多 4 个游戏进程。
1. Python：`RolloutNet(..., row_pool: bool = False)`；`train_rollout_policy.py` 加 `--row-pool` 与 `--aux-weight`（默认 0 = 关）、
   `--aux-horizon K`（默认 8 个决策点）。检查点里记 `row_pool`，`load_student` 读回。默认值下结果必须与现在逐位一致。
2. 导出格式：新版本号 `PVZRN005`（头里加 `row_pool` 字段），旧版本照常能读。
3. C++ CPU 前向（`src/EnvironmentRollout.cpp` 读模型与前向那几段）：实现 row pool；与 Python 前向对拍
   （仿 `python/test_cpp_rollout.py`，随机权重 + 真实局面，误差 < 1e-5，出卡与选格的 argmax 完全一致）。
4. CUDA（`cuda/SimHost.cpp`、`cuda/RolloutCuda.cu`、`cuda/NetForward.inc`）：本机编不了，**只加明确报错**：读到 `PVZRN005` 且
   row_pool=1 时 `Fail("row_pool not supported on CUDA yet")`。不要改 CUDA 前向。
5. 辅助标签：在 `load()` 的同一遍里从档案后续帧算出（注意 wait 行抽样后行与帧的对应），缓存进 `cache_dir` 时带版本号，
   不能让旧缓存被误用。写一个单元测试：手工构造 3 帧小档案，标签符合定义。
6. 冒烟：用本机 `artifacts/student/frames_v2` 里任意 40 局训 2 个 epoch（`--device cpu`），四种组合
   （默认 / `--row-pool` / `--aux-weight 0.5` / 两者都开）都能跑完、能被 C++ 加载、`scripts/search_play.py --rollout-policy` 下完 2 局。
7. 跑现有全部测试（`python3 -m pytest python -q`），全部通过。

## 不要做的
- 不训正式模型、不跑大评估（主控在台式机 GPU 上排 2 种子对照）。
- 不改默认值；不动现有任何 `--pool` 选项的语义。
- 不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。

## 交付
- 每完成一步本地提交一次（分支 `rowpool`）。
- 完成后在本文末尾追加“结果”一节：提交号、对拍误差、冒烟四种组合的训练损失与 C++ 加载结果、测试数，以及 C++ 单步前向耗时
  （默认 vs row_pool，各 1 万次取中位数）。
- 有疑问写在本文末尾“问题”一节，按最佳判断继续，不要停下来等。
