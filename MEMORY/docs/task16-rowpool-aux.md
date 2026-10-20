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

## 问题

1. 本机主仓库和本 worktree 均没有原定的 `artifacts/student/frames_v2`。冒烟改用主仓库既有 `artifacts/student/frames_deep_mac` 的 40 局，种子 1400512–1400551、标签 `deep_fast_search`，仅读取源档案并复制到本 worktree 的 `artifacts/student/frames_v2`。这些旧档案缺少原始 `v2` 字段，训练使用仓库已有的 v2 旧档兼容编码，缺失字段保持既有处理。因此本次验证覆盖 v2 网络输入尺寸和训练路径，未验证原定 v2 完整档案的数据分布。
2. 现有档案没有逐株植物的消失原因。辅助标签采用相邻帧 `(row, col, type)` 计数差，排除已执行的铲除，割草机按首次进入状态 2 或待命后直接消失计算。自然耗尽的一次性植物也会被计入，无法将其与被吃或被毁精确区分。目标的实际含义是“行内植物消失或割草机首次触发”。K 以抽样前的原始决策点计数，训练行 `frames[i-1] → frames[i]` 对应从 `frames[i-1]` 之后开始的 K 步，档案尾部截到最后一帧。

## 结果（Codex，2026-10-20）

全部代码和构建产物位于既有 `rowpool` worktree。本地提交，无 push、无合并，主仓库工作区及其 `build/` 未作改动。执行开始时主仓库已有 10 个正式采集游戏进程，本任务的游戏进程始终串行、峰值 1 个，未操作外部采集进程。检查点及训练缓存均留在忽略目录，未加入 git。

### 实现与提交

| 步骤 | 本地提交 | 完成内容 |
|---|---|---|
| 1 | `3611a910` | Python 可选按行汇总残差、训练专用辅助头、三个训练参数、检查点元数据及推理加载恢复 |
| 2 | `5fb5f5dc` | `PVZRN005` 导出，头部依次为 width / pool / context / observation / row_pool，投影权重附在旧权重列表末尾，辅助头不导出 |
| 3 | `a20245c3` | C++ CPU 加载与前向、Python 环境包装器识别新格式、真实局面对拍与 argmax 检查 |
| 4 | `7e2b87f3` | CUDA 两个加载器对 row_pool=1 明确报错，未改 `NetForward.inc` 或 CUDA 前向 |
| 5 | `b6220803` | 同次档案读取计算辅助标签，`aux_v1_kK` 缓存版本及源码摘要，按原帧索引对齐抽样行，BCE 辅助损失与三帧单测 |
| 6 | `0bac126c` | 四组合各 40 局、CPU 两轮训练及各两局 C++ 实战冒烟结果 |
| 7 | `fcb101eb` | 全部 Python 测试和各 10000 次 C++ 前向耗时记录 |

按行投影在宽 64 时增加 8256 个参数，辅助头另加 65 个。辅助头仅在训练和验证计算辅助损失时运行，`load_student` 丢弃该头，推理权重与 C++ 文件均不包含它。关闭按行汇总时继续使用旧导出版本，保持既有格式字节一致。CUDA 对 `PVZRN005` 读取完整头后，row_pool=1 在读取权重前报 `row_pool not supported on CUDA yet`，row_pool=0 沿用既有权重布局。本机没有 CUDA 工具链，未编译 CUDA 设备代码。

### 对拍与默认兼容性

随机权重加真实对局状态共 48 组配置，覆盖 mean / meanmax、上下文开关、v1 / v2lite / v2 以及 row_pool 开关。关闭时宽 13，开启时宽 64，开启组先推进到至少 tick 6000，使输入包含实际植物和僵尸。三个输出头的最大绝对误差为 **1.6391277313232422e-7**，低于 1e-5，选卡和选格 argmax 全部一致，合法贪心动作完全一致。

与任务开始时保留的旧实现比较，6 组配置的默认初始化权重及三个输出头逐位一致，12 组配置的旧格式导出字节完全一致。

### 四组合冒烟

四组共同参数为 `--epochs 2 --device cpu --pool meanmax --obs v2 --seed 17`，使用同一批 40 局，合计 34596 个决策，其中 1977 个种植动作，按种子留出 3465 行验证数据。表中“默认”指两个新开关均关闭，辅助开启组使用 `--aux-weight 0.5 --aux-horizon 8`。损失为原模仿损失加开启时的加权辅助 BCE。

| 组合 | 第 1 / 2 轮训练损失 | 第 1 / 2 轮验证损失 | C++ 加载 | 完整对局 |
|---|---|---|---|---|
| 默认 | 6.283524 / 5.160043 | 5.492059 / 4.518775 | 成功，PVZRN004 | 2/2，0 胜 |
| `--row-pool` | 6.267214 / 5.258334 | 5.595454 / 4.691534 | 成功，PVZRN005 | 2/2，0 胜 |
| `--aux-weight 0.5` | 6.476728 / 5.286720 | 5.603365 / 4.701046 | 成功，PVZRN004 | 2/2，0 胜 |
| 两者同开 | 6.430961 / 5.226051 | 5.593489 / 4.753039 | 成功，PVZRN005 | 2/2，0 胜 |

本地模型及指标位于 `artifacts/student/task16_smoke/{default,rowpool,aux,both}/`。四组均通过 `scripts/search_play.py --rollout-policy <组目录>/model.pt --arms fast --workers 1 --rollout-sim-skip 0 --executable <本 worktree>/build/pvz-portable` 运行 `day_l7_z13` 的种子 30000、30001。脚本导出并加载 C++ 权重后在 C++ 内下完整局，每组两局分别为 334、276 个决策，在第 7、第 6 波终局，未触及 3000 决策上限。CLI 必需的价值网络参数使用未训练占位权重，fast 臂不调用该网络。

### 测试与耗时

独立构建命令为任务书指定的 `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`，编译成功。`python3 -m pytest python -q` 结果为 **215 passed、12 skipped、423 subtests passed**，耗时 24.40 秒。12 项跳过由既有条件触发，包括 Linux `/proc` 计时测试、CUDA 服务测试，以及需要指定游戏资源和冻结参考构建的测试。

C++ 计时直接调用本 worktree 已编译的 `EnvironmentRollout::Encode + Heads(0)`，包含主干、汇总和三个输出头，不含特征编码及进程通信。固定同一真实局面：seed 1400512 第 420 帧、tick 23280，共 10 个非零输入通道。两组均宽 64、meanmax、obs v2、随机种子 17，Release/BLAS、关闭乘加融合、`VECLIB_MAXIMUM_THREADS=1`，各预热 100 次后逐次计时 10000 次并取中位数。

| 结构 | 单步前向中位数 |
|---|---|
| 默认 | **71.875 µs** |
| row_pool | **90.416 µs** |

本次按行汇总增加 18.541 µs，约 25.80%。
