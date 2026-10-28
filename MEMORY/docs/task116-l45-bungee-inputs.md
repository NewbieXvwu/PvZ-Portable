# task116：L45 蹦极闪电战完整接入（可见蹦极状态进模型输入）

主控 2026-10-27 派给 Mac codex-plus2b。分支 `l45input`，基于 `origin/pvz-env`（4c9ca857 之后，含 deploy14f）。

## 背景

- task110 调查（`MEMORY/docs/task110-minigame-levels-survey.md` §L45）：L45 的动作头不用改（传送带牌 + 种植 / 等待），缺口在**状态表示**——`_frame()` 和 `v2_extra()` 丢掉了蹦极僵尸的目标格、下落阶段 / 高度、被抓植物状态，模型看不见“哪一格将被抓走”。
- task111 已登记 `demo_roof_45_orig_m100`，用户已录一局并通关；task112 / deploy14e 已把它的起点格放进课程（首组 20/24）。
- 用户定规：不写手写规则或公式；只补**玩家屏幕上看得见**的信息（落点靶标、蹦极高度、植物正在被抓），不输入未来牌序、未来出怪等隐藏状态。

## 要做的

1. 在完整 / 精简观测、紧凑帧、Python 学生观测、C++ 推演学生输入、价值网输入里，按现有**输入版本机制**新增一个兼容版本，编码：蹦极目标格、阶段（下落 / 停留 / 上升）或高度、被抓植物标记。旧检查点维度不变，旧版本逐位不变。
2. Python 与 C++ 两端特征一致性：用现有对拍工具在 L45 局面上做一次对拍（小样本即可）。
3. 训练接入：课程控制器 / 打包代码能为新版本产出训练行；不需要在本任务里训练出达标模型。
4. 测试：只跑相关测试（观测、对拍、打包）。按 AGENTS.md §6.5，**不要**整套重跑、不要全机审计、不要加任务书没要求的校验。

## 交付

- 推送分支 `l45input`；报告写在本文件末尾：改了哪些文件、新输入版本号、对拍结果、上线需要主控做什么（学生要用新版本得重训，由主控安排）。
- 不部署生产，不动台式机 qc。

## 实施结果（2026-10-26）

主工作树已执行 `git pull --ff-only`，从 `origin/pvz-env` 的 `49378c3b` 新建工作树与分支 `l45input`，包含 deploy14f。

### 输入版本与可见信息

学生新增 **`v2bungee`**，原生导出观测编号为 **7**，引擎能力号 `rollout_version=5`。它在 `v2boss` 后追加6个棋盘通道，先是可见靶标和身体高度，接着是下降/停留/上升三通道，最后是被抓植物。总棋盘为111×6×10，标量124维，开启任务上下文时为159维。高度取公开身体的整数像素高度，截断到0–600后除以600。

价值网新增 **编码器 v3**（检查点标识 `pvz_value_net.featurize v3`），在价值编码器 v2 后追加相同6通道，总棋盘98×6×10，标量119维，带任务上下文时为154维。旧学生版本与价值编码器 v1/v2 的维度及特征保持原样，历史学生缓存签名保留。

完整/精简观测共用 `bungee` 字段，紧凑帧保留该字段。`zombies` 每行是 `[row,col,target_visible,phase,altitude]`，阶段0/1/2分别为下降/停留/上升，身体不可见时阶段为−1且高度为0。`plants` 只列出可见的被抓植物格。

靶标只在原版移动动画已落定时公开最终格，沿用原绘制函数的限制，运送关联僵尸及上升期间不提供靶标。身体矩形顶部进入屏幕后才公开身体高度和阶段。雾与暴风雨继续服从现有遮蔽条件，没有加入未来发牌、随机数或内部阶段倒计时。

### 修改文件

公开观测与帧记录：`src/LawnApp.cpp`、`src/LawnApp.h`、`src/Lawn/Zombie.cpp`、`src/Lawn/Zombie.h`，以及 `scripts/render_episode.py`。可见性由统一入口计算，观测与原生特征共用。

双端编码与模型协议：`python/pvz_student_observation.py`、`python/pvz_value_net.py`、`python/pvz_rollout_policy.py`、`python/pvz_env.py`，以及 `src/EnvironmentRollout.cpp/.h`、`src/EnvironmentValueNet.cpp/.h`、`src/main.cpp`。原生价值特征文件读取按模型实际通道数处理。新学生的槽位头空牌输入与冻结的 Python 特征对齐，旧版本原生路径保持原样。

训练与缓存：`python/pvz_recipe.py`、`python/pvz_frame_store.py`、`scripts/pack_training_rows.py`、`scripts/train_value_net.py`。课程控制器使用的训练配方接受新版本，并通过共用行选项传入打包器，实际生成的训练行可直接输入新学生网络。价值训练器可用 `--encoder-version 3` 产出新架构。

相关测试集中在 `python/test_bungee_inputs.py`，当日日志写入 `MEMORY/2026-10-26.md`。

### 对拍结果

7项离线测试通过，覆盖旧版本逐位保持、新通道语义以及训练配方到打包张量。L45实机测试复用 `rollout_predict` 与 `leaf_values` 的原生特征轨迹接口，单进程取5个局面，覆盖全部6个新增通道。完整与精简观测转换后的帧相同，学生与价值网的 Python/C++ 输入均逐 float32 位一致。学生种植输出及价值预测的最大绝对误差为 **1.30844×10⁻⁸**，原生价值特征文件评分同样通过。

```bash
PYTHONPATH=python:scripts PVZ_NO_POPUP=1 python3 -m unittest \
  python.test_bungee_inputs \
  python.test_value_encoder \
  python.test_student_observation.ObservationEncodingTest
```

本机引擎已编译完成，测试使用本工作树 `build/pvz-portable`。测试不执行全量胜率测量。

### 上线需要主控安排

主控合并后重建目标机器引擎，将新训练配方的 `obs` 设为 `v2bungee`，重新训练学生。价值网需要使用新状态时，以 `--encoder-version 3` 重新训练并导出。生产配置与台式机 qc 保持现状。

旧压缩档案缺少 `bungee` 字段时追加通道为零。已录的 L45 示范须用新引擎重新生成包含该字段的帧后再打包，才能学习这批可见信息。本任务完成输入与训练行接通，未进行生产部署或模型达标训练。
