# task33（主控 2026-10-22 23:20 下发给 Mac 上的 CodeBuddy 会话 cb-audit）：主干路径全面体检与重构计划

先读仓库根目录 `AGENTS.md`（优先级最高）、`README.md` 末尾的导航表、DESIGN §13–§14、`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22 的几条、
`MEMORY/docs/task32-memory.md`（另一个会话 cb-mem 正在做内存，别和它改同一处，结果互相引用）。

## 背景
用户的判断：代码“屎山越垒越高”。实例：dw5i 训练每个 batch 随机 gather 一个 2.4 GB 的内存映射文件，页缓存一被挤掉就整盘狂读（863 MB/s，GPU 0%），
训练只有 8% 时间在跑，还磨硬盘；台式机 WSL 被内存挤到连不上，只能重启。仓库现有 107 个 scripts、65 个 python 模块，很多是一次性实验脚本层层叠加。
要求：**全面扫描、分析，对必要路径（每天都在跑的主干）重构**，不是到处修补。

## 主干路径（只管这些；一次性实验脚本不在范围内，只需列出“已无人调用”的清单）
1. **采集**：课程控制器 `scripts/curriculum_queue.py` → `scripts/search_play.py`（worker 池、学生前向、搜索 / 推演、价值网络、帧写出）→ `python/pvz_env.py` / `pvz_event_env.py`（进程间协议）→ 引擎 `src/main.cpp`、`src/EnvironmentRollout.cpp`、`LawnApp::Environment*`。
2. **存储**：帧档案格式与目录布局（`artifacts/student/frames_v2/seed_*/<label>_fast_search/`、`pvz_frame_store.py`、npz / jsonl）、HF 传输、Mac→台式机导入。
3. **训练**：`scripts/train_rollout_policy.py` + `python/pvz_frame_store.py` / `pvz_training_rows.py` / `pvz_rollout_policy.py` / `pvz_student_observation.py`（特征缓存、CSR、memmap、`packet_info`、软目标、验证集）；价值网络 `train_value_net.py` / `train_value_branches.py`。
4. **评估**：`scripts/unified_panel.py`（含压力面板）、`eval_panel.py`。

## 要做的
1. **体检（先不改代码）**：对每条主干路径画出数据流（谁产出什么文件、多大、谁读、读几遍、随机还是顺序），并实测（Mac 上小规模即可，≤2 个你自己的游戏进程）：
   - 每局 / 每帧 / 每行的耗时与字节（CPU、内存、磁盘读写、文件数）；进程间协议每步往返次数与 JSON 体积；搜索每个决策的推演次数与引擎调用；
   - 冗余：同一数据存几份、同一特征算几遍、重复 import / 重复加载模型、每局起停进程的成本；
   - 小文件数量（frames_v2 下有多少文件、平均多大）、磁盘写放大；
   - 死代码：主干模块里已无调用的函数 / 参数 / 开关（`git grep` 证明），scripts 下已无引用的一次性脚本清单（只列，不删）。
2. **重构计划**：`MEMORY/docs/hotpath-audit.md`——每个问题一行：现状数字、病根、改法、预计收益（时间 / 内存 / 磁盘）、改动面、风险、验收方式；按收益/代价排序。
   明显的大头（例如训练数据整体改成一次性顺序读进紧凑的常驻数组、帧档案合并成每轮少量大文件、worker 常驻复用而不是每局起停）要写清目标形态。
3. **先做排在最前的 2–3 项**（与 cb-mem 不重叠的），各自一个提交系列。验收门槛（用户定的）：
   - 引擎逻辑：同一动作序列重放，状态逐帧相同（硬门槛）；
   - 训练 / 采集结果：不要求逐位相同，同种子同配方下验证损失、面板胜率、到达波数在种子噪声内即可（给对照数字），收益要足够大；
   - 旧档案仍能读（或给一次性转换脚本，转换前后逐行核对）。
   默认值的切换由主控决定：新路径先做成开关，报告里给出建议。

## 规则
- 不动 Mac 上正在跑的采集（`mac_dagger_*`、`search_play`）与别的会话（cb-mem、cb-ops）；同时最多 2 个你自己的游戏进程；不改主工作区与别的工作树。
- 工作树：主控已建 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/audit`（分支 `audit`，基于 pvz-env）。分支 `audit` 提交，可以 push；不推 pvz-env、不合并。
- 不删文件（只列清单，删除按 AGENTS §4 由主控做）；不写手工规则；不写字面密钥；不 `git add -f`；大文件不进 git。只用绝对路径。
- 每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续。

## 结果

## 问题
