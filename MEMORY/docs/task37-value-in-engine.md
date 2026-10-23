# task37（主控 2026-10-23 08:20 下发给台式机 Codex 第二会话 codex2）：价值网络前向与静态候选打分下沉到引擎（C++ 单进程采集的第一块）

先读 `AGENTS.md`、`MEMORY/docs/hotpath-audit.md`（task33 体检，尤其 §1–§2、§7 第 3 / 4 项、§8）、`MEMORY/docs/task35-engine-cpu.md`（Codex 主会话在做引擎 CPU，分工见下）、
`python/pvz_value_net.py`（`ValueNet`、`featurize`、编码版本——task30 新加了 v4 编码）、`scripts/search_play.py` 的 `_values`（76 行起）、`_leaf_values`（371 起）、`_static_frames`（422 起）、`_search`（433 起）、`_verified_choice`（652 起）。

## 背景（task33 实测，一局搜索 117 s，1240 决策 / 253 次搜索）
- 引擎内仿真 77%、引擎内学生前向 9%；**Python 搜索簿记 6–12%**：其中静态候选 `_static_frames` 4.8 s（每次搜索把 40–240 个合法格逐个 `branch_snapshot`，每个分支回一个完整 observation JSON，Python 再 `_frame` + `featurize` + 价值网前向）、价值网前向 `_values` 2.7 s、叶子打分 0.6 s。
- 目标形态（主控已定）：每个采集 worker 是一个 C++ 进程，整局决策循环在引擎里跑完。本任务是第一块：**“种下去 → 价值网打分”整段在引擎内完成，Python 只拿回分数**。
- 进程间通信本身只占 1–3%，所以本任务的验收看三件事：Python 侧这 8 s/局 是否消失、每个 worker 内存（不再需要 Python 侧价值网）、结果一致。

## 要做的（工作树 `git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/vcpp -b vcpp origin/pvz-env`，单独编译）
1. **C++ 价值网前向**：新文件 `src/EnvironmentValueNet.{h,cpp}`，读一个导出的权重文件（新格式 `PVZVN001`：头部记 width、编码版本、context 维度，正文按层 float32；导出脚本 `scripts/export_value_net.py`）。
   特征构造 `featurize`（棋盘通道 + 标量，含 v4 编码与 context）在 C++ 里重写一份，**直接读引擎内的 Board**，不经过 observation JSON。
   卷积用 im2col + `cblas_sgemm`（与 task35 学生前向同一套做法，但写在自己的文件里，**不要改 `src/EnvironmentRollout.cpp`**——那是 task35 的地盘；共用的小工具放 `src/EnvironmentMath.h`，在“问题”里告诉主控以便协调）。
   验收：同一批局面（从 frames_v2 取白天 / 夜间 / 泳池 / 原版传送带各 256 个），C++ 与 Python 的 logit 最大误差 < 1e-5；v3（旧编码）与 v4（新编码）都要过。
2. **新环境命令** `STATIC_SCORES <root> <动作列表>`：在引擎内对每个候选做“从快照分叉 → 种下 → 推进到静态打分时刻（与 `_static_frames` 现行语义完全一致，读源码确认推进多少 tick）→ 价值网打分”，一次返回所有分数；
   以及 `LEAF_VALUES`（推演叶子局面直接在引擎内打分，替代 `_leaf_values` 的 Python 前向）。`python/pvz_env.py` 加对应方法。
3. **`search_play.py` 改用新命令**：`_static_frames` + `_values` 的组合、`_leaf_values` 改为调用引擎；Python 侧的价值网前向删除（不加开关；旧实现删掉，测试里保留一个用 Python 价值网对拍 C++ 的单元测试）。
   `_init_worker` 不再加载 torch 价值网（学生前向若仍在 Python 侧用于执行步，保持现状，下一块再搬）。
4. **验收**：
   - 逐帧：同一动作序列重放状态相同（引擎逻辑未变）。
   - 老师：同种子、同参数（照 `~/pvz-logs/queue103.sh` 的 `run()`，beta=1）在白天 L7 1.5 倍、夜间 2-4、泳池 L21 各 32 局：老师决策与改前一致率 ≥ 老师自身噪声底线（同老师换随机未来约 75%；先用改前代码测出本机底线），胜局在噪声内。
   - 速度与内存：`scripts/hotpath_profile.py`（task33 的工具）同一局改前 / 改后的分项耗时；每 worker RSS 改前 / 改后。
5. 报告写 `MEMORY/docs/value-in-engine.md`。

## 规则
- 同时最多 4 个你自己的游戏进程；不动 tmux 队列（qc）与别的代理会话；不改别的工作树与主工作区。不重启 WSL。
- 分支 `vcpp` 提交，可以 push；不推 pvz-env、不合并。**不加开关**（重构直接替换，A/B 用上一个提交）；不写手工规则；不写字面密钥；不 `git add -f`；大文件不进 git。只用绝对路径。
- 每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续。持续压榨：做完上面几项后，用 profile 找 Python 侧剩下的最大项继续，连续 3 项收益 < 2% 才停。

## 结果
- 2026-10-23（覆盖审计）：确认四场景各256帧现有测试是Python特征→C++前向，直接Board编码只覆盖128个早期局面，不能把两者合称全局面端到端验收。新增check_value_board_replay.py：按原固定corpus档案动作重放，先证明输入与档案相同，再比较Board特征逐位一致及v3/v4 logit<1e-5；3项档案语义测试通过，真实校验待现有96局改前阶段结束后串行执行。正式改前首个白天/夜间局均完成且获胜，进程仍正常，未重启或复制游戏任务。
- 2026-10-23（恢复第三步）：全套最终314项完成（304通过、10跳过；不伪造缺失的旧稀疏参考），新增门禁测试包含在内。Python剩余三项离线原型：候选键tuple、复用学生特征、复用学生encode，在真实帧/239搜索次数折算下的乐观CPU节省分别0.0473/0.0474/0.1407s（0.0217%/0.0217%/0.0644%），不采纳、不冒充整局A/B；待正式计时复核停止条件。正式96局改前阶段已以2个worker启动，输入锁定，完成后继续改后/必要影子/汇总；task37保持未完成。
- 2026-10-23（恢复第二步）：修复无profile验收在pstats崩溃、原子提交结果并锁定输入SHA256，新增同root/噪声/胜负/CPU汇总门禁（5测试通过）；补入origin的task33 hotpath_profile原工具并在同一worker复用，双端真实命令冒烟通过。v3/v4训练权重叶值/完整状态对拍重过。压力seed5712288的1256决策/239搜索根与动作全同、两端胜；扣除30次噪声搜索的CPU后总CPU231.85→218.33s（-5.83%，仅pilot），噪声27/30。正式三任务各32局改前/改后及必要同root影子验收接续执行，未宣布最终通过。
- 2026-10-23（重启后恢复）：核查 fe2a5aff 及未提交改动；三模型真实帧与1028帧重放结论完整。接回 search 的 SNAPSHOT_FAST / RESTORE_FAST，保留包装层时钟和收入历史，增加真实Board恢复测试；全套309项中298通过、10跳过，唯一失败是旧稀疏能力测试把新引擎同时当成旧参考（环境配置，不是价值路径）。老师旧96局全部在pstats处失败、无result，已作废，修复后重跑；尚未通过老师/CPU最终门槛。
- 2026-10-23：首层im2col仅保留非零通道（保留顺序），900候选首层0.144→0.022s、总前向0.311→0.258s；真实v3/branch1/v4门禁重过（最大3.82e-6）。四场景各257帧动作序列在基线/新引擎的完整公开状态逐帧相同。v4推演与Board oracle两项通过。全套308测试只余fast测试无价值网路径问题，已改按需加载，相关3测试通过；最终全套待重跑。首局195搜索决策逐点相同，改前随机未来噪声pilot25/25相同，压力局噪声与对照在跑。
- 2026-10-23：真实 frames_v2 四场景各256帧 × v3/branch1/v4 全部通过 logit<1e-5（最大3.82e-6）；新增可复跑 corpus 校验与单worker CPU profile 工具。5项既有推演测试通过。价值网加无开关分项计时：900候选初始局面，特征0.0012s、前向0.311s，卷积约46%/38%/12%；下一步先压缩首层零通道。
- 2026-10-23：搜索路径改用 STATIC_SCORES / ROLLOUT_VALUE；driver 只导出一次权重，worker 不持有 torch 价值网；删除 `_values` / `_static_frames` 并迁移诊断调用。32 项搜索测试通过；训练 v3 四场景、mask0/61 的叶值误差 <1e-5 且完整叶状态相同。首局 seed1800000 的997决策/195搜索/6覆盖/胜负/种植统计与基线相同。发现朴素前向性能回退（worker CPU16.46→3.74s，引擎100.58→174.57s），下一步先定位并消除回退，未验收为加速。
- 2026-10-23：完成 PVZVN001 导出、独立 C++ 价值网络（im2col+CBLAS）、直接 Board 编码及 LOAD_VALUE_NET / STATIC_SCORES / LEAF_VALUES 和 Python 包装。双编码 × 四场景 × 16 时点 oracle 通过：特征逐位相同，随机 width64 网络最大 logit 差 7.48e-9、静态概率差 6.01e-8；拒绝动作保持空值、快照恢复一致。真实档案各 256 帧与训练模型门禁仍待执行。
- 2026-10-23：补齐 task30 已有的版本化价值编码（来源 14713bd4，仅取编码与测试）；学生冻结 v1 编码不变，v3 对应编码 1、v4 对应编码 2。开始独立 Release 构建，后续 A/B 固定本分支基线 c790588d。

## 问题
- 已解决：最初缺少task33报告/工具；恢复时已读origin/pvz-env的hotpath-audit全文，并在b306875a补入原版hotpath_profile，正式三任务同工具对照正在执行。
- 已解决：最初缺少task30的v4编码，已从14713bd4移入必要依赖，不另造编码。
- 源码确认：EnvironmentPlant / EnvironmentShovel 本身推进 0 tick，STATIC_SCORES 不额外等待。旧编码须保留 SEED_NONE=-1 写入 scalar[4]/[58] 的历史行为；新编码跳过空槽并按植物身份聚合重复卡。
- 覆盖审计补项：全量真实帧前向门禁须与直接Board编码区分；1024个固定档案采样点的端到端Board重放校验已准备，尚未运行，不记为通过。

## 主控追加（2026-10-23 09:50）：总 CPU 硬门槛
- task33 的 cb-audit（audit 分支 `scripts/hotpath_branch.py`、`hotpath-audit.md` §17 第 10 项）实测：torch 批 18 的价值网前向 93–107 µs/分支，单线程朴素 C++ 卷积比 torch（MKL 多线程）慢 3–6 倍；你自己也测到引擎 100.58→174.57 s。
- 验收加一条硬门槛：同一组种子，**单局总 CPU（worker + 引擎，含所有线程）不得比基线 c790588d 多 2% 以上**（采集是 16 进程并行，CPU 是瓶颈，墙钟单局快不算数）。
- 压不到就把 C++ 前向改成批量 GEMM：一次 STATIC_SCORES / LEAF_VALUES 的全部候选拼成一个矩阵，调 OpenBLAS/MKL sgemm（单进程线程数 ≤ 2）；卷积 im2col 只做一次。
- 仍做不到就如实写进“问题”，附两边的 CPU 数字；主控决定是否保留 Python 侧。不要为了过门槛去改搜索的参数或候选数。
- 合并提示（主控 10:20）：pvz-env 已合并 audit（b95d2cf7）——整条 CUDA 推演线退役（tag `retire-cuda-20261023`），`python/test_cuda_*.py`、`test_candidate_coverage.py` 等已删除。你合回 pvz-env 前先 `git merge origin/pvz-env`，对这些已删文件的改动直接丢弃；`--rollout-backend` 等开关已不存在。
- 主控追加（10:10）：task42 在复验“老师候选改用学生前 3、不用静态网”（B3，10-20 已非劣通过、省 29% 搜索 CPU）。若通过，静态候选打分（`STATIC_SCORES`）整条会删掉。所以**先把 `LEAF_VALUES`（叶子价值）做完并过总 CPU 门槛**——它还是 task40“推演胜负已定就收手”的前提；`STATIC_SCORES` 已做的部分保留在分支上，不再继续投入。
