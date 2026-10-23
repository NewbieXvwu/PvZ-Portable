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

- 00:15 体检工具落地：`scripts/hotpath_bench.py`（引擎启停 / step / rollout 往返、档案读）、`scripts/hotpath_profile.py`（包住 `search_play` 的热函数跑真局，逐项拆时与字节，不改产品代码）。
- 00:40 报告落地：`MEMORY/docs/hotpath-audit.md`（一局的账、分布前 15 项、存储账、冗余、死代码、开关清单与删除候选、重构计划、C++ 整局化依据与要搬的 674 行）。开关工具 `scripts/hotpath_flags.py`（201 份运行配置 + 全文引用，11 个从未被设过、15 个恒定值）、死代码 `scripts/hotpath_deadcode.py`、脚本清单 `scripts/hotpath_scripts.py`（16 个 0 引用、2 111 行）。
- 00:40 **训练装载实测**：`--cache-dir` + 8 预计算进程 48 598 行/s（1.28 M 行 32 s），13.6 M 行约 280 s；不带 cache-dir 慢 5.4 倍（9 037 行/s，为筛行把每局原始 JSON 重解一遍）。档案 21 056 个、5.07 GB、25 M 行、288 B/行落盘、13.8 KB/行解压后。
- 00:15 **一局的账（day_l7_z17 种子 1740001，搜索局 117.4 s / 1240 决策 / 253 次搜索 / 3962 条推演）**：引擎仿真 **90.2 s（77%）**、引擎内学生前向 **10.2 s（8.7%，146 531 次 → 70 µs/次）**、Python 搜索簿记 **6.7 s（5.6%）**、进程间往返 + JSON **≈2.5 s（2.1%）**、Python 学生前向 0.46 s（0.37 ms/决策）、写档案 0.15 s。同种子纯学生局 2.1 s（1425 决策，1.5 ms/决策）。**结论：IPC 只占 2%，C++ 整局化的收益在内存与架构，不在速度。**

- 01:05 **第 1 项已做**（worker 常驻）：8 局纯学生 A/B（上一个提交在 `/tmp/audit-old` 工作树）22.64 → 15.64 s（−31%，−0.88 s/局）；同种子 8 局 `games.jsonl` 的 won/final_wave/decisions/mowers/planted/searched **逐字段相同**。worker RSS 16 局内 31 → 194 MB 峰值后 163–194 来回，无单调增长。
- 01:15 **兼容与实验开关统一登记**：`python/pvz_compat.py`（8 条 LEGACY + 8 个实验开关，决定日 2026-11-06）、`LEGACY(...)`/`EXPERIMENT(...)` 标记、help 前缀、`python/test_compat_registry.py` 双向校验 5 项全过。
- 01:20 观测载荷实测：完整观测 14 280 B，其中 `legal_actions` 45% + `cells` 37%；引擎已有的 `OBS_LEAN` 在同一 tick 是 **51%**，且学生特征逐位相同（棋盘 max abs diff 0.0）。缺 `zombie_count_multiplier` / `loadout_context` 两项（价值网上下文要用）。**没做**：要重编 `build/pvz-portable`，与正在跑的 13 个采集共用二进制，留给 task35 或停采窗口。
- 01:25 训练装载打包原型 `scripts/pack_training_rows.py`：把 `load()` 的结果按档案顺序写成单个 904 MB 的 CSR 文件（1.56 M 行 = 581 B/行），读回一遍即可，不再解压 345 GB、不再重算 `np.nonzero`。

- 01:40 打包原型验收通过：`--verify` 与 `load()` 逐项对比 13 个数组全部相同；读回 1.56 M 行 0.17 s（mmap）对 164 s（无 cache-dir）/ 32 s（有 cache-dir）。旧 JSONL 档案实测 12.70 MB → 1.04 MB（8%），全量 7.54 GB → 约 0.6 GB（`scripts/pack_frames.py` 可用，未动数据）。
- 01:45 面板口径 A/B 修正并重测（改前提交 `4a0dba12`，`/tmp/audit-old3`）：16 局 31.18 → 16.11 s（−48%），16 局结果逐字段相同。
- 01:50 全量测试：272 passed / 31 skipped / 375 subtests passed，6 failed —— 5 个是 `test_cpp_rollout` 在**工作树里没有 `build/`**（不是代码问题），1 个是 `test_dagger_experiment_runner` 的导入顺序依赖（单独跑连模块都导入不了），均与本次改动无关。
- 02:00 旧档案转换逐帧核对：3001 帧 JSONL → npz → 读回，**每帧 dict 相同**（3001/3001）。启动账：每次 `search_play` 进程 1.2 s（torch 导入 0.46 s + 三个网 + 任务），每个 worker 引擎启动约 1.3 s；`export_student` 实测 0 ms、`load_student` 2 ms，所以没有可缓存的东西。
- 01:55 压榨记录见 `MEMORY/docs/hotpath-audit.md` §13：6 项尝试、收益与阻断原因；停止条件**未满足**（第 2/3 项仍是可证明的大额冗余，但剩余工作量在 cb-mem 的 `load()` 与主工作区数据上）。

- 09:10 合并完成：`origin/pvz-env`（task34 OOM 守护）与 `origin/mem`（task32 紧凑数组装载）都进了 `audit`，冲突两处——task32 文档保留 pvz-env 侧、`train_rollout_policy.py` 取 mem 侧（稠密棋盘装载随 task32 删除，`dense-boards` 的 LEGACY 登记一并移除）。281 passed / 34 skipped / 6 failed（环境性）。
- 10:20 **删除批准全部执行**（tag `retire-cuda-20261023` / `retire-flags-20261023`）：CUDA 推演后端整体退役（`cuda/`、`src/CudaExport.*`、`CMake/CudaRollout.cmake`、`pvz_cuda_*.py`、11 个 cuda 脚本、10 个 cuda 测试、`mixed_dagger_round.sh`、`scripts/task6/`），`search_play` 的 13 个开关与分支、`train_rollout_policy` 的 `--ema` / `--soft-weight`、候选覆盖与静态价值消融两条实验线。**101 个文件、-24 165 行**。`--margin`→`MARGIN=0.02`、`--max-actions`→`MAX_ACTIONS=3000` 改常量；`--resource-dir` / `--width` / `--soft-temperature` / jsonl 读取路径按批复保留。229 passed / 6 failed（环境性）。细节见 `hotpath-audit.md` §14。

- 11:05 **第 3 项（打包装载转正）完成**：`train_rollout_policy.load()` 只读包，`pack_training_rows.py` 成为唯一读档案的地方，`curriculum_queue` 每轮结束打包并用 `--prune` 删掉被取代的旧包。验收：同一批 88 692 行，前 200 步 6 个 epoch 的 train/val loss 与 tag `retire-flags-20261023` **逐位相同**（差 0，不是 1e-6）；装载 8.10–8.23 s / 304–305 MB → **0.009 s / 227 MB**，包 262 B/行。231 passed / 6 failed（环境性）。细节见 `hotpath-audit.md` §15。
## 问题

- **包的键包含 `cells`，而它跟着 `--cell-soft-weight` 走**（`load()` 里 `cells=bool(args.cell_soft_weight)`）。生产配方带 `--cell-soft-weight 3`，所以采集端起 `curriculum_queue` 时必须给 `--pack-cells`，其余行选项（`--pack-wait-keep 0.1 --pack-obs v2 --pack-packet-head identity`）也要与训练命令一致；不一致时训练端会打印该跑的打包命令，不会静默用错包。

## 追加（主控 2026-10-22 23:55，用户两点 + 主控决定）
1. **采集去掉 Python ↔ 引擎的进程间通信**（用户提议“干掉它、重写成 C++”，主控同意方向）。目标形态：**每个采集 worker 是一个 C++ 进程**，整局决策循环（学生前向——已有 C++ PVZRN006；
   搜索候选、推演——已在引擎内；价值网络前向；帧档案写出）都在引擎进程里完成，Python 只负责起 N 个进程、汇总结果、训练。不再有每步一行 JSON 的协议、不再每个 worker 带一份 Python + torch。
   本任务先把它写进重构计划并量出依据：一个决策里时间花在哪（引擎推演 / 进程间往返与 JSON 编解码 / Python 特征与前向 / 写盘）、每个 worker 的内存（Python+torch vs 引擎），
   以及 `search_play.py` 的搜索逻辑哪些部分要搬（列函数清单与行数）、价值网络是否已有 C++ 前向。验收口径（用户已定）：引擎逻辑逐帧相同；老师决策与 Python 版的一致率不低于老师自身的噪声底线（换随机未来约 75%）、同种子胜率与到达波数在噪声内。
   **搬完后 Python 版搜索删掉**，不留两份实现。若量出来进程间通信只占很小一部分，也照写，由主控据数字定先做什么。
2. **不再加开关**（推翻任务书第 3 步“新路径先做成开关”）：现在 scripts 里有 732 个命令行参数（search_play 40、train_rollout_policy 34、episode_query 41、unified_panel 23、curriculum_queue 18）。
   规则：**重构不加开关**——新实现直接替换旧实现，A/B 用上一个提交在另一个工作树里对比，验收通过后旧路径在同一系列提交里删除。只有确属实验、可能撤回的行为才允许加开关，且必须登记到 `MEMORY/docs/flags.md`（参数、谁加的、日期、到期决定日），到期要么转正（删开关、只留一种行为）要么删掉。
   本任务加一项：给主干四个脚本 + `pvz_env.py` + 引擎 `-env` 参数做开关清单——每个开关在现行队列 / 课程控制器 / 训练配方 / 面板里是否被实际设置过（读 `~/pvz-logs` 的队列脚本只能台式机，Mac 上读仓库里的调用方与 `artifacts/hard_mac/*.sh`），
   从没被设置或永远设成同一个值的，列为“删除候选”（附证据）。删除由主控批。

## 追加（主控 2026-10-23 00:05，用户）：笔记里已判死的实验开关也删
删除候选不只看“现行队列有没有设置过”，还要对照笔记：`DESIGN.md`、`MEMORY/docs/student-bottleneck-plan.md`、`MEMORY/20*.md` 日志里明确写了**“不采用 / 不并入 / 没过 / 判死 / 已删除 / 退役”**的实验，
它们的开关与代码路径已完成历史使命，即使过去设置过也列为删除候选。例：EMA（10-19 不采用）、宽度 128（两次不采用）、视界 1800 / 自适应视野位 64（没过）、软目标全分布（learn/03 崩到 17/256）等——以笔记原文为准，每条给出处（文件:行）。
例外（必须保留，写明理由）：现行面板 / 文档仍在用的旧检查点（如 v34 slot 头、identity_dw3_s1）加载与重放所需的路径；旧档案 `whatif` 重放（AGENTS §8 第 4 条）所需的字段解析；编码器签名冻结相关的旧版本编码。
删除方式（主控批后执行）：整条线先打 tag（AGENTS §4，如 `retire-<名>-20261023`），再删开关 + 分支代码 + 只服务于它的测试，`git grep` 证明无引用，文档里的旧数字保留不动。清单写进 `hotpath-audit.md` 单独一节，按“删掉能减少多少行 / 多少分支”排序。

## 追加（主控 2026-10-23 00:15，用户）：向后兼容与实验开关在代码库里统一登记、统一标记
用户要求：留下的向后兼容字段 / 路径要让人一看就知道是“已弃用、只为旧东西保留”，并在代码库层面统一管理。做法：
1. **一个登记处** `python/pvz_compat.py`（纯数据 + 小工具，无副作用）：
   - `LEGACY`：每条向后兼容一项——`id`、说明、为谁保留（依赖它的检查点 / 档案 / 面板，写具体路径或族名）、引入日期、**移除条件**（例如“v34 不再出现在任何面板与文档对照里”）；
   - `EXPERIMENTAL_FLAGS`：每个允许保留的实验开关——参数名、所在脚本、谁加的、日期、**决定日**（到期必须转正或删除）；
   - 取代此前说的 `MEMORY/docs/flags.md`（文档里只放一句指向它，不另维护一份）。
2. **统一标记**：代码里每个兼容分支 / 字段旁写 `# LEGACY(<id>)`（C++ 写 `// LEGACY(<id>)`），每个实验开关的 `add_argument` 旁写 `# EXPERIMENT(<id>)`；
   argparse 的 help 文字以 `[legacy]` / `[experiment until YYYY-MM-DD]` 开头，`--help` 一眼可见。
3. **测试兜底** `python/test_compat_registry.py`：扫全仓库的标记，标记里的 id 必须在登记处、登记处的 id 必须在代码里出现（防止删了代码忘了登记或反过来）；
   `EXPERIMENTAL_FLAGS` 的决定日已过就让测试失败（逼着到期做决定）；`LEGACY` 每条必须写了“为谁保留”和“移除条件”。
4. 先把本任务体检出的兼容项与保留的开关全部登记、标记一遍（旧观测版本 v1/v2lite、冻结编码器签名、slot 头加载、`planted` 槽号旧键、无 `packets` 旧档案退回 deck、L10 旧变体等），写进 `hotpath-audit.md` 一节。

## 追加（主控 2026-10-23 00:45，用户）：持续压榨，直到没有油水
用户的原话：“得继续给优化代理加压，让它持续优化直到真正完全炸不出油水为止。”本任务不再以“做完列出的几项”为完成：
每做完一项就重测、把当前的耗时 / 内存分布（前 15 项及占比）贴进报告，再挑下一个最大项继续；**停止条件**：连续 3 项尝试的收益都 < 2%，且分布前 15 项里没有可以证明是冗余的部分。没达到就继续，不要自己宣布完成。

## 下一阶段（主控 2026-10-23 08:30 批复与追加）
报告很好。批复与接下来的工作（按顺序做）：
1. **合并**：把 `origin/mem`（task32 已完成）与最新 `origin/pvz-env` 合进 `audit`，解决冲突（训练装载两边都动过：以第 2 项的“打包装载”为入口、task32 的紧凑数组为常驻形态），全量单元测试通过后提交、push。
2. **删除批准**（按 AGENTS §4 先打 tag `retire-<名>-20261023` 再删，`git grep` 证明无引用，文档旧数字不动）：
   - 6.1 全部批准，例外：`--resource-dir` 保留（两台机器路径不同，是运维参数）；`--margin`、`--max-actions` 改成代码常量（带注释说明来源），删参数；`--cpp-rollouts` 的 Python 参考推演下沉到测试专用；
     CUDA 推演后端（`--rollout-backend cuda`、`--cuda-*`、`RolloutCuda.cu` 及只为它服务的代码）整体退役（只支持 slot 头、正式轮次从未启用）；`--frames-format` 删写入分支，读取 jsonl 保留为 `LEGACY`。
   - 6.2 全部批准，例外：`--width` 参数本身保留（模型结构参数，加载 / 定义模型要用），只删“宽度 128 不采用”的相关对照代码（如有）。
   - 删完在 `python/pvz_compat.py` 登记保留下来的 LEGACY 项（6.3）。
3. **第 2 项落地为唯一路径**：采集端每轮结束（或每个标签写完）把档案打包成顺序大文件（含 CSR），训练只读包；没有包的旧档案走一次性转换（`pack_training_rows.py` 变成正式工具，转换前后逐数组相同）。
   不加开关；旧的逐档案装载路径删除（只留转换器里的读旧档案代码，标 `LEGACY`）。验收：同一批数据改前改后训练前 200 步损失一致到 1e-6；装载耗时与峰值内存对照表。
4. 然后按第 3 项（静态候选下沉到引擎）准备设计，但**不要动 `src/EnvironmentRollout.cpp` 的学生前向**（台式机 task35 正在改），需要接口就在“问题”里写清楚交给主控协调。
仍按停止条件持续压榨。每完成一小步提交并 push。
