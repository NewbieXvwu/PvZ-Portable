# 主干路径体检与重构计划（task33，2026-10-23）

范围：采集 / 存储 / 训练 / 评估四条主干。一次性实验脚本只在 §6.5 列清单。
所有数字都是本机（Mac，48 GB，M3 Max）实测，工具在 `scripts/hotpath_*.py`，命令见各节。

## 1. 一局的账（采集）

同一关（`day_l7_z17`，倍率 1.7），同种子，1 个 worker，引擎 `build/pvz-portable`，
学生 v32 + 价值网 value_v3 + 分叉网 value_branch1_desk。测法：`scripts/hotpath_profile.py`
把 `search_play` 的热函数包一层计时后跑真局，不改产品代码。

| 阶段 | 搜索局（117.4 s，1240 决策，253 次搜索，3962 条推演） | 纯学生局（2.29 s，1384 决策） |
|---|---|---|
| 引擎内仿真 | **90.2 s（77%）** | 计入 step |
| 引擎内学生前向（C++ PVZRN006） | **10.2 s（8.7%）**，146 531 次 → 70 µs/次 | — |
| 进程间往返 + JSON 编解码 | 1–3 s（1–3%），139 MB 响应 / 8 062 次往返 | 计入 step（26.6 MB / 1387 次） |
| Python 搜索簿记（`_search` 减去推演） | 6.7–14.9 s（6–12%） | — |
| ├ 静态候选（`_static_frames`，472 次 branch 命令 / 19 556 个分支） | 4.8 s（4.9%） | — |
| ├ 价值网前向（`_values`，936 次调用） | 2.7 s（2.8%） | — |
| ├ 复核（`_verified_choice`，含 9 次额外推演） | 1.9–4.0 s（2–3%） | — |
| ├ 叶子打分（`_leaf_values`） | 0.6 s（0.7%） | — |
| 引擎启动 + reset（**每局一次**） | 1.3 s（1.1%） | **0.79 s（34%）** |
| Python 学生前向（`_fast_action`） | 0.43 s（0.4%，0.37 ms/决策） | 0.20 s（9%） |
| 帧序列化 + 写档案 | 0.18 s | 0.15 s（5%） |
| env.step（1400 次决策） | 1.2 s | 1.07 s（46%） |

两局合计：`hotpath_profile.py` 输出一行 JSON，见 §9 的提交记录。

**结论一**：搜索局的钱几乎全在引擎里（86%），其中 77% 是仿真本身。进程间通信只占 1–3%。
把整局搬进 C++ 的收益上限是 **13%（Python 侧全部）**，实际能拿到 8–10%（算法本身仍要花时间）。
它真正的价值是内存（§3）和不再维护两套实现，不是速度。

**结论二**：纯学生局的钱一半在引擎启停与进程往返，不在算力。这一档是面板与评估的全部内容。

## 2. 分布前 15 项（搜索局，按耗时排序）

| # | 项 | 秒 | 占比 | 谁负责 |
|---|---|---|---|---|
| 1 | 引擎仿真（9.8 M tick / 局） | 74.9–90.2 | 77% | task35（引擎 CPU） |
| 2 | 引擎内学生前向 | 8.8–10.2 | 9% | task35（C++ 推演） |
| 3 | 静态候选 branch 往返 | 4.8 | 5% | 本任务 |
| 4 | 价值网前向 | 2.7 | 3% | 本任务 / cb-mem |
| 5 | 复核推演 | 1.9–4.0 | 2–3% | 算法 |
| 6 | 引擎启停 + reset | 1.3 | 1.1% | 本任务 |
| 7 | env.step 往返 | 1.2 | 1.2% | 本任务 |
| 8 | 叶子打分 | 0.6 | 0.7% | 本任务 |
| 9 | Python 学生前向 | 0.43 | 0.4% | 本任务 |
| 10 | featurize（17 259 次，24 µs/次） | 0.41 | 0.4% | 本任务 |
| 11 | `_frame`（19 556 次，13 µs/次） | 0.25 | 0.3% | 本任务 |
| 12 | 写档案 | 0.13 | 0.1% | 本任务 |
| 13 | 帧 JSON 序列化 | 0.05 | 0.05% | 本任务 |

纯学生局（面板口径，2.29 s）：

| # | 项 | 秒 | 占比 |
|---|---|---|---|
| 1 | 引擎启停 + reset（每局一次） | 0.79 | 34% |
| 2 | env.step 往返（1384 次，19 KB 响应） | 1.07 | 46% |
| 3 | Python 学生前向 | 0.20 | 9% |
| 4 | 写档案 | 0.15 | 5% |
| 5 | 帧 JSON | 0.03 | 1% |

## 3. 存储账

| 目录 | 文件数 | 大小 | 每文件 | 谁写 | 谁读 |
|---|---|---|---|---|---|
| `artifacts/hard_mac/frames/seed_N/<label>_fast_search/` | 21 056 个 `frames.npz` | 5.07 GB | 均值 235 KB（中位 240 KB，最大 0.5 MB） | `search_play` 每局一个 | 每次训练整目录读一遍 |
| `artifacts/value_net/episodes*` | 8 208 | 4.4 GB | — | 价值网络采集 | 价值网络训练 |
| `artifacts/student/frames*` | 4 096 | 2.2 GB | — | 旧采集 | 旧训练 |

单档案内容（一局 ~1 300 行）：`frames`（原始 JSON 字节，**无损源码副本**）+ `boards`（92×6×10 float16）+ `scalars`（115 float32）
+ `plant/packet/cell/values`。一行 **13.8 KB 解压后 / 288 B 落盘**（zlib 压缩 48:1，因为棋盘 96% 是零）。

- 全量一遍 = 25 M 行 = **345 GB 解压后**，单核 3.3 GB/s → **105 s CPU**，磁盘 5 GB。
- 训练装载实测（`--cache-dir` + 8 个预计算进程，两个标签 1.28 M 行）：**48 598 行/s**，32 s；13.6 M 行 → **约 280 s**，峰值 RSS 786 MB。
- 不带 `--cache-dir` 时慢 5.4 倍（9 037 行/s）：那条路为了决定保留哪些行，把每局的原始 JSON 重新 `decode().splitlines()` 一遍。
- 常驻形态：CSR **364 B/行（boards）+ 97 B/行（scalars）= 461 B/行**，13.6 M 行 = 6.3 GB（cb-mem 的口径更小，是因为 `--wait-keep 0.1` 只留 10% 的等待行）。

**没有冗余副本的地方**：`cache_episode` 在 `obs=v1` 且编码器签名匹配时直接返回原档案，不复制。
**有冗余的地方**：`obs=v2/v2lite` 会在 `cache_dir/v2/` 再写一份带额外通道的副本（每局一个 npz），
以及档案里同时存了原始 JSON（占解压后载荷 18%）与由它算出的全部特征。

## 4. 冗余清单

1. **同一局的原始 JSON 与特征各存一份**（每局 ~2.7 MB JSON / ~15 MB 特征）。JSON 是重放与 `whatif` 的唯一来源，不能删，但它压完只占 18%。
2. **每次训练重算 CSR**：档案里是稠密棋盘，训练器每次 `np.nonzero` 压成 CSR（13 µs/行，25 M 行 = 5.4 min CPU），结果丢掉不存。
3. **每次训练重解压**：25 M 行 × 13.8 KB = 345 GB，每轮一遍，只为了取出 461 B/行。
4. **每局起停进程**：fork + `_init_worker` + 引擎启动 + `load_rollout_net` 共 **1.62 s/局**（§7 第 1 项），纯学生局占 59%。
5. **v2/v2lite 特征缓存副本**（§3）。
6. **每局一次 `load_training_task`**（读 manifest、建任务）在 `_play` 里，可以按 (task_id, seed 段) 缓存，但实测 < 1 ms，忽略。

## 5. 死代码

`scripts/hotpath_deadcode.py`：统计每个模块顶层定义在全仓库的出现次数（排除定义本身）。

- `python/pvz_frame_store.py:167 encode_v2lite_frames` —— **0 次引用**，纯死代码（`encode_student_frames` 已参数化）。
- 其余"≤2 次"的都是真在用的内部函数（`_save`、`encode_frames_from_source` 等），不是死代码。
- `scripts/curriculum_queue.py` 18 个定义里 14 个只在本文件出现，属于正常（纯函数 + 测试）。

结论：主干模块里**没有大块死代码**，屎山感来自"参数与分支"而不是"没人调的函数"（§6）。

## 6. 开关清单与删除候选

工具：`scripts/hotpath_flags.py`。三路证据：argparse 定义、`artifacts/**/config.json`（每次运行把 `vars(args)` 原样落盘，共 201 份）、
`git grep` 全仓库的显式出现。判定：`never-set`（只有默认值）/ `constant`（永远同一个非默认值）/ `used`。

规模：search_play 40 个、train_rollout_policy 34、episode_query 41、unified_panel 23、curriculum_queue 18、pvz_env 0（参数在构造函数里）。

### 6.1 从未被设过（删除候选，证据：201 份运行配置里全是默认值，且无调用方）

| 参数 | 默认 | 证据 |
|---|---|---|
| `--measure-cpu` | False | 113 次运行全 False；只 task18 手工用过 |
| `--candidate-generator` | static | 106 次全 static |
| `--candidate-coverage-model` / `--candidate-coverage-budget` | None / 2 | 106 次全默认；覆盖开关在 plan:307 记为"全部默认关闭" |
| `--shovel-candidates` | 0 | 126 次全 0；plan:151 "铲子先保持关闭"，161 说正确形态是"替换" |
| `--cuda-prefetch-verify` / `--cuda-speculative-halving` | False | 113 次全 False |
| `--cpu-sparse-tracks` | False | 106 次全 False；task6 稀疏轨道"全部默认关闭"（plan:307） |
| `--margin` | 0.02 | 201 次全 0.02 |
| `--rollout-backend` | cpp | 113 次全 cpp（cuda 后端从未在正式轮次启用） |
| `--cuda-socket` / `--cuda-lanes` / `--cuda-gather-ms` | None / 2048 / 80 | 113 次全默认 |
| `--max-actions` | 3000 | 199 次全 3000 |
| `--frames-format` | npz | 151 次全 npz；jsonl 只服务旧档案（保留读取路径，删除写入分支） |
| `--resource-dir` | 环境变量/默认路径 | 201 次全是同一个绝对路径 |
| `--cpp-rollouts` | True | 166 次 True、15 次 None，从没 False；Python 参考实现在测试里（`perf_rollout_check`），应下沉为测试专用 |

### 6.2 笔记里已判死的实验（即使设过也删，附出处）

| 开关 / 代码路径 | 判死原文 |
|---|---|
| `--ema`（指数滑动平均） | `MEMORY/docs/student-bottleneck-plan.md:250` "EMA 不采用" |
| `--width 128` | `MEMORY/docs/student-bottleneck-plan.md:255` "宽度 128 不采用"；`DESIGN.md:526` "诊断后实验：学生任务上下文、宽度 128 都没有提升" |
| `--horizon 1800` | `MEMORY/docs/student-bottleneck-plan.md:276` "视界 1800 不采用" |
| sim_skip 位 64（自适应视野） | `MEMORY/2026-10-17.md:78` "自适应视野位 64 老师 128 局验收没过，不并入" |
| `--soft-weight` / `--soft-temperature`（软目标全分布） | `DESIGN.md:567` "不用全部候选的分布（learn/03：软目标崩到 17/256）" |
| CUDA 扩展实体位图 | `MEMORY/docs/student-bottleneck-plan.md:202` "第 58 轮扩展实体位图无收益不采用" |

### 6.3 必须保留（例外，写理由）

| 项 | 理由 |
|---|---|
| 旧检查点加载（v34 slot 头、`identity_dw3_s1`、`dw4i_s1`） | 现行面板与文档仍在对照这几个模型（运行配置里分别出现 48 / 43 次） |
| 旧档案 `whatif` 重放所需字段 | AGENTS §8 第 4 条：重放走存档里记录的 deck |
| 编码器签名冻结（`encoder_signature` 与旧版本 v1/v2lite/v2） | 旧档案靠它判断能否直接复用，改了就得全量重算 |
| `--frames-format jsonl` 的**读取**路径 | 旧档案存在（`frame_paths` 两种后缀都收） |

### 6.4 删掉能省多少

按上面的清单：search_play 少 13 个参数与 6 条分支（GPU 预取 / 推测减半 / 覆盖候选 / 稀疏轨道 / Python 推演参考 / jsonl 写入），
train_rollout_policy 少 `--ema` `--width 128` `--soft-weight` `--soft-temperature` 四个与它们的分支，
合计约 **600–800 行**（估算，实际以 `git grep` 逐条清理为准）。删除前按 AGENTS §4 打 tag `retire-<名>-20261023`。

## 7. 重构计划（按 收益/代价 排序）

### 第 1 项（已做，§9）：worker 不再每局重建

- **现状（改前）**：`search_play.main` 里 `recycle = 1 if fork else None`（`maxtasksperchild=1`），macOS/Linux 上每个游戏都换一个新 worker 进程，
  于是引擎进程、`load_rollout_net`、torch 都重来一遍。实测 **1.62 s/局**。
- **病根**：为了压常驻内存（task18：macOS 把释放的页留在进程里），代价是每局固定开销。
- **改法（已做）**：`maxtasksperchild=None`（worker 常驻），引擎与网络随之复用。`RESET_V2` 从种子重建棋盘，所以**同种子仍打同一局**。
- **收益（A/B：上一个提交在 `/tmp/audit-old` 工作树里跑同一命令）**：8 局纯学生、1 worker，**22.64 s → 15.64 s（−31% 整跑，−0.88 s/局）**；
  搜索局 117 → 116 s（−1.4%）。面板（`unified_panel` 全部 `--arms fast`）直接受益。
- **代价**：worker RSS 从 fork 后 31 MB 长到 16 局内的 194 MB 峰值、之后 163–194 MB 来回（无单调增长）；搜索局 138 MB 持平。
- **验收（已过）**：8 局 `games.jsonl` 的 `won/final_wave/decisions/mowers_used/planted/searched` **逐字段相同**（0 处差异）。
- **风险**：Linux 上 glibc 的 arena 高水位比 macOS 更黏；台式机 16 worker 若内存吃紧，改成每 N 局回收（`maxtasksperchild=N`）即可，收益仍保留 N−1/N。

### 第 2 项：训练装载不再每轮重解压、重算 CSR

- **现状**：25 M 行 × 13.8 KB = 345 GB 解压 + 5.4 min 的 `np.nonzero`，每轮训练一遍；落盘只有 5 GB。
- **病根**：档案存的是稠密棋盘（96% 零），训练要的是 CSR（461 B/行）。同一份数据被"展开 → 压缩"两次。
- **改法**：采集端在写档案时同时写 CSR（`indices/values/offsets`），或每轮结束后把整轮档案合并成一个顺序大文件
  （`frames/<label>.pack`：头部索引 + CSR 块）。训练端顺序读，不再 `np.load` 21 000 个文件。
- **收益**：装载 48 598 行/s → 估计 300 k–1 M 行/s（去掉解压与 CSR 两遍），13.6 M 行 280 s → 20–45 s；磁盘 5 GB → 约 2.5 GB（CSR 直接落盘，不再存稠密）。
- **改动面**：`pvz_frame_store.write_frames*`（多写一组数组）、`train_rollout_policy.load`（优先读 CSR）、
  旧档案兼容（没有 CSR 就现算 → LEGACY 路径）。
- **风险**：档案格式变更；用编码器签名 + 新键名做版本判定，旧档案原样可读。
- **验收**：同一批档案，改前改后训练出的前 200 步损失一致到 1e-6；CSR 与原路径逐元素相同（`np.array_equal`）。
- **与 cb-mem 的边界**：cb-mem 管"常驻数组怎么压"（dtype、memmap、去副本）；本项管"从档案到常驻数组的这一段不再重复劳动"。合并实施时以本项为装载入口。

### 第 3 项：静态候选的 branch 往返（4.8 s/局，5%）

- **现状**：每次搜索把 40–240 个合法格子逐个 `branch_snapshot`（已按 `BRANCH_SNAPSHOT` 分块），每个分支回一个完整 observation JSON，
  Python 再 `_frame` + `featurize` + 价值网前向。
- **病根**：候选打分全在 Python 侧，每个分支一次 JSON 往返 + 一次 92 通道特征构造。
- **改法**（与第 4 项一起做）：把"种下去 → 静态网打分"整段下沉到引擎（C++ 侧已有 PVZRN006 与快照机制，价值网需要新增 C++ 前向）。
- **收益**：4.8 + 2.7 + 0.4 + 0.25 ≈ 8 s/局（7%）。
- **风险**：价值网要移植（权重格式、sigmoid、context 拼接），必须与 Python 逐位/1e-6 对齐。
- **验收**：同一批局面，C++ 与 Python 的价值网输出最大误差 < 1e-6；老师决策一致率不低于噪声底线 75.6%。

### 第 4 项：整局搬进 C++（§8，13% 上限）

### 第 5 项：档案小文件（21 056 个）

- **现状**：均值 235 KB，读取 92 MB/s（磁盘），解压 3.3 GB/s。
- **判断**：**不是瓶颈**。235 KB 的文件顺序读对 SSD 与页缓存都友好，解压比读盘快一个数量级。合并成大文件只在第 2 项里顺带做（为了顺序读与少一次解压），单独做没有收益。

## 8. 整局搬进 C++ 的量化依据

| 问题 | 数字 |
|---|---|
| 进程间通信占一局多少 | **1–3%**（139 MB 响应 / 8 062 次往返；JSON 解码约 0.5 s） |
| Python 侧全部可搬走的开销 | **13%**（6.7–14.9 s 搜索簿记 + 1.2 s step + 0.4 s 学生前向 + 0.2 s 写档案） |
| 每个 worker 的内存 | Python+torch 111–138 MB（RSS，含 3 个网络与观测缓存）+ 引擎 229 MB |
| 16 个 worker | Python 侧 1.8–2.2 GB + 引擎 3.7 GB |
| 引擎内学生前向 | 已有（PVZRN006，70 µs/次，8.7%） |
| 价值网 C++ 前向 | **没有**（`src/EnvironmentRollout.cpp` 里没有价值网，只有学生；GPU 侧叶子也是回到 Python 打分） |

要搬的函数（`scripts/search_play.py`，行数为函数体）：

| 函数 | 行 | 职责 |
|---|---|---|
| `_search` | 133 | 候选枚举、min-gap、候选并集、调用 halving |
| `_halving_choice` | 79 | 逐轮减半、推演分配 |
| `_verified_choice` | 67 | 复核、margin 判定、override 计数 |
| `_rollout_policy` | 58 | 一条推演（C++ 已是 `ROLLOUT`） |
| `_leaf_values` | 42 | 叶子 → 价值 |
| `_cuda_continuations` | 35 | GPU 续走（未启用） |
| `_continuations` | 23 | 批量推演 |
| `_fast_action` | 17 | 学生动作（C++ 已有） |
| `_values` | 14 | 价值网前向 |
| `_static_frames` | 12 | 静态候选打分 |
| `_score_terminal_or_value` | 6 | 终局/叶子判定 |
| `_frame_json` | 3 | 档案行序列化 |
| `_play` | 185 | 整局循环、DAgger、档案写出 |

合计 **674 行**（其中搜索本身 451 行、整局循环 185 行）。搬完后 Python 版搜索删除，不留两份实现。

**建议顺序**：先做第 2 项（训练装载，收益确定、与引擎无关），再做第 3 项（价值网 C++ 前向，是整局化的前置），
最后整局搬进 C++。第 1 项已做，收益最大的是面板（59%）。

## 9. 已做的改动

| 提交 | 内容 | 验收 |
|---|---|---|
| 86316a28 | 体检工具 + 一局的耗时拆解 | 工具本身 |
| （见 `git log`） | worker 常驻（§7 第 1 项） | 同种子 `games.jsonl` 逐字段相同 |

## 10. 复现命令

```bash
python3 scripts/hotpath_bench.py protocol           # 引擎启停 / step / rollout 往返与字节
python3 scripts/hotpath_bench.py archive --frames /path/to/frames
python3 scripts/hotpath_profile.py -- <search_play 的全部参数>
python3 scripts/hotpath_flags.py                    # 开关清单（201 份运行配置 + 全文引用）
python3 scripts/hotpath_deadcode.py                 # 顶层定义引用计数
```


### 6.5 scripts 下已无人调用的一次性脚本（只列，不删；`scripts/hotpath_scripts.py`）

0 次引用（全仓库没有任何文件提到它的文件名）：

| 脚本 | 行数 |
|---|---|
| analyze_n10p_eval.py | 63 |
| check_capacity_native.py | 78 |
| cuda_v2lite_check.py | 188 |
| dagger_data_audit.py | 131 |
| dagger_data_overlap.py | 102 |
| dagger_dilution_tables.py | 49 |
| dagger_dilution_tail_check.py | 90 |
| dagger_dilution_verify.py | 242 |
| dw4_night_card_logit.py | 193 |
| make_night_ab_panel.py | 77 |
| pack_frames.py | 31 |
| plot_value_curve.py | 181 |
| rollout_bench.py | 231 |
| static_value_ablation_stats.py | 105 |
| teacher_attribution_trace.py | 104 |
| teacher_counterfactual_audit.py | 246 |

合计 16 个脚本、2 111 行。1 次引用（只在某份历史报告 / 计划 json 里被提到一次，没有任何现行调用方）另有 20 个，
其中 `adaptive_verify.py`、`dagger_dilution_*`、`run_*_study.py` 都是已结束实验的驱动脚本。
删除按 AGENTS §4 由主控批。

