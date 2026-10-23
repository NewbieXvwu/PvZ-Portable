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

纯学生局（`--arms fast` **带** `--frames`，走 Python 决策循环；2.29 s）：

| # | 项 | 秒 | 占比 |
|---|---|---|---|
| 1 | 引擎启停 + reset（每局一次） | 0.79 | 34% |
| 2 | env.step 往返（1384 次，19 KB 响应） | 1.07 | 46% |
| 3 | Python 学生前向 | 0.20 | 9% |
| 4 | 写档案 | 0.15 | 5% |
| 5 | 帧 JSON | 0.03 | 1% |

**注意口径**：面板（`unified_panel`）跑 `--arms fast` 且**不写档案**，此时 `_play` 走整局一条 C++ ROLLOUT
（`env.rollout(None, 0, max_actions=...)`），没有 Python 决策循环：实测 **1.01–1.95 s/局**（改前 / 改后，见 §7 第 1 项），
不是上面 2.29 s 那一档。

## 3. 存储账

| 目录 | 文件数 | 大小 | 每文件 | 谁写 | 谁读 |
|---|---|---|---|---|---|
| `artifacts/hard_mac/frames/seed_N/<label>_fast_search/` | 21 056 个 `frames.npz` | 5.07 GB | 均值 235 KB（中位 240 KB，最大 0.5 MB） | `search_play` 每局一个 | 每次训练整目录读一遍 |
| `artifacts/value_net/episodes*`（**旧 JSONL 格式**） | 6 866 个 `frames.jsonl` | **7.54 GB** | 均值 1.1 MB | 价值网络采集 | 价值网络训练（每次训练重新 `json.loads`） |
| `artifacts/student/frames*` | 4 096 | 2.2 GB | — | 旧采集 | 旧训练 |

**artifacts 全量 13.52 GB / 59 535 个文件**，按后缀：`.jsonl` 7.54 GB（56%，全是旧档案）、`.npz` 5.36 GB、
`.log` 0.49 GB（221 个，最大的两个各 27 MB，无人轮转）、`.json` 0.06 GB（29 679 个，都是每局一个的 config/meta/summary）。

**旧 JSONL 档案换成 npz 是白捡的**：实测一个 3.72 MB 的 `frames.jsonl` → `frames.npz` **0.24 MB（7%）**，
`scripts/pack_frames.py` 在一个 seed 目录上跑 8 个档案：**12.70 MB → 1.04 MB（8%）**，每个 0.76 s。
全量 7.54 GB → 约 0.6 GB，省 7 GB 磁盘，并让训练装载不再 `json.loads` 文本。
**逐帧核对已做**：3001 帧 JSONL → npz → `read_frames` 读回，**每一帧的 dict 完全相同**（`a == b`，3001/3001）。

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
- **收益（A/B：改前提交 `4a0dba12` 在 `/tmp/audit-old3` 工作树里跑同一命令）**：
  面板口径（`--arms fast`、不写档案、整局一条 C++ ROLLOUT）16 局 **31.18 → 16.11 s（−48%，1.95 → 1.01 s/局）**；
  采集口径（`--arms fast` + 写档案）8 局 **22.64 → 15.64 s（−31%，−0.88 s/局）**；搜索局 117 → 116 s（−1.4%）。
  `unified_panel` 全部 `--arms fast`，因此面板直接受益。
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
- **原型已做并验证**（`scripts/pack_training_rows.py`）：两个标签 1.28 M 行打成 904 MB 单文件（581 B/行）；
  `--verify` 与 `load()` 逐项对比 **13 个数组全部 `np.array_equal` 相同**（rows / board_indices / board_values /
  board_offsets / scalar_* / plant / weight / packet / cell / values / seed / shape）。读回 1.56 M 行 **0.17 s**（mmap，惰性），
  对照 `load()` 的 164 s（无 cache-dir）或 32 s（带 cache-dir + 8 进程）。
- **真实收益要按"每份数据 4 个种子"算**（现行配方 D10–D13 都是 4 个种子）：一轮 4 次训练 × 280 s = **19 min** 的装载，
  打包一次 164 s + 4 × 2 s ≈ **3 min**。代价是包文件 904 MB/1.28 M 行 = 581 B/行（比压缩档案 288 B/行 大一倍，
  但比训练时重算省 100 倍 CPU）；也可用 `savez_compressed` 换磁盘，代价是一遍 zlib（单流，约 3 GB/s）。
- **与 cb-mem 的边界**：cb-mem 管"常驻数组怎么压"（dtype、memmap、去副本）；本项管"从档案到常驻数组的这一段不再重复劳动"。合并实施时以本项为装载入口。

### 第 3 项：静态候选的 branch 往返（4.8 s/局，5%）——**实测后撤回，见 §16.0**

- **现状**：每次搜索把 40–240 个合法格子逐个 `branch_snapshot`（已按 `BRANCH_SNAPSHOT` 分块），每个分支回一个完整 observation JSON，
  Python 再 `_frame` + `featurize` + 价值网前向。
- **病根**：候选打分全在 Python 侧，每个分支一次 JSON 往返 + 一次 92 通道特征构造。
- **原估收益**：4.8 + 2.7 + 0.4 + 0.25 ≈ 8 s/局（7%）。
- **实测（§16.0）**：8 s 里只有 json 序列化/解析 + `_frame` + `featurize` 是冗余，合计 0.7–1.0 s/局（0.6–0.85%）；
  每分支 220–280 µs 中引擎还原快照 61–90 µs、往返 73–120 µs 是候选真的在仿真，前向 93–107 µs 是 torch（MKL 多线程）本身。
  单线程 C++ 卷积比 torch 慢 3–6 倍，搬过去倒亏。**不实施**，设计留在 §16 备查。

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

**建议顺序**（2026-10-23 修订）：第 2 项（训练装载）**已落地**（§15）；第 3 项（静态候选下沉）**实测后撤回**（§16.0），
所以剩下的只有第 4 项“整局搬进 C++”——它的 13% 上限里，价值网前向那 3% 已经不划算了（torch 比单线程 C++ 快 3–6 倍），
真正能拿到的是 Python 搜索簿记里剩下的几秒，而它的主要价值仍是内存（§3）与只维护一套实现，不是速度。

## 9. 已做的改动

| 提交 | 内容 | 验收 |
|---|---|---|
| 86316a28 | 体检工具（`hotpath_bench` / `hotpath_profile`）+ 一局的耗时拆解 | 工具本身 |
| 4a0dba12 | 报告 + 开关清单 / 死代码 / 脚本清单工具 | `hotpath_flags` 覆盖 201 份运行配置 |
| **3854e758** | **worker 常驻（§7 第 1 项）** | 8 局 + 16 局 `games.jsonl` 逐字段相同 |
| 10df7165 | `pvz_compat.py` + LEGACY/EXPERIMENT 标记 + 双向校验测试 | `test_compat_registry` 5 项通过 |
| 93abd87e | `pack_training_rows.py` 打包原型 | `--verify` 13 个数组逐项相同 |
| bc9e2b91 / b78dec5d / 58447d65 | 面板口径 A/B 修正、打包验收与旧档案换算、压榨记录 | 见各节 |

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


## 11. 向后兼容与实验开关的统一登记（2026-10-23）

登记处：`python/pvz_compat.py`（纯数据 + `expired()`）。代码里每处兼容分支写 `# LEGACY(<id>)`、
每个保留的实验开关在 `add_argument` 上方写 `# EXPERIMENT(<id>)`，help 文字以
`[legacy]` / `[experiment until 2026-11-06]` 开头（`--help` 一眼可见）。

`python/test_compat_registry.py` 双向校验（5 项，全部通过）：标记必须在登记处、登记处必须在代码里出现、
决定日已过就失败、每条 LEGACY 必须写"为谁保留"与"移除条件"、每个实验开关必须在它声称的脚本里真实存在。

已登记的 8 条 LEGACY：

| id | 保留给谁 | 移除条件 |
|---|---|---|
| obs-versions | 21 056 个旧档案 + v34 / identity_dw3_s1 / dw4i_s1 检查点 | 旧档案重算完、面板不再对照 v2lite/v2 |
| jsonl-archives | 2026-10 以前的 JSONL 档案（whatif 重放） | artifacts 下不再有 frames.jsonl |
| slot-packet-head | v34 及更早的学生（面板对照） | v34 退出所有面板与文档 |
| python-rollout-reference | `perf_rollout_check.py` 的逐帧对拍 | 对拍改用存档回放 |
| conveyor-start-key | L10 传送带任务族 | 传送带任务族退役 |
| dense-boards | CSR 与稠密两条路径的对拍测试 | CSR 有独立逐元素测试 |
| aux-labels | task16 辅助头实验数据 | 辅助头判死或转正 |
| pvzrn-versions | 旧导出文件与 `test_cpp_rollout.py` 版本矩阵 | 测试只留 PVZRN006 |

已登记的 8 个实验开关（决定日统一 2026-11-06）：`--rollout-backend`、`--cuda-prefetch-verify`、
`--cuda-speculative-halving`、`--candidate-generator`、`--candidate-coverage-model`、
`--shovel-candidates`、`--cpu-sparse-tracks`、`--measure-cpu`。到期测试会失败，逼着转正或删除。

## 12. 观测载荷（下一项的依据）

一局学生决策的响应 11.7–14.3 KB，构成（同一 tick 实测）：

| 键 | 字节 | 占比 |
|---|---|---|
| `legal_actions` | 5 257 | 45% |
| `cells` | 4 375 | 37% |
| `packets` | 676 | 6% |
| `zombies` | 374 | 3% |
| `loadout_context` + `defenses` + `player_profile` | 668 | 6% |

引擎已有 `OBS_LEAN`（`LawnApp::EnvironmentLeanObservation`，搜索推演一直在用）。实测同一 tick 下
lean 观测是完整观测的 **51%**（14 280 → 7 341 B），且 `render_episode._frame` 与
`pvz_value_net.featurize_numpy` 在 lean 上产出**逐位相同的棋盘**（max abs diff 0.0）。

缺的键：`cells`、`coins`、`loadout_context`、`player_profile`、`playthrough`、`projectiles`、
`protocol_version`、`task_version`、`zombie_count_multiplier`。其中 `zombie_count_multiplier` 与
`loadout_context` 被 `_play` 用来构造价值网的任务上下文，**必须补进 lean** 才能整局用 lean。

**这一项没有做**：它要改 `src/LawnApp.cpp` 并重编 `build/pvz-portable`，而 build 目录与正在跑的
13 个采集进程共用，重编会换掉它们手里的二进制。留给 task35（引擎侧）或在采集停下的窗口做。
预计收益：面板局 step 0.88 → 约 0.65 s（−0.23 s，−20% 面板局）；搜索局 step 1.2 s 同比例。


## 13. 压榨记录与停止条件（2026-10-23 01:30）

按"每做完一项就重测、挑下一个最大项"执行，已试与未试的都在这里，避免重复劳动。

| # | 尝试 | 量到的收益 | 状态 |
|---|---|---|---|
| 1 | worker 常驻（不再每局重建进程） | 采集口径 −31%（8 局 22.64→15.64 s）、面板口径 −48%（16 局 31.18→16.11 s）、搜索局 −1.4% | **已做**，结果逐字段相同 |
| 2 | 训练行打包成单个顺序 CSR 文件 | 装载 164 s → 0.17 s（读回），13 个数组与 `load()` 逐项相同 | **原型已验**，接线归 cb-mem（同一个 `load()`） |
| 3 | 旧 JSONL 档案转 npz | 体积 8%（12.70→1.04 MB / 8 局），全量 7.54 GB → 约 0.6 GB；训练不再 `json.loads` | **已量，未做**：要动主工作区 artifacts，按规则由主控批 |
| 4 | 整局用 lean 观测（step 与静态候选的 branch 返回） | 载荷 51%（14 280→7 341 B），学生特征逐位相同；缺 `zombie_count_multiplier`/`loadout_context` 两项 | **被阻断**：要重编 `build/pvz-portable`，与在跑的采集共用二进制 → task35 或停采窗口；估 1–2% |
| 5 | 价值网批大小 32→128 | 上限 2.3%（`_values` 全部 2.7 s） | 未做（上限低于门槛） |
| 6 | 静态候选下沉到 C++（含价值网 C++ 前向） | 原估 8 s/局（7%）；**逐项实测后可回收只有 0.7–1.0 s（0.6–0.85%）**，且 C++ 前向比 torch 慢 3–6 倍 | **已量，不实施**（§16.0） |

**停止条件**：要求“连续 3 项尝试收益 < 2% 且前 15 项里没有可证明的冗余”。本节写于 01:30，当时的判定是**不满足**——
第 2 项（每轮重解压 345 GB + 重算 CSR）与第 3 项（7.54 GB 文本档案）都算冗余。这两条后来都有结论，见 §15 与 §17：

- 第 2 项**已落地**（§15：训练只读包，装载 8.1 s → 0.009 s，逐位相同）；
- 第 3 项（静态候选下沉）**实测后撤回**（§16.0：可回收只有 0.6–0.85%，C++ 前向比 torch 慢）；
- 旧 JSONL 档案（1.7 GB / 6 659 个）在打包落地后不再是热路径项（训练不读档案），只剩磁盘与价值网训练的解析时间。

**当前判定**：满足（§17 末尾）。下面这段 01:30 的结论保留原文，只作当时的记录——其中“最大单项 7%”已被 §16.0 修正为 0.6–0.85%。

前 15 项里剩下的耗时（引擎仿真 77%、引擎内学生前向 9%）属于 task35 的引擎 CPU / C++ 推演；
本任务能碰的部分（Python 簿记 13%）里，可做的最大单项是 §7 第 3 项的 7%。

**结论**：采集侧 Python 的油已经基本榨干（剩下的每一项都 < 2%，且都要改引擎）；
真正还剩的大头在"数据搬运"（训练装载与旧档案格式），需要与 cb-mem / 主控分工执行。


## 14. 删除执行记录（2026-10-23，主控 08:30 批复）

Tag：`retire-cuda-20261023`（整条 CUDA 线）、`retire-flags-20261023`（开关与判死实验）。全部用 `git grep` 复查过无引用，
文档里的旧数字一律没动。**共 101 个文件、-24 165 行 / +184 行**（一次提交）。

| 线 | 删掉的东西 | 行数 |
|---|---|---|
| CUDA 推演后端 | `cuda/`（RolloutCuda.cu、SimCore.h、SimTick.inc、spec/ 等 30 个文件）、`src/CudaExport.{cpp,h}`、`CMake/CudaRollout.cmake`、`python/pvz_cuda_{rollout,root}.py`、9 个 `test_cuda_*.py`、`test_gpu_collection_pause.py`、11 个 `scripts/cuda_*.py` | ≈7 700 |
| 同一条线的实验开关 | `search_play.py` 的 `--rollout-backend/--cuda-{binary,socket,lanes,gather-ms,prefetch-verify,speculative-halving}` 与它们的分支（`_cuda_continuations`、`_wait_for_gpu_pause`、speculative halving、prefetch 缓存）、`env.cuda_dump` 与引擎 `CUDA_DUMP` 命令 | ≈500 |
| 混合采集 | `scripts/mixed_dagger_round.sh`（整个脚本只为 CPU+GPU 混合采集存在）、`scripts/task6/`（r66_*、compare_worker_mix、sparse-track 检查） | ≈1 900 |
| 候选实验 | `--candidate-generator`、`--candidate-coverage-model/-budget`、`--shovel-candidates`、`--cpu-sparse-tracks`、`--measure-cpu`、`--cpp-rollouts`、`--frames-format`（写入分支）、`--margin`/`--max-actions`（改常量 `MARGIN=0.02`、`MAX_ACTIONS=3000`）与它们的分支 | ≈450 |
| 判死实验 | `--ema`、`--soft-weight`（软目标全分布）及保存到 checkpoint 的两个字段；候选覆盖实验线（`pvz_candidate_coverage.py`、`train/teacher_candidate_coverage.py`、`teacher_coverage_stats.py`、对应测试）；静态价值消融线（`static_value_{ablation,compare,noninferiority}{,_stats}.py` 与 3 个测试） | ≈3 500 |
| 观测与登记 | `python/pvz_compat.py`：`EXPERIMENTAL_FLAGS` 清空（8 个全部删除），`dense-boards` 一条随 task32 删除，`python-rollout-reference` 改成 `_python_rollout_reference` | — |

**保留（按批复）**：`--resource-dir`（两台机器路径不同）；`--width`（模型结构参数，加载/定义模型要用）；
`--soft-temperature`（`--cell-soft-weight` 仍在用，help 已改写为只讲格子级软标签）；`--frames-format` 的 jsonl **读取**路径（旧档案）。

**没删、留给下一批的**：

1. 引擎侧 `SIM_SKIP` 位 256（稀疏轨道）与 `pvz_env.require_sparse_tracks`——Python 侧已无调用方，但改 `src/EnvironmentRollout.cpp`
   会和台式机 task35 撞车，按主控要求留给 task35 或主控协调后再删。
2. 宽度 128 对照代码只存在于 `scripts/run_dw4_day_diagnosis.py`（task23 已结束的一次性 study runner）；
   它同时包含别的阶段，整脚本删除不在本次批复范围内，列进 §6.5 的同类清单。
3. 视界 1800、`sim_skip` 位 64 是“值”不是“开关”，代码里没有分支，删的是用法不是代码——无动作。

**验收**：`python3 -m pytest python -q` → 229 passed / 24 skipped / 56 subtests，6 failed 全部是环境性
（5 个 `test_cpp_rollout` 需要 `build/pvz-portable`，工作树里没有；1 个 `test_dagger_experiment_runner` 的导入顺序）。
`python/test_compat_registry.py` 5 项全过。`compileall scripts python` 通过。

## 15. 打包装载转正为唯一路径（第 3 项执行记录，2026-10-23）

### 改了什么

| 位置 | 之前 | 现在 |
|---|---|---|
| `train_rollout_policy.load()` | 逐档案解压 `frames.npz` + 重算 CSR，**每次训练都做一遍** | 只读包：`pack_path(data, options)` 命中就直接 mmap 读回；缺包时报错并打印该跑的打包命令 |
| 谁读档案 | 训练端 + 打包原型 | 只有 `scripts/pack_training_rows.py`（`build_from_archives`）。jsonl 旧档案的读取仍在 `pvz_frame_store.frame_bytes`，标 `LEGACY(jsonl-archives)` |
| 采集端 | 只写档案 | `curriculum_queue.run_round` 每轮结束调 `write_pack`：把“本队列写过的全部标签 + `--pack-labels`”打成包，并 `--prune` 删掉被它取代的旧包 |
| 开关 | — | 无。没有“读档案 / 读包”的开关，旧路径已删除 |

包键 = 所有改变行的选项：`labels`、`wait_keep`、`obs`、`cells`、`aux_horizon`、`packet_head`、`context`、`label_weight`、
以及 `sources`（档案相对路径 + 大小的 sha1[:16]，不含 mtime）。`pack_options()` 一处定义，打包器与训练端各自算出同一个
`<data>/packs/<key>.npz`，谁都不用被告知包在哪。**注意 `cells` 跟着 `--cell-soft-weight` 走**（`load()` 里
`cells=bool(args.cell_soft_weight)`），生产配方带 `--cell-soft-weight 3` 时采集端必须用 `--pack-cells`。

包内容（npz 顺序大文件，不压缩）：board / scalar 两套 CSR（常数平面已从棋盘里提出去）、`plant`/`packet`/`cell`/`weight`/`seed`、
`values`（稀疏）、`cands`（可选）、`aux`（可选）、`source`/`source_archive`/`source_row`（逐行来源，`--source-row-report` 要用）。
即 task32 的紧凑数组是常驻形态，打包只是把它落盘。

### 验收一：同一批数据，改前改后训练前 200 步逐位相同

`artifacts/audit_bench/frames`（11 标签、88 692 行、7 509 个种植决策、47 个种子、16 917 验证行），
`--epochs 6 --max-steps 200 --seed 1 --device cpu --width 32`：

| epoch | train_loss（逐档案 / 读包） | val_loss（逐档案 / 读包） |
|---|---|---|
| 1 | 6.405445768169641 / **相同** | 5.994540454296029 / **相同** |
| 2 | 5.413421458825962 / **相同** | 5.055646445316126 / **相同** |
| 3 | 4.381605839548274 / **相同** | 4.21451249065964 / **相同** |
| 4 | 3.5603428646049116 / **相同** | 3.653569811292593 / **相同** |
| 5 | 3.118014608646926 / **相同** | 3.271407952903116 / **相同** |
| 6 | 1.602656311226489 / **相同** | 3.14441666852824 / **相同** |

对照侧用 tag `retire-flags-20261023` 的独立工作树（`git worktree add --detach /tmp/audit-prepack`），两边的
`type_acc` / `plant_recall` / `packet_acc` / `cell_acc` 也逐位相同——差 0，不是 1e-6 以内。

### 验收二：装载耗时与峰值内存

同一份数据、`wait_keep=1.0`、`aux_horizon=0`（训练不带 `--aux-weight` 时的实际取值），各跑两次：

| 装载路径 | 行数 | 耗时 | 峰值 RSS | 行/s |
|---|---|---|---|---|
| 逐档案（tag `retire-flags-20261023`） | 88 692 | 8.23 s / 8.10 s | 305.4 MB / 304.3 MB | 10 900 |
| 读包（本次） | 88 692 | **0.009 s / 0.009 s** | **227.5 MB / 226.8 MB** | 9 650 000 |

打包成本（一次性，之后每轮只重打并集）：冷缓存 6.5 s、热缓存 2.1 s（88 692 行），产物 23 207 118 B = **262 B/行**
（不压缩的 npz；13.6 M 行约 3.5 GB）。13.6 M 行按 10 900 行/s 的旧路径要 **21 分钟**，读包仍是毫秒级。

### 生产配方也验过（不只是默认选项）

上面两组验收用的是默认行选项（v1、slot 头、无 cands）。生产配方是 `--wait-keep 0.1 --obs v2 --packet-head identity
--cell-soft-weight 3`（训练端 `cells=bool(cell_soft_weight)`），它多走三条路径：v2 观测的历史通道、identity packet 列、
候选格子（`CandidateStore`）。同一份 11 标签数据按这套选项再打一次包：

| 项 | 结果 |
|---|---|
| 行数 / 体积 / 打包耗时 | 17 069 行（wait_keep 0.1）/ 10 024 562 B / 23.2 s（冷缓存） |
| `--verify` 逐数组 | **12 项全部 same**（meta、rows、board、scalar、plant、packet、cell、weight、seed、source、values、cands） |
| 训练 1 epoch（`--cell-soft-weight 3 --aux-weight 0.5 --obs v2 --packet-head identity`） | 正常跑通；`data_memory` 里 `cand_bytes` 173 579、`packet_bytes` 1 194 830，说明候选与 identity packet 确实是从包里读到的 |

### 采集端接线

`scripts/curriculum_queue.py`：`pack_labels()`（本队列 `state['log']` 里出现过的标签 + `--pack-labels`）、
`pack_command()`（argv）、`write_pack()`（子进程跑打包器、把输出 JSON 写进队列日志、失败则 `SystemExit`——
放在状态落盘之后，采到的局不会因为打包失败丢掉）。打包器新增 `--prune`：删掉标签集是本包子集的旧包
（每轮的并集都包含上一轮，13.6 M 行时一个包 3.5 GB，不删会按轮数线性堆）。

**采集端的打包成本（写在明处）**：打包器从档案重建 CSR，速度取决于行选项。默认选项 2 100 行/s（88 692 行 2.1 s，热缓存），
生产配方（obs v2 + identity + cands）**3 800 行/s**（88 692 行 23.2 s，冷缓存）。每轮重打的是"到目前为止的并集"，
所以成本是 O(轮数²)：8 轮、13.6 M 行时最后一轮约 1 小时 CPU。相对每轮数小时的采集可以接受，且它跑在采集之后、不占 worker。
要消掉它只能把等待行的抽样改成"每档案独立种子"（rng 流不再依赖前序档案），那会改变行选择——**本次不做**，因为验收要求
与改前逐位相同。

单测：`python/test_curriculum_queue.py` 加 `test_pack_labels_union_the_log_and_the_base_corpus`、
`test_pack_command_carries_the_row_options`。全量 `pytest python -q` → **231 passed / 24 skipped / 56 subtests**，
6 failed 全是环境性（同 §14：5 个 `test_cpp_rollout` 缺 `build/`，1 个 `test_dagger_experiment_runner` 导入顺序）。

## 16. 第 4 步：静态候选下沉到引擎（设计，未实施）

主控要求只准备设计、**不动 `src/EnvironmentRollout.cpp` 的学生前向**（台式机 task35 正在改）。下面的接口按“新增文件 + 一处 dispatch”设计，
需要 task35 配合的点写在 §16.6，同时抄进任务书的“问题”。

### 16.0 实测复核（2026-10-23 13:20）：这一项的收益是 **≤1%**，不是 7%——不实施

§13 第 6 项把“4.8 + 2.7 + 0.4 ≈ 8 s/局（7%）”当成可回收的收益，但 4.8 s 里绝大部分是**引擎真的在干活**（还原快照 + 执行动作），
2.7 s 是价值网前向本身，两者都不是冗余。逐项量出来的每分支成本（scripts/hotpath_branch.py，1 个进程、现役 build/pvz-portable、
L7 倍率 3、最坏 lean 观测 3 825 B / 10 只僵尸；两次运行取区间，机器上有 15 个采集进程，±30%）：

| 每分支 | µs | 搬进 C++ 后 |
|---|---|---|
| 引擎还原快照（BENCH_SNAPSHOT restore） | 61–90 | 留下（候选必须真的执行） |
| BRANCH_OBSERVE 往返（含执行、序列化、传输、解析） | 73–120 | 只剩“执行” |
| json.loads（Python 侧，最坏 3.8 KB） | 21–23 | **可去** |
| render_episode._frame | 4–6 | **可去** |
| pvz_value_net.featurize_numpy | 8–10 | **可去** |
| 价值网前向（torch，批 18） | 93–107 | 换成 C++ 前向——**不会更快** |
| 合计 | ≈220–280 | 可去 ≈35–50 µs = **15–20%** |

19 556 个分支 × 35–50 µs ≈ **0.7–1.0 s/局 = 0.6–0.85%**（按最坏载荷算的上限）。而“把前向也搬进 C++”这一半更糟：价值网每帧 6.1 M 次乘加，
torch 用 MKL 批 18 帧跑到 93–107 µs/帧（约 76 GFLOP/s，多线程）；单线程朴素 C++ 卷积按 20 GFLOP/s 算是 0.3–0.6 ms/帧，
**比 torch 慢 3–6 倍**，19 556 帧要多花 4–10 s，把上面省下的 1 s 全吃掉还倒亏。

结论：静态候选下沉**不实施**。下面的设计留档，等引擎侧有批量/多线程的卷积实现（或价值网换更小的结构）再回来看；
现在按停止条件记为“实测后不成立”，不是“待办”。测法：scripts/hotpath_branch.py（同一状态取 8 个合法格子，分别量往返、解析、
_frame、featurize 与前向，并用整局里最大的 lean 观测当解析样本）。
### 16.1 搬的是哪一段

现在（`scripts/search_play.py`）：

```
_search
  → _static_frames(env, root, plants, income)        # 472 条 BRANCH_OBSERVE/局，19 556 个分支
      → env.branch_snapshot(root, chunk, retain_snapshots=False)
          → 每个分支回一个 observation（JSON，§5 口径：完整 14 280 B / lean 7 341 B）
      → render_episode._frame(observation)            # JSON → frame dict
      → pvz_value_net.featurize(frame)                # 92×6×10 棋盘 + 113 标量（再拼 35 维 context）
  → _values(..., static=True)                         # torch 前向 + sigmoid
  → 按卡分组取前 --per-card 个格子
```

下沉的是**“在快照上执行动作 → 得到一个分数”整段**，返回只有浮点数。候选集合的构造（每卡 top-k、学生 top-k、NOOP、策略动作、`--per-card` 的排序）
**留在 Python 不动**——这是语义最容易出错的地方，搬它没有收益。

### 16.2 账（§13 第 6 项）

§13 的原始口径（现在只作对照，逐项拆解见 §16.0）：分支往返 4.8 s、价值网前向 2.7 s、`_frame`+`featurize` 0.66 s。
拆开之后**只有 json 序列化/解析 + `_frame` + `featurize` 是冗余**，合计 ≈1 s/局（0.6–0.85%）；分支往返的大头是引擎还原快照
与执行动作（每分支 61–90 µs），前向本身 torch 已经比单线程 C++ 快。

### 16.3 接口

**权重格式 `PVZVN001`**（新，`python/pvz_value_net.py::export` 写、`src/ValueNet.cpp` 读）：

```
magic  "PVZVN001" (8 B)
int32  width, context, board_channels(=92), rows(=6), cols(=10), scalars(=113), context_size(=35)
float32, 按此顺序：conv0.w[W][92][3][3] conv0.b[W] conv2.w[W][W][3][3] conv2.b[W]
                    conv4.w[W][W][1][3] conv4.b[W] scalar.w[W][113+35] scalar.b[W]
                    head0.w[W][W*8] head0.b[W] head2.w[1][W] head2.b[1]
```

读取时校验 magic / 所有形状常量 / 文件长度，任何一项不符直接拒绝加载（不猜、不兼容旧格式）。小端，与 `PVZRN00x` 同一约定。

**命令**（`src/main.cpp` 的命令分发里加两行；实现全在新文件）：

| 命令 | 请求 | 响应 |
|---|---|---|
| `VALUE_LOAD <path>` | 价值网权重路径（与 `LOAD_STUDENT` 并列） | `{"ok":true,"width":64,"context":true}` |
| `BRANCH_VALUE <snapshot_id> <income> <spec>...` | spec 与 `BRANCH_OBSERVE` 同一编码（`P:packet:row:col` / `S:` / `W:`），上限沿用 128 | `{"values":[0.42,null,...]}`，顺序与 spec 一致；`null` = 该分支 `ok=false`（Python 侧照现在的规则跳过） |

- 复用 `BRANCH_OBSERVE` 的循环体（restore 父快照 → `ExecuteEnvironmentAction` → 打分），**不复制一份**：在同一个循环里按模式选择“输出 observation”或“输出分数”，
  这样两条路径的 restore/execute 逐行相同，不会分叉。
- `income` 由 Python 传（`sun_income_rate` 是一局的运行估计，引擎里没有）。
- context（multiplier / roster）由引擎在 reset 时已知，`VALUE_LOAD` 之后按任务算一次，不需要每条命令传；`context=false` 的权重忽略它。
- 特征化在 `src/ValueNet.cpp::Features(LawnApp*, double income, board, scalars)`：直接读棋盘结构，不经过 JSON。
  **不复用 `EnvironmentRollout::Features`**——那是学生观测（obs v1/v2lite/v2 会改同一批 92 通道的内容），价值网的 92 通道是 v1 语义，混用会静默算错。

### 16.4 等价性验收

1. **权重往返**：`export` 写出的二进制读回，逐权重与 torch 的 `state_dict` 按位相同（`np.array_equal`，不是 allclose）。
2. **特征化**：同种子 `day_l7_z17` 1740001 的真实局抽 10 000 个决策点，Python `featurize` 与 C++ `Features` 的 board/scalars **逐元素相同**（float32 同序加法，允许 0 误差；有差就是搬错了）。
3. **分数**：同一批局面的 C++ 与 Python 价值网输出 `max |diff| < 1e-6`（含 sigmoid；先比 logit 再比概率，便于定位）。
4. **决策级**：同种子 A/B（1740000–1740015，学生 dw4i_s1 自走），`games.jsonl` 的 won / decisions / searched / planted 逐字段相同；
   老师决策一致率不低于噪声底线 75.6%。
5. **性能**：按 §13 的口径重测一局的账，期望 −7.9 s（−6.7%）；`BRANCH_VALUE` 的单次中位耗时用 `BENCH_SNAPSHOT` 同样的办法拆开（restore / execute / featurize / forward）。

### 16.5 风险

- **浮点顺序**：卷积用 float32、同一 padding/stride、同一 kernel 累加顺序；CMake 若对这段开了 `-ffast-math` 要关掉（先查 `CMakeLists.txt` 的 Release 选项）。1e-6 门槛足够容忍重排，但不容忍 `-ffast-math`。
- **逐字照抄的细节**：`_col_of`（`(x-40)/80` 截到 0..9）、`min(kind, PLANT_TYPES-1)`、`health/200`、`health/max(1,max_health)`、
  `amax(dim=3)`（每行最坏格，不是 mean）、`mean(dim=(2,3))`、拼接顺序 pooled → rows → scalar、`on_board` 判定。
- **并列分数的稳定性**：`sorted(..., key=-score)` 是稳定排序，Python 侧必须保持同一顺序（C++ 只返回分数，不返回顺序，所以这点不受影响）。
- **回退**：`BRANCH_VALUE` 出任何不一致，Python 侧仍可用 `BRANCH_OBSERVE` + `_values` 走原路（实现期保留到验收通过为止，通过后按“不留两份实现”删除）。

### 16.6 需要的接口（**作废**：本项不实施）

原设计要 task35 确认 `main.cpp` 的命令分发与 `BRANCH_OBSERVE` 循环体是否会被他重写。§16.0 量完之后这一项不做了，
所以这些接口需求一并撤回，不需要占用他的时间。留档的原因：如果将来价值网换成更小的结构、或者引擎有了批量多线程的
卷积实现，这套接口（`VALUE_LOAD` / `BRANCH_VALUE` / `PVZVN001`）可以直接拿起来用。

## 17. 压榨第二轮：三项尝试与停止条件（2026-10-23 11:50）

第 2 项（打包装载）落地后重测，剩下的 Python 侧候选逐项量完，三项都低于 2%：

| # | 尝试 | 量到的收益 | 结论 |
|---|---|---|---|
| 7 | **step 也用 lean 观测**（`OBS_LEAN` 已在现役二进制里，不用重编） | 每步 **−23…−26%**（两次运行：0.32→0.23 ms、0.44→0.33 ms（L7）、0.56→0.43 ms（L21））；400 步 × 2 关的 `_frame` 逐条相同、载荷 15.8–16.7%。但搜索局的 step 只占 1.2 s/117.4 s，−26% = **−0.3 s（0.26%）** | **不收**（低于 2%）。结论留档：lean 对 step 是安全的（帧相同），将来整局搬 C++ 时顺手用 |
| 8 | 价值网批大小 32 → 128 | `_values` 实际每批 **18 帧**（17 259 次 featurize / 936 次调用），`VALUE_BATCH=32` 从未被填满 → 收益 **0** | **不收**（不是"收益小"，是这条路不存在） |
| 9 | 静态候选跳过 `_frame`（直接从 lean 数组喂 `featurize`） | `_frame` 13 µs × 19 556 = 0.25 s，上限 **0.2%** | **不收** |
| 10 | 静态候选下沉复核（§16.0，原估 7%） | 逐分支拆开后可回收 35–50 µs × 19 556 = **0.7–1.0 s（0.6–0.85%）**；C++ 前向比 torch 慢 3–6 倍 | **不实施**，接口需求撤回 |

第 7 项的测法：`scripts/hotpath_lean.py`（1 个进程、现役 `build/pvz-portable`）——同种子同动作流跑两遍（`OBS_LEAN 0/1`），
比较每一步 `_frame` 与载荷。两关 400 步全部 `frames_identical: true`。这条同时证明 §16 之外还有一条不用重编的瘦身路，
但它在搜索局里的权重太小。

**停止条件（“连续 3 项 < 2% 且前 15 项无可证明冗余”）的判定**：

- 连续 3 项 < 2%：**满足**（第 7/8/9 项，见上表；第 10 项把原估 7% 的静态候选下沉复核到 0.6–0.85%，也 < 2%）。
- 前 15 项里可证明的冗余：**没有了**。
  1. 第 3 项（静态候选 branch 往返 5% + 价值网前向 3%）原先被当成“8 s 的冗余”，§16.0 逐项拆开后只有
     json 序列化/解析 + `_frame` + `featurize` 是冗余，合计 0.7–1.0 s/局（0.6–0.85%），而且换 C++ 前向会更慢。
     **实测后不成立**，按停止条件不算冗余。
  2. 第 2 项（引擎内学生前向 9%）与第 1 项（仿真 77%）是 task35 的引擎计算，不是冗余。
  3. 第 5 项（复核推演 2–3%）是算法取舍（多推演换准确率），不是冗余。
  4. 第 6 项以下（引擎启停 1.1%、step 1.2%、叶子 0.7%、Python 学生 0.4%、featurize 0.4%、`_frame` 0.3%…）单项都 < 1.2%，
     且第 7 项已经证明 step 这一档最多只能再挖 0.26%。
- 旧 JSONL 档案（§13 第 3 项，1.7 GB / 6 659 个文件）**不再是热路径项**：打包落地（§15）之后训练只读包，档案的唯一读者是打包器；
  转换它只剩磁盘 1.5 GB 与价值网训练（`train_value_net.py` 仍逐档案读）的解析时间，属于数据整理，要动主工作区、等停采窗口，
  与“采集/训练热路径的冗余”不是一回事。

**结论**：停止条件满足。本任务可独立执行的部分已经压到底；剩下的都是别人手里的计算（引擎）或需要窗口的数据整理，
不是“还有没榨的油”。
