# task38：训练配方与统一运行指标

工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/recipe`，分支 `recipe`。
不更改主工作区、其他工作树、tmux 或队列；冒烟最多使用一个自己的游戏进程。

## 配方契约

唯一 schema/default 实现在 `python/pvz_recipe.py`，当前 `schema_version=1`。
`kind` 为 `training` 或 `panel`。未知字段（包括 optimizer / validation / provenance 内部）、
重复 JSON 键、错误类型、越界数字、非绝对路径均报错。完整默认值在装载时展开，
每次运行在输出目录保存 `recipe.resolved.json`；学生检查点同时记录配方与原始配方 SHA256。

训练只接受 `--recipe` 与运维覆盖 `--out`、`--device`、`--threads`、`--precompute-workers`。
覆盖后的值也进入 resolved 配方。输出覆盖同时迁移 `source_rows.json`，不会写回历史运行目录。
配方中的路径必须为绝对路径；跨机器使用前应修改数据/输出/缓存路径，并保存为新的配方。

| 训练字段 | 类型 / 默认 | 含义 |
|---|---|---|
| data / labels / out | 必填绝对路径 / 非空字符串数组 / 绝对路径 | 数据根、明确的来源标签清单、输出目录 |
| label_weight / row_weights | 对象 `{}` / 可空绝对路径 | 标签权重、已有逐行干预清单 |
| wait_keep / packet_balance | 1.0 / 0.0 | 等待行保留与逆频率权重 |
| spread_weight / spread_cap | 0.0 / 5.0 | 搜索价值跨度权重 |
| width / pool / row_pool | 64 / mean / false | 网络结构 |
| packet_head / obs / context | slot / v1 / false | 身份头、冻结观测编码版本、任务上下文 |
| aux_weight / aux_horizon | 0.0 / 8 | 辅助损失与标签视野 |
| cell_soft_weight / soft_temperature | 0.0 / 0.05 | 格子软标签 KL 与温度；温度仍服务格子软标签 |
| optimizer | AdamW, lr=0.002, weight_decay=0.0001, betas=[0.9,0.999], eps=1e-8, amsgrad=false | 显式优化器设置；嵌套对象也拒绝未知键 |
| epochs / max_steps / batch_size | 12 / null / 2048 | 训练长度、可选固定步数、批大小 |
| seed / validation / validation_batch_size | null / {seed:0,fraction:0.1} / 4096 | torch 种子、按环境种子的留出划分、验证批大小 |
| lr_schedule / init / save_epochs | constant / null / 0 | 学习率调度、初始检查点、保留末尾 epoch 数 |
| cache / cache_dir / read_cache_dir | true / null / null | 特征缓存与只读缓存；默认写输出父目录下 .feature_cache |
| dense_boards | false | 保留现有稠密/CSR 对拍能力 |
| source_row_report | false | 在输出目录写 source_rows.json |
| device / threads / precompute_workers | cpu / 1 / 1 | 可被运维 CLI 覆盖的设备与并发 |
| provenance | 对象 | 原始命令文件与 SHA、原始可写路径和说明，不参与训练 |

按 task33 §6 批准清单删除训练 EMA 和全候选软分布入口及分支；width 保留为结构字段，
cell-soft 的 temperature 保留。旧报告需要的 checkpoint `ema=0` / `soft_weight=0` 元数据仍写出。
删除前 tag：`retire-recipe-training-flags-20261023`。

## 三份实际训练配方

| 配方 | 来源标签数 | 原始命令记录 |
|---|---:|---|
| recipes/identity_dw3_s1.json | 82 | identity/artifacts/task19/identity_training.command.json |
| recipes/dw4i.json | 170 | dw4/artifacts/task23/training_s1.command.json |
| recipes/dw5i.json | 281 | dw5/artifacts/task27/training_s1.command.json |

来源都在 `/home/newbiexvwu/PvZ-search/.claude/worktrees/` 下。
使用迁移前 argparse 解析实际记录，逐字段提取显式值和默认值；来源 SHA 保存在 provenance。
三份都是 identity、width64、meanmax、row_pool、obs v2、seed1、12 epoch、cosine、
wait_keep0.1、packet_balance0.5、spread_weight8/cap10、cell_soft_weight3；
早期 z13–z16 标签权重1.5，其余1.0。输出与可写缓存迁到 recipe 工作树，原数据只读。

旧命令示例（实际标签清单省略）：

```bash
/home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/dw4/scripts/train_rollout_policy.py --data /home/newbiexvwu/PvZ-search/.claude/worktrees/dw4/artifacts/task23/training_frames --labels '<170个标签>' --out /home/newbiexvwu/PvZ-search/.claude/worktrees/dw4/artifacts/task23/dw4i_s1 --packet-balance .5 --pool meanmax --wait-keep .1 --obs v2 --spread-weight 8 --spread-cap 10 --seed 1 --device cuda --lr-schedule cosine --cell-soft-weight 3 --row-pool --packet-head identity
```

迁移后完整命令：

```bash
/home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/recipe/scripts/train_rollout_policy.py --recipe /home/newbiexvwu/PvZ-search/.claude/worktrees/recipe/recipes/dw4i.json
```

`run_dw4_study`、`run_identity_validation`、`run_conveyor_study` 和历史容量诊断驱动引用基础配方，
先将实际数据清单/种子/输出变体冻结为 JSON 再启动；已有不同内容的冻结文件不得覆盖。
新 study plan 包含配方哈希。没有执行这些历史 study，也没有重新启动已经结束的容量实验。
`train_rollout_policy_dilution` 的新计划每个 job 提供 `recipe` 路径；labels/seed/out 必须匹配该 job。
旧 dilution 结果验证器保留历史命令解析，只核验旧结果，不启动旧 CLI。

## 面板配方

`recipes/panel_v1.json`、`panel_v2.json`、`panel_pressure.json` 保存 manifest、模型、价值网、
可执行文件/资源目录、每组种子数、逐任务覆盖、任务子集与压力/生成设置。
v2 直接读取 task27 已冻结清单，保留原任务/卡组/种子；pool 为每任务64种子。
此迁移不改任务族。面板 CLI 只留 recipe 与 out/rollout-policy/executable/resource-dir/workers 运维覆盖。
示例：

```bash
/home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/recipe/scripts/unified_panel.py --recipe /home/newbiexvwu/PvZ-search/.claude/worktrees/recipe/recipes/panel_v2.json --workers 2
```

本工作树需自行提供 `build/pvz-portable`，也可传 `--executable` 绝对路径。
任务子集与种子数只在配方修改；子集评估保留 `complete=false`。

## 指标字段与生命周期

`python/pvz_metrics.py` 只依赖标准库，后台线程立即采样、每30秒再采样、正常退出追加末样本。
默认路径 `/home/newbiexvwu/pvz-logs/metrics/<进程名>-<pid>.jsonl`。
UTC 换日时将上一日活动文件改名为 `<进程名>-<pid>.YYYY-MM-DD.jsonl`，继续写同名活动文件；
每天清理该指标目录中最后写入已超过7天的 JSONL。日志与模型均不进入 git。

| 字段 | 单位 / 解释 |
|---|---|
| time / timestamp / monotonic_s | UTC ISO8601 / Unix秒 / 单调秒；时间差用单调时钟 |
| name / pid / training / schema_version | 进程身份、是否训练、格式版本1 |
| rss_bytes / children_rss_bytes | 当前进程 RSS / 全部存活后代 RSS 之和；共享页会重复，不能当作物理内存用量 |
| mem_available_bytes | /proc/meminfo MemAvailable，字节 |
| pressure_memory / pressure_io | /proc/pressure/memory、io 的 some/full，avg10/60/300 与 total 原值 |
| io | /proc/self/io 的累计 rchar/wchar/syscr/syscw/read_bytes/write_bytes/cancelled_write_bytes；只属于当前进程 |
| cpu_percent / tree_cpu_percent | 相邻样本 CPU 时间差 / 墙钟；单核100%，进程树值不包含采样间已经退出的子进程 |
| gpu | 可选 nvidia-smi 的设备 index/uuid、utilization_percent、memory_used_bytes/memory_total_bytes；设备全局值，不冒充进程独占利用率 |
| progress / rates | 调用方累计计数及相邻样本每秒增量；训练 step/rows，采集 games/decisions，课程 games/decisions/round，面板 games/tasks |
| gpu_active_ratio_10m / gpu_observed_seconds_10m | 最近10分钟中 GPU利用率>50%的时间比例与已观测时长 |
| level / message | 连续覆盖完整10分钟且有效比例<50%时写 WARN；持续低效每10分钟重复一次 |

缺失 /proc 或 nvidia-smi 的字段为 null；GPU未知或采样中断超过60秒打断告警窗口。
训练是否有效仅按任务规定的设备 GPU 指标判断，CPU 训练也会如实记录设备读数。

采集父进程在 worker 池启动后开指标线程；worker 通过每任务独占的无锁共享槽汇报决策（每32决策刷新，
局末补余数），整局 C++ 纯学生接口在局末汇报。父线程汇总进程树 RSS，避免每个 worker 启动 GPU探针。
课程和面板另记控制进度，三个训练入口都记录 step 与 rows。

## 报告器与验收

```bash
/home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/recipe/scripts/metrics_report.py /home/newbiexvwu/pvz-logs/metrics
```

报告器按 name/pid 合并日分段，按秒加权统计 GPU有效时间占比、磁盘 read_bytes/write_bytes 增量、
MemAvailable最低点、进程树 RSS最高点、WARN数与末尾进度；容忍写入中的不完整 JSON 行。
GPU无观测时比例为 null，避免将“看不到GPU”当作低效。磁盘数只统计当前进程，不含采集子进程的写盘。

真实单 worker 采集冒烟摘要（两次采样，3.112秒）：

```json
{"name":"search_play","samples":2,"training_effective_ratio":0.0,"disk_read_bytes":5060,"disk_write_bytes":299008,"minimum_mem_available_bytes":23005356032,"peak_tree_rss_bytes":890720256,"warnings":0,"progress":{"games":1,"decisions":1086}}
```

该样例是短时 CPU 纯学生采集，GPU为0符合预期；不能用它判断长训练有效率。
v2 面板真实单任务冒烟1/1，结果 complete=false；课程现有11项、面板7项、数据行6项、
schema6项、identity训练5项、指标6项测试已分别通过。
指标测试包括真实 /proc、自有子进程 RSS、时间加权摘要、10分钟 WARN写入、轮转与7天保留。

200步对拍工具：`scripts/verify_recipe_migration.py`。读取记录中的完整原始 CLI 与完整语料，
对照版本 `6fdee2af`；[原始脚本来源核验](task38-source-audit.json)按历史计划SHA定位了原始版本，
其装载、batch_loss和去掉观测语句后的epoch循环与对照版本AST完全相同；不改 epochs/max_steps/批大小/种子或余弦长度，观察200个损失后中断。
主线合入后只构建一次完整紧凑训练包，两次分别克隆可变行权重；历史侧恢复 float32 标签、int64 类别和完整候选价值矩阵，历史损失与优化器代码不变。复用前核对两次装载请求的全部语义参数相等；输出、缓存与只读原数据分开，训练包在本工作树内先写临时文件再原子发布。
原运行的特征缓存先复制到私有目录（原子创建、不覆盖、与原文件不共享 inode），缓存摘要照常校验，避免重算。
每批检查没有其他训练进程，遇到其他训练只停止本次对拍。没有游戏进程。

**200步对拍状态：重启前 dw4i 记录为失败（观察回调与逐行报告同名，尚无200步结果）；旧产物作废，三份均待重跑。**


## 课程训练包（主控 10:10 追加）

已合并 `origin/pvz-env`（29fd6896）。课程控制器默认 `--recipe` 指向
`recipes/dw4i.json`，与默认采集学生 dw4i 对应；更换学生时显式传对应训练配方。
删除全部八个 `--pack-*` 参数（含来源标签参数）。基础标签来自配方，与队列已写标签取并集。
`pvz_recipe.training_row_options` 是训练 load 和课程 pack 唯一的行参数解析函数，
`pvz_training_pack.pack_flags` 统一生成打包命令及缺包诊断。
`aux_horizon` 仅在辅助损失开启时进入包键；`cells` 由格子软标签权重决定。
测试实际解析 pack 命令、截获 trainer.load 请求，核对三份历史配方以及 context/aux/cells 变体的全部包键。
合并回归还修正 CandidateStore 补齐项的格子索引 NaN：恢复原格式 `[-1,-1,NaN]`，防止训练 gather 越界。


新合入的 `run_dw5_study` / `run_task27_pretraining` 也已迁移：训练与 v1/v2 参考面板先冻结配方再执行；
`task27_common.ROOT` 从当前脚本定位，避免迁移后的执行器写回 dw5 工作树。
面板 `version` 纳入 schema，v1/v2 JSON 显式保存；历史 study 未重启。
对拍回调改名为 `loss_observer`，不再遮蔽历史训练函数内部的来源行 `observer`；
12档案/84行的回归在来源行报告开启下实际跑满两侧200步，损失差0，验证中断点及回调分离。

完整语料验收另核对历史 `metrics.json` 行数：dw4i 13,643,901、dw5i 15,788,614、identity 11,051,782；
有 `source_rows.json` 的前两份还逐来源核对档案数、行数及训练/验证划分。
该检查可在已启动的dw4i任务完成后独立执行，无需重启正在运行的对拍。

### 合并后缓存签名回归修复

主线给 `pvz_student_observation.py` 和 `packet_features` 增加兼容标记注释，
但现有缓存签名包含完整模块 / 函数源码，注释也会使缓存失效。恢复观测模块历史原文，
把卡特征的标记放到函数外（obs 标记保留在 frame_store.encoder_signature 旁），不改变编码行为。
v1 / v2lite / v2 与 packet 签名已固定为历史值；回归实际构造历史摘要缓存并验证不重算、不重写。
错误签名任务已主动终止，其产物作废；修复后的任务在 `artifacts/task38/parity_frozen/` 重跑。
这项修复应随本分支合入主线，否则现行包准备也会付出无意义的全量重算成本。
