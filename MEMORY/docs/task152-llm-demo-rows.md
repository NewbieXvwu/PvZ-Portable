# task152：GPT Pro 对局本身的动作，有没有进训练行？

主控 2026-10-29 定。Mac 上做，先调查、再实现，分支 `llmrows`，以 `origin/prodint16` 为底。

## 背景

- task105 已导入 1576 份 LLM 存档（GPT Pro 两轮加三份较弱模型的产物，通关 812 个种子），10:25 有 246 格作为**课程起点格**进了生产：学生从示范对局的中途局面接着打。
- 不清楚的是：这些存档里模型自己的操作序列，是否也作为监督训练行进了训练包？task132 提过“示范行（`demo_*` 任务的人类操作）也只是普通行”，但 LLM 导入链路（`import_llm_run.py` → adaptive → register）是否产出训练行，没有人确认过。
- task148 实测：在 GPT Pro 能稳定通关的屋顶卡组上，学生 1/112、老师 5/112。如果 GPT Pro 的整局操作能当示范行学，可能比只拿起点格更直接。

## 要做的

1. 查清楚并写明：生产训练包（`pvz_training_pack.py` / `pvz_training_rows.py` / 配方里的数据源）读哪些对局。人类示范、LLM 导入存档、起点格接续局，各自是否产生训练行，靠什么元数据判定。给出代码位置。
2. 如果 LLM 存档的整局操作没有进训练行，设计并实现一个可选开关，默认关，旧配方逐位不变，把这些存档作为示范行加入训练包：
   - 只用 deploy14 引擎 verify 过的通过存档（task105 账本里有）；
   - 放弃局和输局默认不进，可配置；
   - 和 task132 的 `demo_row_weight` 组合，先写清楚乘权顺序；
   - 存档只在台式机上，所以实现要能在台式机打包时读取 `/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport/artifacts/task105/import/sha256/<sha>/`。路径做成配方参数。
3. 小测试：默认关时张量与权重逐位不变；打开时只多出预期的行。按 AGENTS §6.5 第 8 条：只用 `-k`，加 `timeout 300` 和 `ulimit -f 2000000`，同一时间只开一个 pytest。
4. 写一个实验配方草稿（`recipes/` 下新文件），并估算台式机重打包的时间与磁盘。

## 约束

- 不训练、不部署，不连台式机生产。
- 不加自校验。
- 结论写在本文件末尾，推送 `llmrows`。

补充：已有分支 `demorows` 是 task104 的示范行研究（“four-draw study … paired 992-game verdict”），先读它的结论（`origin/demorows` 的 MEMORY/docs/task104*），别重复做。

## 调查与交付（2026-10-28，Mac）

工作分支为 `llmrows`，基于 `origin/prodint16` 的 `1cc6941c`。已读 `origin/demorows` 的 task104 全文：当时增加 15,419 条人类决策和 16,623 条 LLM 决策，四次训练、992 局评估未证实收益，泳池平均下降 35.42 个百分点。本任务只接入 task105 的后续大批量导入档案，没有重复训练或评估，也没有把“能够加入训练”当成收益结论。

### 现行训练包怎样选对局

`python/pvz_frame_store.py:16` 只发现数据根目录两层下的 `frames.jsonl/frames.npz`。`scripts/curriculum_queue.py:1025` 将历史已完成采集组的标签与配方显式 `labels` 合并，`pack_command`（1034 行）将同一行合同交给打包器。`python/pvz_recipe.py:training_row_options` 决定观察版本和采样等参数，`scripts/pack_training_rows.py:build_from_archives` 从第 i−1 帧特征预测第 i 帧标签或实际动作，优先采用已有 teacher 标签。`python/pvz_training_pack.py:training_rows` 只解释元数据里的显式决策索引，本身不会发现任何额外原始示范目录。`python/pvz_training_rows.py` 的来源统计与权重回调也不负责增加数据源。

人类示范只有在导出到训练数据根目录、且其标签被配方选中后才产生行。prodint16 的 `scripts/human_demo.py:284` 导出完整帧及元数据，未写显式决策索引，因此其等待行沿用普通采样合同。task104 分支已经补过“保留全部成功 plant/wait 决策、保留完整历史”的导出方式，本任务复用相同决策选择合同，没有全局更改人类导出器或已有配方。来源依赖 `meta.json.label_source`，人类为 `human`。task132 加权依赖这些元数据，不依赖 `demo_*` 名称。

LLM 原始档案保存在 task105 的内容地址目录中，**原始整局操作没有自动进入生产训练包**。prodint16 的训练数据发现与标签选择均不读取该目录。真实的后续链路位于 `origin/llmimport`：`scripts/llm_import_sha.py:133` 将单份档案复制到 `import/sha256/<sha>/`，88–96 行生成逐 SHA 的验证回执，187–211 行保存导入记录；`scripts/task105_import_batch.py:41` 按 SHA 合并到 `import/merged_progress.json`。`scripts/task105_pipeline.py:117` 调用 adaptive，120 行调用 register；`scripts/takeover_scan.py:703` 登记的是任务清单里的 `demo_start`。这些步骤保存原始轨迹并登记接手局面，不会把其全部动作导出为训练行。

起点格接续局只有在后续采集完成、标签进入训练快照后才产生训练行。其 `label_source=teacher`，且元数据带 `demo_archive`，监督标签来自搜索老师，与 LLM 自己执行的整局动作分别统计。生产配方 `recipes/dw4i2_v2timer.json` 只列基础采集标签，队列再增加历史采集组。本文依据已提交代码和 task104/task105 结论判断数据流，没有连接台式机读取最新训练包，因此不报告当前生产包中未经读取确认的行数。

### 可选接入与权重合同

实现提交 `4ddf8533`。新增配方字段 `llm_demo_import`，默认 `null`，填入 task105 的 `import` 目录才开启。该目录下读取 `merged_progress.json`、`receipts/<sha>.json` 和 `sha256/<sha>/`。`llm_demo_results` 默认 `[1]`，仅选择通关。`[1,2]` 可包含输局，`[0,1,2]` 再包含中途结束的局。三种结果始终要求导入账本通过、回执 `verified=true`，并且回执的引擎指纹严格为 deploy14 的 `76ff36dc026f67641aefccd5f30f6cdae65eee14ee197dc24497413123a7874a`。验证失败和其他引擎的档案均不选入。相同 SHA 只选一次，同任务同种子的不同 SHA 保留为不同轨迹。

`python/pvz_llm_demo_rows.py:demo_paths` 在训练数据目录的 `.llm_demo_rows/<sha>/seed_<seed>/llm_demo_decisions/` 生成数值化副本，原始档案及回执保持原样。只生成成功的 plant/wait 监督索引，初始帧不作标签，拒绝操作与铲除等不作标签。所有帧仍完整保存并参与历史编码，示范等待行不再下采样，也没有 1/wait_keep 补偿。首次导出按档案逐份处理并原子发布，后续直接复用已导出的内容地址目录。没有重放、训练前全量复验或新自校验。

新增来源并入既有不可变标签段和小索引。关闭时 pack_options 不增加字段，原指纹、采样顺序与张量权重保持原值；打开时只重建新增示范段，已有标签段继续复用。队列的 pack_flags 和训练器 load 使用相同路径及结果配置。配置路径不参与共有段身份，其他档案不会因开关失效。

prodint16 尚未包含 task132，因此先复用该任务原实现 `63d2fa41`，本分支对应 `6bb0686c`。`demo_row_weight` 继续作用于 `label_source=human`，本次新增的 LLM 示范另以 `llm_demo_sha256` 标记，并使用同一倍数。未通过本开关纳入的旧 LLM 行维持 task132 原行为。`demo_start_row_weight` 仍只作用于 teacher 来源且带 demo_archive 的接续局。

乘权顺序为等待采样补偿与 label_weight → source_weight → packet_balance → spread → row_weights → 关卡平衡 → demo_row_weight/demo_start_row_weight。新 LLM 示范默认 label_weight=1、等待补偿=1，来源原值保留，可配置 `source_weight.llm_gpt_pro` 或 `source_weight.llm_agent`。例如来源倍数 0.5、demo_row_weight=4 时，在其他倍数均为 1 的情况下行权重为 2。最终示范倍数不参与前面的上限，也不重新归一化，验证行同样加权。`level_shares.json` 仍描述最终示范倍数之前的阶段。

### 小测试

Mac 串行执行一个 pytest，带 `-k`、`timeout 300` 与 `ulimit -f 2000000`，7 项通过，耗时 1.49 秒。默认关闭时逐位对比旧张量及权重，开启时合成两份通过胜档恰好多 12 行，已有段复用，拒绝行及不支持的动作排除，完整帧历史保留。另覆盖 SHA 去重、同种子不同档案、输局/中途结束配置、非 deploy14 或失败回执排除，以及 task132 乘权和队列/训练器参数一致。测试只使用小型合成数据，未启动游戏、训练或部署。

### 实验草稿与重打包预算

新增 `recipes/dw4i2_v2timer_llm_rows_task152.json`，以 deploy16 A 的 v2timer 配方为草稿，导入路径为 `/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport/artifacts/task105/import`，对应原始存档正是题目给出的 `sha256/<sha>/`。结果过滤保持 `[1]`，两个示范倍数均为 1，以便先比较单纯增加示范行的效果。数据、输出和可写缓存均指向 llmrows 私有产物目录，已有特征缓存只作只读来源。该私有基础数据视图尚未创建。正式实验的 A/B 必须使用同一冻结轮次的基础标签与不可变段，草稿里保留的是 A 配方的静态基础标签，未伪装成最新生产快照。

未连接台式机，以下为预算外推。task105 已提交结论有 1,246 份通关存档候选，812 是通关种子数，两者不可互换。以 task104 的 12 份 LLM 胜档共 16,623 行外推，新增约 173 万行。按 task143/task145 的 21.1M 行段体积 11.2 GiB，新增段约 0.9 GiB；数值帧副本及冷特征缓存再预留约 0.5–2 GiB，打包阶段新增磁盘合计约 1.5–3 GiB。实际行数与体积随档案长度变化。

已有基础段与观察版本不变时，无需重建整份生产语料。首次处理这 1,246 份新档案，按既有帧写入约 0.63 秒/档及冷编码的历史速率，预估 30–90 分钟。全部新示范归为一个标签段，标签并行不会将这部分时间简单除以 worker 数；生产负载和缺失的基础缓存会延长耗时。重复打包只读取小账本、回执及已有导出元数据，并复用新增段。训练器首次读取新组合时，现有合并机制还会在私有数据视图生成整份语料的 `.merge/shared-v1`，另需约 11–14 GiB，后续实验预留 20 GiB 空闲磁盘较合适。若基础 v2timer 段尚不存在，另计基础语料重打包时间，该成本不包含在新增示范的 30–90 分钟里。

本次只在 Mac 完成代码、合成小测试与文档草稿，未连接台式机生产，未读取或重打包真实 task105 存档，未训练或部署。目标分支为 `llmrows`。

### task205 合入主线

deploy26 复用此转换实现，转行接在统一导入复验通过之后，训练只读 task194 已转换并清理的 `.llm_demo_rows`，不在打包前重扫 task105 账本或重复重放。新增转换同样排除未遮挡雾关与空槽种植行。旧私有实验配方留在 `origin/llmrows`，主线使用 task205 的 A 重新配比及 B 干净谱系配方，见 task204 第 11 步。
