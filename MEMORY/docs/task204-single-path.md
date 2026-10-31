# task204 单一入口审计

- 日期：2026-10-31
- 生产基线：`origin/deploy25` @ `97994745`（本工作树 `singlepath`）
- 旁支：`origin/llmplay-decks` @ `3c43bc13`、`origin/llmimport` @ `a06b9edf`
- 方式：只读 `git show` / `grep`；未改代码、未跑测试、未触碰生产。

## 0. 结论（先看这里）

体积问题在“写”和“导入”两处，不在“读”。读端本来就能同时读 jsonl 与 npz。

- **写端有三条路**：`search_play`（npz）、`human_demo.Recorder`（jsonl）、`llmplay.Game.export_archive`（deploy25 基线上是 jsonl）。
- **导入端有两条链**：`import_llm_run`（原样 `copytree`）与 `llm_import_sha`（原样复制 `FILES` + SHA 收据），两条都**不在入库时做 JSONL→NPZ 转换**。
- **task203 已经动过手，但修在两个旁支、没有合流**：`llmplay-decks` 把导出改成 `pvz_frame_store.write_frames(..., cache=False)`；`llmimport` 让 SHA 收据在外部原地转换后不失效。deploy25 基线本身仍是旧行为，所以“双入口”在代码库层面依旧存在，只是其中一半被搬到了旁支。
- 真正意义上的一条路应当是：**`pvz_frame_store` 是唯一的帧容器**，所有生产者写 `frames.npz`，所有消费者用 `frame_path/read_frames`，导入在入库时转换，旧档只读兼容。
- **按 §6.6 复核新增 6 项**（`origin/pvz-env:AGENTS.md`）：老师搜索树、`StudentHistory`/太阳历史、C++ 事实重算、常驻 Python 参考实现、未采纳的 `v2fogstack4` 原型、两个 `profile_for_deck`；部署脚本复制直接违反第 4 条。详见 §0.5。

下表按“对体积 / 正确性 / 维护的危害”排序。P0 是体积，P1 是会产出错误数据或错误晋升，P2 是维护成本。

| 序 | 重复项 | 危害 | 现行实现数 | 建议保留 |
|---|---|---|---|---|
| 1 | 存档写入：npz vs jsonl | 体积（GB 级） | 3 | `pvz_frame_store.write_frames`（npz） |
| 2 | 导入链：legacy / SHA / batch / live / installer | 体积 + 维护 | 5 | 单一导入器，入库即转 npz |
| 3 | `meta.json` 同名 schema 四种形状 | 正确性 | 4 | 一种 schema，版本字段显式 |
| 4 | 对局驱动 / 录制 / 复演 | 正确性 + 维护 | 3 | `human_demo` + `pvz_env` 为唯一录制层 |
| 5 | 任务定义与卡组推导 | 正确性 | 3+ | `pvz_tasks` + `render_episode` 为权威 |
| 6 | 门禁 v6 / v7 与 `select_student` | 正确性（晋升判定） | 2 套统计 + 1 套预测 | v7；v6 显式标记 legacy |
| 7 | 部署脚本 `qc_deploy17..25` | 维护（违反 §6.6 第 4 条） | 9 | 一个参数化启动器 |
| 8 | `recipes/` 旧配方 | 维护 | 14 份，3 份在用 | A / B / warm 三份 |
| 9 | 观测编码 Python vs C++ | 维护（训练侧允许） | 2 | 训练侧留；生产逐步 history 迁 C++（S1-2） |
| 10 | 场景文本渲染 | 维护 | 3 | 共享渲染器 |
| 11 | 任务生成器 / panel 生成 / llmplay 分支家族 | 维护 | 多 | 一个生成器、一条 llmplay 分支 |

---

## 0.5 §6.6 复核：C++ 权威与 Python 待迁移清单

依据 `origin/pvz-env:AGENTS.md` §6.6（2026-10-31 用户定；deploy25 基线的 AGENTS.md 尚未含此节，但已作为项目约束生效）。本节把原清单按新标准再过一遍：

- **合规薄接口**：只调用 C++ 的 `reset/step/observe`、示范重放、调试分析、训练（PyTorch）。
- **待迁移**：实现已被生产采纳，但权威代码仍在 Python。
- **重复实现**：Python 里又算了一遍 C++ 已有的规则、特征、搜索或推理。

判定原则：C++ 是游戏规则、观测特征、老师搜索、推理的权威；Python 的 PyTorch 侧和调试工具可以保留，但不能成为生产推理/搜索的第二实现。原型被采纳要同批移入 C++，没采纳就删。部署脚本不复制粘贴。

### S1 生产已采纳、Python 版仍在（按危害）

| # | 项 | Python 位置 | C++ 对应 | 生产用法 | 结论 | 迁移代价 |
|---|---|---|---|---|---|---|
| S1-1 | 老师搜索树 | `scripts/search_play.py:_search`、`_continuations`、`_halving_choice`、`_save_value`、`_save_probe`、`_save_plan`、`_verified_choice`（约 283–664 行）；`_leaf_values`/`_static_scores` 已调用 C++ | 没有整棵搜索命令，只有 `BRANCH_SNAPSHOT_FAST`、`ROLLOUT(_VALUE)`、`LEAF_VALUES`、`STATIC_SCORES`、`SIM_SKIP` 原语 | qc 的 `fast_search` 每个决策都走；`--native-games` 不影响 | **违反 §6.6 第 2 条**：候选生成、每卡取格、减半、验证选择、预算/剪枝都在 Python。把搜索树移入 C++（新 SEARCH/PLAN 命令），Python 只发请求、记结果 | 高 |
| S1-2 | StudentHistory / 太阳收入历史 | `python/pvz_student_observation.py:StudentHistory`（含 `extra`/`features`）；`search_play._fast_action` 每步更新；`pvz_env._student_options` 序列化 `V2H`；`pvz_env.rollout` 传 `_sun_production_history`；`pvz_env._annotate_observation` 算 `sun_income_rate` | `src/EnvironmentRollout.h:History`（注释明确“mirrors the Python StudentHistory”）、`Update`、`Features`、`Predict`；`main.cpp:ReadStudentOptions` 解析 `V2H`；`ROLLOUT` 内部也按 history 算 income | `--native-games` 的 `ROLLOUT_PREDICT` 与 `ROLLOUT` | **Python 重算 C++ 已能维护的历史**。让 C++ 持有跨命令 history；训练侧 encoder 可留，但生产逐步接口不再由 Python 算/传 | 中高 |
| S1-3 | 事件 / 波次 / 损失原因 / 卡组校验 / 等待语义 | `llmplay/game.py:_snapshot`、`_mower_states`、`_note_events`、`_loss_reason`、`_close_wave` 从公共观测反推事件（`info["events"]` 只用于 totals）；`scripts/pvz_tasks.py:terrain_covered` 与 `llmplay/game.py:validate_deck` 重写地形覆盖；`python/pvz_wait_events.py:public_state/triggers/reference_wait` 是等待语义的 Python 参考 | `LawnApp::EnvironmentDeckWarnings`（`pool`/`roof`/`upgrade`）、`EnvironmentWaitEvents`、native `info["events"]`/wave/defense | GPT 包 `Game`；任务生成；`pvz_event_env` 校验 | 事件叙述与卡组校验应消费 C++ 输出；wait 参考按第 3 条只留一次性对比。`Game` 不是生产主路径，优先级低于 S1-1/2；`validate_deck` 可直接改调 `env.deck_warnings`；`terrain_covered` 若必须在无引擎的生成期使用，保留但注明是任务设计预筛，不是第二权威 | 中 |
| S1-4 | 常驻 Python 参考实现与等价测试 | `scripts/search_play.py:_python_rollout_reference`（注释 LEGACY，仅 test/perf）；`python/pvz_wait_events.py:reference_wait`；测试 `test_cpp_rollout`、`test_identity_cpp`、`test_value_encoder`、`test_sleep/timer/bungee_inputs`、`test_wait_events` | `EnvironmentRollout`、`EnvironmentValueNet`、`EnvironmentTokens`、`EnvironmentWaitEvents` | 仅测试/诊断 | 与 §6.6 第 3 条、§6.5 冲突：移植只做一次小对比，不常驻重复校验。建议删除完整 Python oracle，保留少量固定 fixture 的 golden 测试 | 低 |
| S1-5 | 未采纳的 Python 原型 | `v2fogstack4`：`pvz_student_observation.StudentHistory` 的 `zombie_frames` 记忆通道；`pvz_rollout_policy.export_student` 明确抛“Python memory experiment; native export is not implemented”，但仍列在 `pack_training_rows --obs`、`pvz_recipe` 的可选值，并有 `test_fog_memory.py` | 无（C++ 未实现） | 训练可选、测试 | 未被 C++ 采纳的 Python 原型；若实验仍在进行，明确保留期限，否则按第 3 条删除/归档，不再出现在运行代码/CLI 选项 | 低 |
| S1-6 | 两个 `profile_for_deck` / `training_task` | `python/pvz_env.py:profile_for_deck` 硬编码升级集合 `40 <= type <= 47`；`scripts/pvz_tasks.py:profile_for_deck` 从 `pvz_constants.upgrade_plants()` 解析；`pvz_env.training_task` 与 `render_episode.training_task_kwargs` 重复 | `Plant::IsUpgrade`（经 `pvz_constants` 解析）、`LawnApp.cpp` 的 profile 校验 | `pvz_env` 版本只在 `test_boss_l50`、`test_bungee_inputs`、`test_cpp_rollout`、`test_deck_warnings`、`test_env_protocol` 使用；生产走 `pvz_tasks` | 删除 `pvz_env` 版本，测试改用权威实现；硬编码 40..47 与 C++ 清单有漂移风险 | 低 |

### S2 Python 重算 C++ 已有逻辑（训练侧，允许但需标记）

这些不是违规，因为 §6.6 明确允许“训练（PyTorch）”。但它们确实是同一逻辑的第二份实现，必须满足两个条件：只服务训练/数据准备，不得成为生产推理或搜索的后备路径；训练侧与 C++ 的定义要保持可验证的一致。

| Python 位置 | C++ 对应 | 用途 | 判定 |
|---|---|---|---|
| `pvz_value_net.featurize/encode_frame`、`pvz_student_observation.StudentHistory.extra`、`pvz_frame_store.encode_frames` | `EnvironmentValueNet::Features`、`EnvironmentRollout::Features` | 训练帧编码 / 训练缓存 | 允许的训练侧；`encode_frames` 在 `search_play` 写档时使用，属于训练数据准备，不是推理 |
| `pvz_rollout_policy.RolloutNet.forward/act/ranked_plants` | `EnvironmentRollout::Predict` | 训练、非 native 诊断/A-B | 允许训练；non-native `search_play` 只应是实验/诊断，不能进 qc |
| `pvz_token_observation`、`pvz_student_tokens`、`pvz_token_blocks` | `EnvironmentTokens`（`OBS_V3`） | token 训练拼装 | 允许训练；生产 token 由 C++ 发出，Python 只做 collate |
| `pvz_env._annotate_observation` 的太阳收入窗口 | `ROLLOUT` 内 income 计算 | 逐步接口 | 见 S1-2，应并入 C++ history |
| `pvz_tasks.deck_for_level` / `random_deck_for_seed` | 无对应 C++（任务设计） | 任务/卡组定义 | 保留一份 Python 实现即可；不要另写第二份 |

### S3 判定为合规的清单（不要误删）

- `python/pvz_env.py` 的协议封装：薄接口。
- `python/pvz_event_env.py`：native wait 的薄适配器。
- `scripts/human_demo.py:replay/verify`：示范重放/校验，§6.6 明确允许。
- `scripts/episode_query.py`：调试与分析。
- `scripts/render_episode.py`、`scripts/llmplay/scene.py`：训练报告/调试渲染（渲染器本身应合并，见第 10 项）。
- `scripts/llmplay/game.py` 的 `plant/shovel/wait/wait_until/observe/summary`：外部模型逐步对局接口，允许；其中 `_note_events` 等推导不属于薄接口，见 S1-3。
- `curriculum_queue` / `promotion_gate` / 导入编排：不是 C++ 模拟逻辑，但“同一件事只一份”仍适用（第 2、5、6、7 项）。

### S4 规则 4：部署脚本

`scripts/qc_deploy17.sh` … `qc_deploy25.sh` 是九份整份复制，直接违反 §6.6 第 4 条。应改为一个参数化启动器，每版文件只保留版本常量、recipe、panel、引擎路径；`start_qc` 一处来源，看门狗与部署共用。详见第 7 项。

---

## P0：体积

### 1. 对局存档的写入路径有三条

**实现 A（生产）**
- `python/pvz_frame_store.py:write_frames` / `write_frames_from_source`：唯一容器格式 `frames.npz`（`ZIP_LZMA`，内含原始帧 JSON 字节 + 预计算特征数组 + 编码器签名）。`read_frames` / `frame_path` / `frame_bytes` 同时接受旧 `frames.jsonl`。
- `scripts/search_play.py:_play`：落盘 `write_frames_from_source(archive / "frames.npz", ...)`。
- `scripts/human_demo.py:export`：`write_frames(target / "frames.npz", read_frames(frame_path(archive)))`，这就是既有的明文转 npz 路径。
- 体积依据：`e75668ee`（task32 第 16 步）把 6658 份 JSONL 转 npz，为原体积 **7.50%**，失败 0。

**实现 B（GPT 包，deploy25 基线）**
- `scripts/llmplay/game.py:Game.export_archive` 写 `ticks.jsonl` + `frames.jsonl`；`load_archive` 自己逐行 `json.loads` 读 `frames.jsonl`。
- `scripts/llmplay/delivery.py:FILES = ("meta.json","ticks.jsonl","frames.jsonl")`，`_archive_digest`、`export_bundle`、`_store/_restore_archive` 都按这三个名字走。
- `scripts/llmplay/README.md` 当时明确写着“frames 仍是原 JSONL，现有导入侧无需更改”，是有意分叉。

**实现 C（半官方）**
- `scripts/episode_query.py` 在采集 / 反事实分支里仍写 `frames.jsonl`，与 `pvz_frame_store` 注释里“text archives are read, never written”直接矛盾。

**谁在用**
- A：生产采集（`search_play`）、人工示范导出转换、训练数据打包、全部测试。
- B：GPT Pro 交付包（`llmplay-decks` 构建），以及导入库里的原始存档。
- C：`episode_query capture` 的临时存档；`search_episode.py` 走它。

**行为差异**
- 体积：jsonl ≈ 13 倍 npz（按 7.5% 反推）。
- 内容：`write_frames` 默认附带编码缓存；GPT 接口包不需要训练特征，task203 新增 `cache=False` 只存无损原始字节。
- 兼容：`frame_path` 优先 `.npz`，旧 `.jsonl` 仍可读，不会静默读错。

**task203 已做的**
- `3c43bc13`（llmplay-decks）：`export_archive` → `write_frames(..., cache=False)`；`load_archive` → `frame_path/read_frames`；`delivery.FILES` 同时列 npz 与 jsonl；打包 npz 用 `ZIP_STORED`；`build_package` 增加 `pvz_frame_store.py`。
- 但它**没有**把 `FILES` 收窄成只认 npz，也**没有**合入 deploy25。

**建议保留**
- 只保留 `pvz_frame_store`。`human_demo` 与 `llmplay` 都调用它，不再有自有写入器。`episode_query` 的临时存档也改用 `write_frames`。
- 导入时对旧 jsonl 做一次性转换，之后删除明文。

**迁移代价**
- 低到中。第一步合并 `3c43bc13`；第二步把 `FILES` 改为 `("meta.json","ticks.jsonl","frames.npz")`（读旧档时回退 jsonl）；第三步把 `episode_query` 两处写入 swap；第四步对导入库跑一次转换并校验 SHA 收据（`a06b9edf` 的用例就是为这一步写的）。

### 2. 导入链有五层，且都不做格式转换

**五层实现**
1. `scripts/import_llm_run.py:import_archive`：`shutil.copytree(archive, target)`，原样复制，含 `frames.jsonl`；`archive_problems` 还按 jsonl **行数**校验 `frame_count`。
2. `scripts/llm_import_sha.py:run_sha`（仅 `origin/llmimport`）：内容寻址到 `dest/sha256/<sha>`，`FILES = ('meta.json','ticks.jsonl','frames.jsonl','frames.npz')` 原样复制，靠 SHA 收据跳过重复复验。
3. `scripts/task105_import_batch.py`（仅 `origin/llmimport`）：串行批处理，读 `llm_import_sha.FILES`。
4. `scripts/pvz_live_import.py`：monkeypatch `scan_plan` 关闭离线扫描，`main_with` 优先调 `run_sha`，`merge_rows` 写 `merged_progress.json`。
5. `scripts/deploy22_import.py:install`：**就地重写已安装的 `import_llm_run.py` / `task105_import_batch.py`**，再复制 `pvz_live_import.py`。部署源与仓库源因此可能不一致。

**旁路**
- `scripts/import_mac_labels.py`（仅 `origin/llmimport`）：从 HF 取 Mac 标签，直接写共享语料 `artifacts/student/frames_v2`，不走上面任何一层。它做的是“把帧送进语料”，与导入器职责重叠。

**谁在用**
- 生产 qc：`$L`（llmimport 工作树）上用 `deploy22_import.install` 装入 `pvz_live_import`，后者调 `run_sha`，索引由 `pvz_live_curriculum` 读取。
- 人工批量：`task105_import_batch`。
- 兼容/测试：legacy `import_llm_run.run`。
- 注意：qc_deploy25 定义了 `IMPORT_FILES=(import_llm_run.py task105_import_batch.py pvz_live_import.py)`，但 `task105_import_batch.py` 在 deploy25 基线上**不存在**，只在 `$L` 的 llmimport 分支上。文件清单与真实来源已经脱节。

**行为差异**
- legacy 有扫描计划、会安排 take-over 局；live 适配器把扫描全部关掉。
- SHA 层会因“文件大小不一致”拒绝复放；task203 之前，旧收据在原 jsonl 被转成 npz 后会被判为“大小不符”。
- legacy 校验的是 jsonl 行数，SHA 校验的是整档哈希，两者对同一份档的口径不同。

**task203 已做的**
- `a06b9edf`（llmimport）：`run_sha` 识别“源仍是 jsonl、目标已是 npz”的转换瞬态，跳过大小校验以保留收据；新增 `test_cached_jsonl_receipt_survives_task32_inplace_npz_conversion`。

**建议保留**
- 一个导入器、一个索引。保留 SHA 收据（它是唯一防止重复复验的机制），把 legacy 的扫描计划降为可选策略；把 `pvz_live_import` 的“关扫描”变成参数；安装器只复制文件，不重写源码。

**迁移代价**
- 中。需要合并 `a06b9edf`、去掉 `deploy22_import.install` 的源码改写、把 `merged_progress.json` 与 `import_progress.json` 合为一个索引（`pvz_live_curriculum.py` 当前两个都读）。转换已有数据是离线一次性操作。

---

## P1：正确性

### 3. `episode_archive_v1` 被四种不同的 `meta.json` 形状共用

**四种形状**
- `scripts/human_demo.py:Recorder`：`environment_task`（含 `profile`、`preplanted`）、`task`、`deck`、`frame_count`、`operations`、`totals`、`outcome`。
- `scripts/llmplay/game.py:export_archive`：在上面之外还有 `cycle`、`workflow_layer`、`opening`、`run_id`、`environment_fingerprint`、`deck_reason`、`strategy`、`deck_review`、`abandoned`、`robustness`、`manual_patch`、`engine_sha`、`context`。
- `scripts/search_play.py` 的 `meta.json`：**没有 schema 字段**，键是 `label_source`、`task`、`context`、`outcome`，胜负词表是 `win/not_won`。
- `scripts/episode_query.py:SCHEMA`：自称 `episode_archive_v1`，但字段是它自己的 `task/outcome/totals/frame_count`。

**谁在用**
- `import_llm_run.discover` 用“有 `meta.json` + `ticks.jsonl` + schema 匹配”筛选；`llm_import_sha` 用整档哈希；`pvz_live_curriculum` 读 `meta` 的卡组与结果；`episode_query load` 按自己的形状解。
- 结果是：同一 schema 名不能保证字段可互操作，任何消费者都得知道“这份档是谁写的”。

**行为差异**
- 缺字段是常态（例如 search_play 无 schema），下游靠 `.get()` 容错，错误会被静默吞掉。
- `outcome` 的键名与胜负词表不完全一致。

**建议保留**
- 一种 schema，显式 `schema` + `schema_version`；公共必备字段（`task`、`deck`、`outcome`、`totals`）固定，层/流程字段进入 `extra` 或独立 `llmplay.json`。search_play 补上 schema。

**迁移代价**
- 中。读端需要双读一段过渡期；写端（4 处）都要改。历史存档可以只在导入时补齐，不必批量重写。

### 4. 跑一局 / 录制 / 复演的语义有三份

**实现 A：生产学生循环**
- `scripts/search_play.py`：`_fast_action` / `_search` / `fast_state` / `native_root`，支持纯 Python、C++ 整局 `env.rollout`、原生根 + C++ 续走三种模式；帧由 `_frame_json` 统一产出。

**实现 B：人工示范录制**
- `scripts/human_demo.py:Recorder`（`_step`、`_frame`，60 tick 帧距）、`replay`（逐步校验 tick/ok，可反事实覆盖）、`verify`（独立复验）。

**实现 C：GPT 包**
- `scripts/llmplay/game.py:Game`：`_start`、`_reset_engine`、`_snapshot`、`_mower_states`、`_note_events`、`_loss_reason`、`_close_wave`、`_replay_to`、`plant/shovel/wait/wait_until`、`rewind`、`branch`、`robustness`、`observe`、`summary`。
- 它在同一个 `PvZEnv` 之上重写了录制的几乎所有语义：帧距、事件注记、快照/回退、分支、稳健性种子、终局原因。

**谁在用**
- A：生产 qc（`search_play`）与训练 rollout。
- B：导入复验（`import_llm_run` 的 `verify`）、反事实（`episode_query`）、人工示范导出。
- C：GPT Pro 包的全部对外 API。

**行为差异**
- 帧距与事件注记规则由各自维护，改动一处不会同步另一处。
- `Game.robustness` 与生产的 robustness 检查点来自不同实现（`required_checkpoint` 也在 llmplay 里重写了一份）。
- `Game` 的 `_replay_to` 与 `human_demo.replay` 对分歧的处理不同：前者是内部重放，后者会比对 `ticks.jsonl` 的每一步。

**建议保留**
- 录制/复演语义收敛到 `human_demo` + `pvz_env`；`llmplay.Game` 保留为薄 API（决策、快照、任务流转），不保留独立的落盘与复演实现。`search_play` 的搜索循环是性能路径，可以保留，但帧产出应复用同一函数。

**迁移代价**
- 高。需要把 `Game` 的 rewind/branch/robustness 搬成 `pvz_env` 能力或复用 `human_demo.replay`，并保持 GPT 包 API 兼容。建议先合并写盘（第 1 项），把这一步拆成后续任务。

**§6.6 复核**
- `Game` 对外的 `reset/step/observe` 属允许的薄接口；但 `_snapshot`、`_mower_states`、`_note_events`、`_loss_reason`、`_close_wave` 是从公共观测反推 C++ 已有的事实（`info["events"]`、wave、defense），不属于薄接口（S1-3）。先让 `Game` 消费 C++ 结构化事件；若 C++ 事件不够细，扩 C++ 输出，而不是在 Python 再推一遍。
- `_python_rollout_reference` 是常驻 Python 参考实现，按 §6.6 第 3 条只应保留到一次一致性对比为止（S1-4）。

### 5. 任务定义与卡组推导有三套以上

**实现 A：生产权威**
- `scripts/pvz_tasks.py`：`deck_for_level`（按地形插 16/43/9/33）、`profile_for_deck`、`terrain_covered`、`random_deck_for_seed`、`deck_for_seed`。
- `scripts/render_episode.py`：`load_training_task`（唯一的清单查找）、`training_task_kwargs`（任务→`TaskSpec`）。
- 面板生成：`make_day_panel.py`、`zone_panel.py`（`Family`/`Zone`/`task_document`/`check_decks`/`main_for`）、`make_fog_user_panel.py`、`make_fog_user_v2_panel.py`、`make_roof_panel.py`、`make_multiplier_tasks.py`、`unified_panel.read_panel`、`day_orig_tasks.py`。
- 清单：`configs/human_demo_tasks.json`；另有 `curriculum_queue.py` 对 live193 格子内联 `--task-json`。

**实现 B：GPT 包**
- `scripts/llmplay/game.py:task_spec`（任务→`TaskSpec`，但**跳过 `deck_for_seed`**）、`validate_deck`（重写地形覆盖校验）、`required_checkpoint`、`robustness_seeds`。
- `scripts/llmplay/decks.py:legal_cards` / `region` / `seed_at` / `attempt_seeds`，自带地形黑名单。

**实现 C：分支分叉**
- `origin/llmplay-decks:scripts/pvz_tasks.py` 新增 `player_deck_levels`、`deck_pool_for_level`，deploy25 没有。也就是说“权威卡组函数”本身也分叉了。

**谁在用**
- A：生产 qc、面板生成、门禁、训练。
- B：GPT 包的任务选择与卡组合法性。
- C：llmplay-decks 的 deck 模式。

**行为差异**
- GPT 的 `task_spec` 不复用 `deck_for_seed`，同一任务在不同入口可能拿到不同的卡组族；`validate_deck` 与 `terrain_covered` 的判定边界也可能不一致。
- 面板生成器各自拼任务 dict，schema 重复（`unified_panel._task`、`zone_panel.task_document`、`make_day_panel` 各一份）。

**建议保留**
- `pvz_tasks` + `render_episode` 是唯一权威；llmplay 直接 import，不再自带 `task_spec` / `validate_deck` / `required_checkpoint` / `robustness_seeds`。面板生成器收敛为一个共享 helper + 少量族定义。

**迁移代价**
- 中。先合并 `llmplay-decks` 的 `pvz_tasks` 增量，再把 llmplay 的四个函数换成 import。内联 `--task-json` 与清单需要保留，但都走同一 `load_training_task`。

**§6.6 复核**
- `terrain_covered` 与 C++ `LawnApp::EnvironmentDeckWarnings` 的 `pool`/`roof` 判定重复，`llmplay.game.validate_deck` 又重写一遍（S1-3）。任务生成期没有引擎时保留 `terrain_covered` 作为设计预筛，但 reset 前应以 C++ `deck_warnings` 为权威；`Game.validate_deck` 直接改调它。
- `python/pvz_env.py:profile_for_deck` 是 `scripts/pvz_tasks.py:profile_for_deck` 的第二份，且把升级植物硬编码成 `40 <= type <= 47`，与 `pvz_constants.upgrade_plants()`（从 `Plant::IsUpgrade` 解析）有漂移风险；它目前只被测试使用，应删除并让测试改用权威实现（S1-6）。`pvz_env.training_task` 同理。

### 6. 门禁 v6 / v7 并存，`select_student` 仍按 v6 预测

**实现**
- v5：只剩面板与监控任务名（`python/pvz_promotion.py:V5_PANEL`、`V5_MONITOR_TASKS`），没有独立 gate 代码。
- v6：`python/pvz_promotion.py`，`GATE_VERSION = 6`，`policy` / `freeze_panel` / `decide_scored` / `decide`。
- v7：`python/pvz_promotion_v7.py`，`policy` / `initial_panel` / `freeze_panel` / `next_panel` / `summary` / `decide` / `ab_summary`，内部 `import pvz_promotion as legacy` 复用统计。
- 调度：`scripts/curriculum_queue.py` 的 `--gate-version` 默认 **7**；`scripts/promotion_gate.py` 的 `--gate-version` 默认 `GATE_VERSION`（6），`evaluate_dual` 在 v7 时同时跑两套。
- 预筛：`scripts/select_student.py` 固定 `from pvz_promotion import decide_scored`，`score['prediction'] = decide_scored(pairs)`（第 96 行），**即使主线用的是 v7**。

**谁在用**
- 生产 qc：v7。
- 双判/回归：`promotion_gate --gate-version 6`。
- 选学生：v6 预测。

**行为差异**
- v7 是顺序检验 + A/B 汇总；v6 是单步统计。用 v6 预测去筛 v7 的候选，预筛与终判口径不同。
- 两套 `freeze_panel` 都来自 legacy，v7 又删除了监控任务，边界容易混。

**建议保留**
- 只保留 v7，v6 显式改名 `legacy` 并只供复现；`select_student` 的预测改为按当前 `gate_version` 选择，或明确标注“仅用于排序、不构成晋升”。

**迁移代价**
- 中低。代码量不大，风险在历史结果的可复现性；需要保留 v6 入口但不再作为默认。

---

## P2：维护

### 7. 部署脚本 `qc_deploy17..25` 是九份整份复制

**事实**
- `scripts/qc_deploy17.sh` … `qc_deploy25.sh` 共约 1619 行；相邻版本 diff 60–160 行，绝大部分是同一份 `start_qc()`、备份/回滚、文件清单。
- `scripts/qc_watchdog.sh:watchdog_latest_deploy` 按 `LOG_DIR/qc_deploy*.sh` 的 mtime 取最新，`watchdog_extract_start` 抽取 `start_qc` 依赖闭包来重启。
- `scripts/deploy17_at_boundary.sh` … `deploy22_at_boundary.sh` 是 4 行 wrapper；23–25 没有对应 wrapper。
- `qc_deploy25.sh` 定义了 `IMPORT_FILES` 却未使用；实际导入器来自 `L`（llmimport 工作树），与 `FILES` 从 deploy25 取源码是两套来源。

**谁在用**
- 生产部署与 watchdog 重启；`python/test_deploy17/21/22.py` 引用历史脚本。

**建议保留**
- 一个参数化启动器（版本号、recipe、panel、额外 manifest、引擎路径作为变量），每个 `qc_deployNN.sh` 只保留差异与 `source`；`start_qc` 只留一份，watchdog 的抽取契约不变。旧脚本在队列排空、wrapper 不再引用后归档。

**迁移代价**
- 中。必须确认 `~/pvz-logs/qc_deployNN.sh` 与 boundary wrapper 的指向，以及 watchdog 的 mtime 选择规则；不能在 qc 运行时删除旧文件。

**§6.6 复核**
- 这不仅是维护问题，直接违反第 4 条“部署脚本不复制粘贴”。迁移目标是一个 `qc_launch.sh` 参数化命令，每版 `qc_deployNN.sh` 只保留版本常量与 `source`；`start_qc` 的文本仍由 `watchdog_extract_start` 从同一处抽取，契约不变。

### 8. `recipes/` 14 份，只有 3 份在用

- 在用：`dw4i2_v2timer.json`（A）、`dw4i2_v2timer_seed2cold.json`（B）、`dw4i2_v2timer_warm.json`（回退）。
- 其余多被 `qc_deployNN.sh` 的拷贝清单与测试引用；`dw4i2_v2timer_deckcap05` 只出现在 deploy18/19 与 `test_ab_queue`。
- 删除前必须查：队列里的 `job['recipe']` 存的是**绝对路径**；deploy 脚本的 `FILES` 清单；`python/test_ab_queue.py`。
- 建议：保留三份，其余移入归档目录或随旧 deploy 一起删除；先把 deploy 清单里的死引用清掉。

**迁移代价**
- 低，但顺序不能反：先清引用与队列，再删文件。

### 9. 观测编码 Python / C++ 两份（训练侧允许，生产逐步接口例外）

- Python：`pvz_value_net.featurize/encode_frame`、`pvz_rollout_policy.packet_features`、`pvz_student_observation.StudentHistory`、`pvz_token_observation`。
- C++：`src/EnvironmentValueNet.{h,cpp}`、`src/EnvironmentRollout.{h,cpp}`、`src/EnvironmentTokens.cpp`。
- 用途：C++ 整局 rollout 与原生根搜索必须与 Python 训练侧逐位一致；等价测试（`test_cpp_rollout`、`test_identity_cpp`、`test_value_encoder`、`test_sleep/timer/bungee_inputs`）就是契约。
- 原结论：这是性能所需的双实现，不是“双入口”。

**§6.6 复核**
- 这个结论只在 Python 侧**仅用于训练/调试**时成立。C++ 是特征、推理、搜索的权威；当前生产 `--native-games` 已走 C++ inference，Python encoder 用于训练缓存，判合规。
- 例外是 `StudentHistory` 与太阳收入历史：它们在生产的逐步 `ROLLOUT_PREDICT`/`ROLLOUT` 里被 Python 计算并传给 C++，属于 S1-2，必须迁 C++。
- 训练侧 encoder 保留的前提是永不成为生产后备；长期按第 3 条只做一次性等价对比，不维护完整 Python oracle（S1-4）。不要为了“单一实现”删掉 C++ 路径。

**迁移代价**
- 训练侧不迁移；生产逐步 history 按 S1-2 迁移，中高成本。

### 10. 场景文本渲染三份

- `scripts/llmplay/scene.py`（`header` / `grid` / `rows` / `cards` / `events` / `legend` / `render` / `timeline`）。
- `scripts/episode_query.py:_fmt_scene`（生产查询）。
- `scripts/render_episode.py:render_text`（训练文本报告）。
- 建议：GPT 包直接调用共享渲染器；先把 `scene.py` 的独有段落（卡组/开局面板）合并进共享模块，再删副本。

**迁移代价**
- 低到中，纯展示，不影响存档。

### 11. 任务生成器重复与 llmplay 分支家族

- 生成器：`make_day_panel` / `zone_panel` / `make_fog_user_panel` / `make_fog_user_v2_panel` / `make_roof_panel` / `make_multiplier_tasks` / `unified_panel` / `day_orig_tasks`，各自拼任务 dict；`unified_panel._task` 与 `zone_panel.task_document` 是同一个 schema 的两份写法。
- 分支：`llmplay`、`llmplay2`、`llmplay3`、`llmplay4`、`llmplay5`、`llmplay-decks`；旧 `make_llmplay_tasks.py` 拒绝运行却仍与 `make_llmplay3_tasks.py` 并存。未采纳的 `v2fogstack4` 原型见 S1-5。
- 建议：任务 dict 由 `unified_panel._task` 一处产出；llmplay 只保留 `llmplay-decks` 一条分支，其余在确认无未合并提交后归档。

**迁移代价**
- 中。分支清点必须先比对未合并提交；生成器合并要看各 panel 的族定义是否真的不同。

---

## 附：读者与写者对照

| 文件 | 写 | 读 | 备注 |
|---|---|---|---|
| `search_play.py` | npz | — | 生产学生局 |
| `human_demo.py` | jsonl（录制）、npz（export） | 两者 | 导入复验依赖它 |
| `episode_query.py` | jsonl（capture/反事实） | 两者 | 与 frame_store 注释矛盾 |
| `llmplay/game.py` | jsonl（deploy25）/ npz（task203） | 两者（task203） | GPT 包 |
| `import_llm_run.py` | 复制原样 | jsonl 行数 | legacy |
| `llm_import_sha.py` | 复制原样 | 整档哈希 | 生产导入 |
| `pvz_live_curriculum.py` | — | 两个索引文件名 | 消费导入结果 |
| `pvz_frame_store.py` | npz | 两者 | 唯一的容器层 |

## 建议执行顺序

1. **合并 task203 两处旁支修复**（`3c43bc13`、`a06b9edf`），让 GPT 写端与导入收据进入同一基线。这一步是体积问题的主要止血。
2. **收窄 `FILES`** 到 `meta.json/ticks.jsonl/frames.npz`，读端保留 jsonl 回退；`episode_query` 改用 `write_frames`。
3. **导入器入库即转换**，并统一索引文件；删除安装器的源码改写。
4. **§6.6 低成本清理**（不阻塞上面的体积工作）：删 `pvz_env.profile_for_deck`/`training_task`（S1-6）与 `v2fogstack4`（S1-5）；`Game.validate_deck` 改调 `env.deck_warnings`；`_python_rollout_reference`/`pvz_wait_events` 参考做一次 A/B 后删除（S1-4）。
5. **统一 schema** 与必备字段；search_play 补 schema。
6. **门禁**：v6 标记 legacy，`select_student` 预测对齐当前 gate 版本。
7. **§6.6 高成本迁移**（独立轨道，不阻塞 1–6）：`StudentHistory`/太阳历史 → C++（S1-2）；搜索树 → C++（S1-1）；`Game` 事件叙述改消费 C++ 输出（S1-3）。每项只做一次 A/B 对比，不保留常驻双跑。
8. **部署与配方**：一个参数化启动器（§6.6 第 4 条）；清引用后再归档旧 recipe / 旧 deploy。
9. 最后才考虑 **`llmplay.Game` 录制语义** 与 **场景渲染** 的合并；这两项改动面大、收益是维护性。

## task205 / deploy26：第一批落地（2026-10-31，待主控部署）

基于 `origin/deploy25`（`97994745`）建立 `deploy26`，采用 §6.6，完成第 1–6 步和第 8 步。task203 的写端与 SHA 收据已合流，`llmplay-decks` 取到最新 `bbfd1124`，`llmimport` 最新仍为 `a06b9edf`。GPT 包以后从主线构建，`scripts/llmplay/` 的修复直接进主线。

所有帧写入共用 `pvz_frame_store`，新录制及调试捕获写 NPZ，交付的存档文件限定为 `meta.json / ticks.jsonl / frames.npz`。旧 JSONL 继续由 `frame_path/read_frames` 读取，导入时转换一次并删除明文。`pvz_episode_archive` 统一元信息，写入 `schema=episode_archive_v2`、`schema_version=2`，生产 `search_play` 补上任务环境和终局字段。读旧存档直接使用录下的任务与卡组，不依赖现行清单仍保留同名任务。

导入仅留 `scripts/import_llm_run.py`：保留 SHA 收据、一次复验和原子入库，支持一次传多个包，发布 `merged_progress.json`。第一次新导入会合并旧 `import_progress.json` 后删除它，课程只读统一索引。不再改写源码或登记生成任务清单。删除 `llm_import_sha.py`、`pvz_live_import.py` 和 `deploy22_import.py`，旧离线扫描与注册路径一起退役。

删除环境层重复的 `profile_for_deck / training_task`，调用统一走 `pvz_tasks` 和 `training_task_kwargs`。删除 `v2fogstack4` 及测试，移除 Python rollout 与 wait 参考实现，native wait 的协议常量和薄接口检查留在 `pvz_event_env`。GPT 建局校验调用 `env.deck_warnings`。门禁统一到 `pvz_promotion` 的 v7，实现和执行入口各一份，删除 v5/v6 判定、独立 selector 及其面板刷新代码，旧版本历史作业不续跑也不改写。

部署命令只在 `qc_launch.py`，看门狗直接调用它。删除仓库九份 `qc_deploy17..25.sh` 及六份边界包装脚本，原边界助手更名为 `qc_boundary.py`。`deploy26.py` 共用这些入口，在组结束且状态落盘后切换，备份实际安装文件及软链，可恢复删除项和原有软链。GPT 地形路由沿用 deploy25，v5 / v5roof 和 branch1 不变，当前 A/B 槽位按下文第 11 步执行。

**行数**：合入 task203 后的代码及配置清理删除 **40,342 行**、新增 **1,388 行**，其中代码与测试删除 **6,482 行**、新增 **1,059 行**。种子表把连续数组缩为等价区间，删除 33,859 行、增加 210 行，任务与种子值不变。相对 deploy25，代码与测试合计增加 8,820 行、删除 5,165 行，增加部分主要来自本次要求合回主线的 GPT 包。

保留原面板及留出卡组数据供 v7 和任务设计使用，保留历史配方供旧记录读取，老师搜索树及 StudentHistory 迁入 C++ 按要求留给独立工程。

Mac 串行执行相关 `pytest -k`，每次带 `timeout 300` 和 `ulimit -f 2000000`，存档转换、压缩交付、v7 三方判定以及队列重试等小测试通过，`curriculum_queue.py --help` 可运行。未跑引擎对局、训练或三份重型测试整套，未操作台式机生产。

### 主控切换与回滚

在已审的 deploy26 源工作树执行下列命令。脚本自行通过 `systemd-run --user --wait --collect --property=KillMode=process` 进入临时单元，等组边界后安装。生产任务配置、课程状态和种子账本原样保留，GPT 包构建使用 deploy26 源工作树中的配置。

```bash
~/.venvs/ml/bin/python3 scripts/deploy26.py --revision origin/deploy26
```

切换脚本把 `llmimport/scripts/import_llm_run.py` 链到 cq 的同一实现，旧批量入口随之退役。task105 后续导入统一改用下面的命令，`<输入包>` 可以给多个路径，不再调用 `task105_import_batch.py`，也不执行旧安装器：

```bash
~/.venvs/ml/bin/python3 ~/PvZ-search/.claude/worktrees/llmimport/scripts/import_llm_run.py <输入包> \
  --dest ~/PvZ-search/.claude/worktrees/llmimport/artifacts/task105/import \
  --executable ~/PvZ-search/artifacts/production/deploy20/pvz-portable
```

成功后脚本打印备份目录 `~/pvz-logs/deploy26_<时间>_<进程号>`，回滚仍从同一源工作树执行：

```bash
~/.venvs/ml/bin/python3 scripts/deploy26.py --rollback ~/pvz-logs/deploy26_<时间>_<进程号>
```

台式机 `~/pvz-logs` 中的旧版 qc 部署副本保留，仓库打 `pre-deploy26` 标签指向 deploy25。以上脚本只交付给主控，本任务未部署。

### task205 追加第 9 步：1 倍复习份额

依据台式机 task206 的停滞诊断，在原课程配额之外追加 1 倍复习。`curriculum_queue.py --review-share` 默认 `0.25`，范围 `0..1`，设为 `0` 关闭。`qc_launch.py` 可透传同名参数，未指定时使用课程入口的默认值。当前实际倍率不为 1 的任务按原配额乘份额、向上取整追加，例如原来 48 局 3 倍继续采满 48 局，另外采 12 局 1 倍，0.5 倍任务也适用，已是 1 倍的不追加。

复习沿用同一任务定义及卡组，从原任务种子表的后续位置取种子，使用原 offset 键和既有续种账本。种子不足时先按原机制续领，不能续领则只使用剩余种子并记入现有课程日志。复习成功后进入普通训练标签，不更新课程窗口和探针计数，已完成的复习局不扣原轮次预算，中断重启会补排尚未完成的复习组。

示范起点帧数与实际倍率分别处理，原本 1 倍的示范不会因为起点档位是 90 等数值而追加。非 1 倍示范的复习按同关、同卡组从头运行 1 倍，接用原示范格的未来种子表，原示范起点与轮换游标保持不变。相关 `-k` 小测试已通过，未运行真实采集或训练，未新增配置文件或 JSON 清单，未部署。

### task205 追加第 10 步：B 槽重新配比对照

根据 task206 的“训练数据构成”一节，新增 `recipes/dw4i2_v2timer_rebal_seed2cold.json`，以 seed2cold 为底，仅添加 `source_weight` 的关卡平衡参数（`level_balance=0.25`、`level_balance_power=0.5`、`level_balance_max_multiplier=20`），并把现有 33 个 z* 历史标签权重从 1.5 改为 1.0。统一启动器的 `--alternate-recipe` 默认选新配方，A 文件不改，B 仍每轮冷启动 12 个 epoch、使用种子 2，L7 历史降采样选项不改。

在临时目录调用现有排程函数生成并打印 `recipe_b.json`，实际输出确认三项关卡平衡参数和 33 个 1.0 标签权重均保留，`init=null`、`restart_every=1`、`epochs=12`，A/B 种子为 `[1,2]`。除允许分别应用的来源权重和标签权重外，两臂打包选项相同，排程未报错，临时目录已删除。

两臂仍一同参加 v7 三方配对门禁，沿用逐轮 `ab_result.json` 的 A−B 方向和分地形结果，跨轮累计从新 B 配方启用后的轮次开始，旧 B 配方结果不混入。回滚与切换失败恢复会显式选回 deploy25 的原 seed2cold 配方，避免删除新配方文件后仍按新默认值启动。相关 `-k` 小测试 4 项通过，未启动真实训练或对局，未部署。

### task207：部署前审查与演练（2026-10-31，台式机）

对照 `origin/deploy26`（adf0bcc6）与生产 `origin/deploy25`（BASE 97994745）做只读审查，并在临时目录完成一次最小端到端演练。未改任何代码，未碰生产 qc / 看门狗 / stop_curr，`~/pvz-logs/curriculum_state.json` 本体只被复制读取。

#### 1. 启动命令逐项对照

从正在运行的 `curriculum_queue.py` 进程取出完整 argv（`ps -eo args`），与 `scripts/qc_launch.py:command()` 用生产同一组参数生成的命令按 token 排序比对。差异只有四处，均不改变语义：

| 项 | 生产现行 | qc_launch | 说明 |
|---|---|---|---|
| `--candidates-per-recipe` | 显式 `1` | 不传 | 新旧 argparse 默认都是 1（deploy25 `curriculum_queue.py:2487`） |
| `--gate-version` | 显式 `7` | 不传 | 新入口已移除该参数，每轮强制 `GATE_VERSION`（`curriculum_queue.py:2486`）；deploy25 默认也是 7 |
| `--state/--log/--seed-ledger/--start-seed-ledger` | 不传 | 显式写默认路径 | 与 argparse 默认值逐字相同 |
| `--review-share` | 不传 | 显式 `0.25` | 生产靠默认值，显式值与默认值一致 |

其余 `--rounds 100 --workers 16 --min-eval-units 4 --round-games 768 --games-per-group 48 --training-parallelism 1`、各面板 / 模型 / 引擎路径、`--obs-v3 --native-games`、`2>&1 | tee -a ~/pvz-logs/qc.out` 完全一致。用记录 argv 的假 `tmux` 执行 `qc_launch.main()`，退出码 0，生成单条 `tmux new-session -d -s qc -x 200 -y 50 ...`，载荷与现行生产逐项相同。

#### 2. 看门狗拉起路径

- 生产看门狗 argv 是 `/bin/bash /home/newbiexvwu/pvz-logs/qc_watchdog.sh`（systemd-run 单元 `qc-watchdog.service`）；`deploy26.py:50-58` 按 `command[1]` 匹配包含该路径的集合，能停下。
- 重启分支（`qc_watchdog.sh:104`）用 `$PYTHON`（venv）调 `$WORKTREE/scripts/qc_launch.py --repo ... --worktree ... --logs ...`，与部署启动参数一致；部署后该脚本在 cq 工作树内。因此 qc 崩溃（tmux 会话消失）时拉起的是本次核对过的同一条命令。
- 两点提醒（非缺陷）：新看门狗由 `deploy26.py:61-64` 用 Popen 直接拉起，不再挂在 systemd 单元下，原 `qc-watchdog.service` 会变成 inactive（transient 单元本就不跨重启保留）；部署时启动解释器见问题 7。

#### 3. 旧状态与旧导入进度

- 生产课程状态共 67 个作业：`gate_version=None` 57 个（31 done + 26 superseded，全部终态），v7 10 个（5 done / 3 queued / 2 superseded）。恢复逻辑（`curriculum_queue.py:1126`）只重新排队 `gate_version == GATE_VERSION` 的作业，旧作业不参与、不报错；重启时还会强制 `state['gate_version']=7`（`:2486`）。若将来出现非终态 v5/v6 作业，会被静默搁置而非续跑。
- 旧 `import_progress.json`：演练目录预置一份旧格式进度后导入，`import_llm_run.py:316` 与 `merged_progress.json` 合并，写入新台账后删除旧文件（`:403-404`），核查通过。

#### 4. 复习份额的种子间隔

对生产状态副本跑 `--dry-run`：本轮 20 组、264 局 = 8 个原配额组 ×24 + 12 个复习组 ×6。复习起点为 `max(group.offset + group.games, state.offsets[offset_key])`（`:1887`），标签 `..._review1_o<offset>`（`:1905`），1 倍组在 `:1879` 直接跳过。按 offset_key 机械检查全部 20 个区间：**零重叠**。示例：`day_l3_train_d7` 原段 504–528、复习 528–534；6 个 rd193 组 24–48 → 复习 48–54。
理论风险：若同一 offset_key 上两个非 1 倍组在同一轮排入且 `state.offsets` 未推进，会算出同一起点；本轮无此情况（见问题 5）。代价是每轮采集从 192 局增至 264 局（+37.5%），起始种子按轮加速消耗；种子不足时沿用 `renew_start_seeds`，续领失败降级为只采剩余并记日志，不会崩。

#### 5. 最小端到端演练

环境：新建 deploy26 工作树；课程状态、门禁 / 起始种子账本、`teacher_rates.json` 复制到 `/tmp/t207`。`free -g` 确认 16 GB 可用后，全程 `ulimit -v 8000000`、`nice -n 19`。工作树缺 gitignore 的运行时 artifacts，演练时临时软链 14 个目录到 cq，结束后已删除；`/tmp/t207` 已整目录删除。

1. **排程预演** `--dry-run`：20 组 / 264 局，倍率变化仅 `day_l3 2→2.5`。
2. **采集**：deploy20 引擎，`day_l3_train_d7`，`--seed-offset 0 --seeds 2`，2 局、1 胜 1 负；帧写 `frames.npz`，`ticks.jsonl` 正常。
3. **打包**：`curriculum_queue.pack_command` 按演练标签打包，得到 503 行样本（116 个植物动作），包 `775fed7c...`。演练配方只保留演练标签的 `label_weight` 以通过配方校验。
4. **训练**：1 epoch、CPU、batch 128、`max_steps=2`；503 行、2 个种子、验证 240 行，写出 `epoch_01.pt` 与 `model.pt`，退出码 0（CUDA 初始化 OOM 警告无害，实际 device=cpu）。另按小样本调了 `source_weight={}`、验证比例 0.5，属演练配置调整，非代码缺陷。
5. **门禁最小单元**：单任务 `task101_l1_day_d7`、2 个种子，A=刚训练的模型、incumbent=`dw4i_s1`；`promotion_gate._unit_command` 跑完后 `pvz_promotion.decide` 返回 `continue`、`delta_rate=-0.5`，`next_panel` 正常写出。`decide` 要求 monitor 行非空，最小演练用 `monitor={}`。
6. **导入**：临时 dest 导入两个小包——(a) npz 存档重新打包的 zip，(b) 原始 `frames.jsonl` 包的 zip。两者均通过 `human_demo.verify`；(b) 入库时转 `frames.npz` 并把 v1 meta 升到 v2；旧 `import_progress.json` 合并后删除。回执、`sha256/<sha>`、报告均正常。

演练覆盖“采集 → 打包 → 训练 → 门禁 → 导入”整条链，未发现会崩溃或写坏状态的缺陷。

#### 6. 发现的问题（交主控处理，均未改代码）

1. **cq 工作树有一处未提交修补会被覆盖（最需先处理）。** cq 的 `scripts/episode_query.py` 比 deploy25 多两行：写入 `meta["checkpoint_sha256"]`（cq:91）与展示指纹（cq:1275），是生产本地热修、从未提交。deploy26 恰好重写了同一函数（改用 `write_meta` / `write_frames`），但不含这两行；部署会静默覆盖，此后新采集存档不再带检查点指纹（溯源元数据丢失，非崩溃）。建议部署前把该修补并入 deploy26。
2. **`configs/` 不随 deploy26 部署。** `deploy26.py:23-27` 只分发 `python/ scripts/ recipes/`，本次改动的 4 个 configs 不进 cq；其中 `human_demo_tasks.json` 新增 49 个 `llmplay_decks_*` 注册。后果：(a) 导入器默认清单是 `ROOT/configs/human_demo_tasks.json`（`import_llm_run.py:22`，经 llmimport 软链解析到 cq），这些族在 cq 未登记，`resolve_family` 会在不比对卡组 / 倍率的情况下当新族接受（`:168-170`）；(b) 若将来门禁校准引用只存在于 deploy26 配置的 `demo_task`，`freeze_panel` 会 KeyError（`pvz_promotion.py:267`）。当前面板只引用 `demo_roof_46_doom_m200`、`demo_storm_40_orig_m100`，cq 配置都有，暂不触发。
3. **`archive_sha` 把帧文件名算进哈希。** `import_llm_run.py:242-256` 对 `meta.json`、`ticks.jsonl`、帧**文件名**做哈希；同一局以 `frames.jsonl` 与 `frames.npz` 两种包装到达会得到不同 SHA。deploy26 起新包统一写 npz，若把旧局以 npz 形式重新导入，会被当成新档重复计入。当前生产台账 3791 行、无重复 (task_id, seed)，属潜在问题；演练中同一存档重打包确实生成了第三个 SHA。
4. **行构造直接取原始 meta 的必需键。** `:389-390` 无保护地取 `meta['outcome']['result']`、`meta['outcome']['final_wave']`、`meta['operations']`、`meta['frame_count']`；`metadata()` 的补齐只在 `read_meta` / `write_meta` 路径生效，这里用的是原始 `json.loads`。缺任一键会 KeyError 并中止整批导入（无逐档隔离）。已知生产包都带这些键，外部手打包需注意。
5. **复习种子共享 offset_key 的理论重叠。** `:1887` 的 `max(...)` 在同一 offset_key 上两个非 1 倍组同轮排入、且 offsets 未推进时，会算出相同起点。本轮 20 个区间零重叠，专精组均为 1 倍被跳过，暂不触发；以后新增同 key 多倍率组需复查。
6. **每轮局数与种子消耗上升。** 本轮 192 → 264 局（+37.5%）；1 倍复习为 `review-share=0.25` 对非 1 倍组配额的追加，起始种子按轮加速消耗，续领失败会缩短该组数据量。
7. **部署启动解释器覆盖。** `deploy26.py:68-70` 用 `sys.executable` 作为 `--python` 与 tmux 载荷解释器，覆盖 qc_launch 的 venv 默认值；若不用 venv 执行部署脚本，首次启动会用缺 torch 的系统解释器而失败，而 `tmux new-session` 仍返回 0，部署显示成功；看门狗 120 秒后用 venv 重启但会消耗 2 次重启预算。按文档用 `~/.venvs/ml/bin/python3 scripts/deploy26.py` 执行即可。
8. **已核对、未发现问题的项。** deploy26 删除的旧模块（`select_student`、`pvz_promotion_v7`、`pvz_selection_refresh`、`pvz_wait_events`、`replay_roof_wave_gate`、`pvz_live_import`、`deploy22_import`）在现行 `python/scripts/recipes` 与活跃 `~/pvz-logs/*.sh` 中无引用，退役导入入口只出现在旧部署脚本里。回滚时 `deploy26.py:71-72` 用安装快照里的 qc_launch 启动，不传 `--candidates-per-recipe/--gate-version/--review-share`，与 deploy25 队列默认值等价，回滚命令有效。cq 工作树在这 42 个将被覆盖的路径中，除 `scripts/episode_query.py` 外均与 deploy25 逐字节相同；24 个删除路径在 cq 已不存在。

#### 7. 补演：第 11 步的干净 B 谱系（2026-10-31 追加，台式机）

对照新尖 `origin/deploy26`（e60d07e6 / 28f6774a）在 `/tmp/t207b` 重跑排程、打包、训练与重启四条，只读复用 task194 的转换池（取其中 3 份示范档）。全程 `ulimit -v 8000000` + `/usr/bin/nice -n 19`，`free -g` 先确认 16 GB 可用；未改代码，未碰生产 qc / 看门狗 / stop_curr，演练完 `/tmp/t207b` 整目录删除。第 1 节的启动命令对照在新尖只余一处实质变更：`qc_launch.py:21-25` 的 `--recipe` 默认改成 A 的重新配比配方、`--alternate-recipe` 改成 B 的干净配方，`--b-init/--b-initial-epochs/--b-epochs` 只在显式给出时传入，回滚分支 `deploy26.py:71-73` 显式把两个配方选回 deploy25 的 `dw4i2_v2timer.json` / `dw4i2_v2timer_seed2cold.json`。

1. **排程预演（真实 `enqueue` 代码路径）。** 新进程一初始化就把两条非终态旧 B 作业（round 69、70，原 status queued）标为 `superseded`，原因 `clean B experiment replaced old recipes`，同时把 `b_student / b_round / b_lineage` 写成 gptbc_s1 / 0 / `task205-clean-b`（`curriculum_queue.py:1096-1099`）。**部署当刻的副作用**：生产队列里任何尚未训完的旧 B 作业会被同样丢弃，A 侧不受影响。第 71 轮实际写出：
   - `round_000071/recipe_b.json`：`init=gptbc_s1/model.pt`、`epochs=8`、`seed=1`、`source_weight={default:0, llm_demo:1, teacher_r52:1, teacher_gptbc:1}`；
   - `round_000071/recipe.json`（A）：`init=null`、`epochs=12`、`source_weight.llm_demo=0.0`，保留 `level_balance=0.25 / power=0.5 / max_multiplier=20`；
   - A / B 标签集 2438 项逐字相同、`llm_demo_import` 相同 ⇒ 弹出 `source_weight / label_weight` 后的行选项一致，两臂共享同一个包；`train_rollout_policy.py:555-558` 在合并分段时才套用来源权重，共享包 + 各臂权重是构造性质，不是巧合。

2. **来源权重实测。** 演练包 = 2 份生产老师档（`z15dag18_fast_search` / `n19dag2_fast_search`，395 行）+ 3 份 GPT 示范转换档（1,924 行），打包命令由训练器自己打印（`--wait-keep 0.1 --obs v2timer --packet-head identity --aux-horizon 0 --cells --llm-demo-import ... --llm-demo-results 1`），2319 行、分段 3 个。`data/.llm_demo_rows` 由 `demo_paths` 自动软链到转换池（`pvz_llm_demo_rows.py:18-21`），示范行确实进了包。按训练器同一条 `load(..., source_weight)` 逐存档实测因子：
   - 3 份示范：`source_kind=llm_demo`（`label_source=llm_agent` + `llm_demo_sha256`）、因子 1.0、权重合计 1924.0；
   - 2 份旧老师档（meta `label_source=None`、`task.checkpoint=None`）：因子 0.0（落 `default`）、权重合计 0.0；
   - 整包 2319 行中 395 行权重为 0。训练日志 `3 seeds` 是 `seed_of[weight > 0]` 的去重计数（`train_rollout_policy.py:951`），只含 3 个示范种子，旧档连种子划分都不参与；`source_rows.json` 里示范 1248 训练 / 676 验证，两个老师标签 271 / 124 行但验证 0。

3. **B 从 gptbc 起训跑通。** 演练配方是 `dw4i2_v2timer_clean_b.json` 的原样克隆（obs / pool / packet_head / row_pool / cell_soft_weight / wait_keep / source_weight / init 全部不动），只把 data、out、cache、labels 缩到演练档，并设 1 epoch、CPU、batch 128、`max_steps=3`、验证比例 0.5。退出码 0，写出 `model.pt`（736 KB）、`metrics.json`、`source_rows.json`。warm-start 核对：`load_student(gptbc_s1)` 与 B 的网络在 pool / context / row_pool / packet_head / obs 及输入通道（117）上完全一致，21/21 张量被父网络权重覆盖，`trunk.0.weight`、`scalar.0.weight` 前缀逐位相等、无加宽列；训练后每个被覆盖的张量都相对初始化有位移 ⇒ 确实是从 gptbc 起步而不是随机初始化。CUDA 初始化 OOM 警告无害（实际 device=cpu）。

4. **模拟重启续谱系（两种恢复来源）。** 在同一份课程状态副本上，把第 71 轮的 B 模型路径写成 `round_000071/seeds/seed_000002/trained.json`：
   - 只有轮次标记（把 `b_student / b_round / b_lineage` 改坏，模拟状态写盘前崩溃）：`_b_parent(inf)` 扫标记恢复出 `b_student=<第 71 轮 B 模型>、b_round=71`（`curriculum_queue.py:1380-1392`）；
   - 只有课程状态（删标记）：直接读 `b_student / b_round`，结果相同。
   两种情况下第 72 轮的 `recipe_b.json` 都是 `init=<上一轮 B 模型>、epochs=4`（`b_initial_epochs=8` 仅在 `parent_round == 0` 时生效，`_b_recipe` 的判定见 `:1395-1401`），A 仍是 `init=null / epochs=12 / llm_demo=0.0`，断言全过。

5. **问题 7 复核：未修。** 新尖 `deploy26.py:68-70` 的 `launch()` 仍把 `sys.executable` 同时用作 `qc_launch.py` 的解释器和传给它的 `--python`，覆盖 `qc_launch.py:11` 的 venv 默认值；第 11 步把两个配方也从这条路径传入，所以这个覆盖的影响面没变、只是多了配方参数。按文档用 `~/.venvs/ml/bin/python3 scripts/deploy26.py` 执行即可回避；若用系统解释器跑部署，首次启动的 tmux 载荷会因缺 torch 失败而 `tmux new-session` 仍返回 0，部署显示成功、看门狗 120 秒后用 venv 拉起但吃掉 2 次重启预算。交主控决定是改成 `DEFAULT_PYTHON`（venv）还是在部署脚本里断言解释器。

### task205 追加第 11 步：B 干净谱系，A 重新配比

本步替代第 10 步的槽位安排。A 使用 `dw4i2_v2timer_rebal_seed2cold.json`，种子 1，每轮冷启动 12 epoch，保留关卡平衡和 z* 权重 1.0。B 使用 `dw4i2_v2timer_clean_b.json`，共享 A 的同一个包，来源权重为 `default=0、llm_demo=1、teacher_r52=1、teacher_gptbc=1`。生产行按存档 `task.checkpoint / task.policy` 判定先验，与 task206 的分类一致，未记录先验和旧老师行均为 0。GPT 示范对 A 的权重为 0。零权重行不进入训练或验证批次，也不参与种子划分、类别频率和关卡平衡计数，epoch 按有效行计算。

合入 `origin/llmrows` 的 task152 转行实现，复用 `llm_demo_import` 与 `training_rows`。两份配方都读取 `~/pvz-logs/gptbc/data/.llm_demo_rows`，已只读确认 task194 的 1190 份转换结果存在，直接复用已清理的 1,105,910 行，不重放或冷转换这批档案。统一导入器在验证通过后调用同一个转换函数，默认输出到 `--llm-demo-data ~/pvz-logs/gptbc/data`，新增示范继续排除未遮挡雾关及传送带空槽种植行。训练侧只读转换池，不再扫描原始导入回执。task152 的旧私有实验配方不恢复。

B 首次初始化为 `~/PvZ-search/artifacts/student/gptbc_s1/model.pt`，`--b-initial-epochs` 默认 8，后续从最近成功训练的 B 候选初始化，`--b-epochs` 默认 4，初始化模型可用 `--b-init` 指定。课程状态记录 `b_student / b_round / b_lineage`。初始化在开训前再按已有 B 完成标记确定，已排队的下一轮不会误用 r52 或旧的初始化预览，重启可从候选标记恢复。B 的门禁失败不打断谱系，B 未训成或被跳过的轮次不推进谱系。

门禁仍对 A、B 和现任三方同种子配对，当前现任为 r52。`ab_result.json` 使用实验标记 `task205-clean-b`，跨轮 A−B 从本标记重新累计，旧记录保留。切换时旧配方未完成作业标为 superseded，不再续跑旧 B，已完成历史不改。回滚显式选回 deploy25 原 A / seed2cold，沿用同一个启动器。

临时排程实际打印的 B 首轮参数为 `init=/home/newbiexvwu/PvZ-search/artifacts/student/gptbc_s1/model.pt、epochs=8`，有效来源权重为旧老师 0、未知 0、r52 1、GPT 先验 1、GPT 示范 1。后续排程打印了上一轮 B 路径及 4 epoch。共享打包选项一致，相关 `-k` 小测试通过，覆盖转换接入、过滤、来源权重、谱系排队和重启以及三方门禁。本步只改代码并跑小测试，台式机仅只读查看，未训练或部署。

### task205 紧急修复：r71 B 找不到共享训练包

只读生产记录确认，r71 已完成的 A 包为 `frames/packs/2fc4cbcdf4da5811.npz`，29,758,882 行，B 却查找 `b9dfef852bafe40d.npz`。代码中 `source_weight` 原已被最终 `pack_options` 忽略，但训练入口会重新扫描来源指纹并计算包路径，没有绑定本轮打包器公布的文件。两份配方的行内容选项相同，权重差异不应参与定位。此次同时去掉 `training_row_options` 中的两类权重，将 `label_weight` 与 `source_weight` 统一留到训练加载时施加，打包键只反映内容和抽样选项。

新控制器向 A、B 显式传同一 `--pack-file`，来自现有 `pack.log`。训练入口也兼容当前仍在运行的旧控制器：B 从已有 `--pack-recipe` 找到本轮日志，A 从已有 `round/seeds/seed_N/recipe.json` 找到本轮日志，直接读已发布包，不重新计算来源指纹。没有新增清单或中间文件。现有显式 1.0 标签权重的分片原地复用，不复制或重新编码。旧包若存有非中性标签权重，训练端按目标与已存权重的比例调整。

小测试用同一个实际包确认：GPT 示范确实在包内，A 的示范行权重为 0，B 的旧老师及未知来源行权重为 0，r52/GPT 先验生产行及示范保持有效。两臂修改各自私有权重，不修改共享缓存。相关 `pytest -k` 串行通过，包含旧启动参数、现有分片复用及旧包标签权重转换，均只用临时小样本，没有启动对局或真实训练。

主控部署时热更新本提交的五个运行时文件：`python/pvz_recipe.py`、`python/pvz_training_pack.py`、`scripts/train_rollout_policy.py`、`scripts/pack_training_rows.py`、`scripts/curriculum_queue.py`。无需重启生产 qc，也无需执行边界切换脚本。正在训练的 r72 A 已加载代码和数据，不受文件更新影响，之后新启动的训练子进程即采用修复。控制器下次正常重启后才采用显式 `--pack-file`，此前由训练入口兼容现有参数。r71 已失败 B 的状态未改，本次不自动重试。本提交未部署，未写生产状态或重新打大包。
