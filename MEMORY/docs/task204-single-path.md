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

部署命令只在 `qc_launch.py`，看门狗直接调用它。删除仓库九份 `qc_deploy17..25.sh` 及六份边界包装脚本，原边界助手更名为 `qc_boundary.py`。`deploy26.py` 共用这些入口，在组结束且状态落盘后切换，备份实际安装文件及软链，可恢复删除项和原有软链。启动参数沿用 deploy25 的 A / seed2cold、GPT 地形路由、v5 / v5roof 与 branch1。

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
