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

下表按“对体积 / 正确性 / 维护的危害”排序。P0 是体积，P1 是会产出错误数据或错误晋升，P2 是维护成本。

| 序 | 重复项 | 危害 | 现行实现数 | 建议保留 |
|---|---|---|---|---|
| 1 | 存档写入：npz vs jsonl | 体积（GB 级） | 3 | `pvz_frame_store.write_frames`（npz） |
| 2 | 导入链：legacy / SHA / batch / live / installer | 体积 + 维护 | 5 | 单一导入器，入库即转 npz |
| 3 | `meta.json` 同名 schema 四种形状 | 正确性 | 4 | 一种 schema，版本字段显式 |
| 4 | 对局驱动 / 录制 / 复演 | 正确性 + 维护 | 3 | `human_demo` + `pvz_env` 为唯一录制层 |
| 5 | 任务定义与卡组推导 | 正确性 | 3+ | `pvz_tasks` + `render_episode` 为权威 |
| 6 | 门禁 v6 / v7 与 `select_student` | 正确性（晋升判定） | 2 套统计 + 1 套预测 | v7；v6 显式标记 legacy |
| 7 | 部署脚本 `qc_deploy17..25` | 维护 | 9 | 一个参数化启动器 |
| 8 | `recipes/` 旧配方 | 维护 | 14 份，3 份在用 | A / B / warm 三份 |
| 9 | 观测编码 Python vs C++ | 维护（有意保留） | 2 | 两份都留，等价测试是契约 |
| 10 | 场景文本渲染 | 维护 | 3 | 共享渲染器 |
| 11 | 任务生成器 / panel 生成 / llmplay 分支家族 | 维护 | 多 | 一个生成器、一条 llmplay 分支 |

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
- 生产 qc：`$L`（llmimport 工作树）上用 `pvz_live_install` → `run_sha`，索引由 `pvz_live_curriculum` 读取。
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

### 8. `recipes/` 14 份，只有 3 份在用

- 在用：`dw4i2_v2timer.json`（A）、`dw4i2_v2timer_seed2cold.json`（B）、`dw4i2_v2timer_warm.json`（回退）。
- 其余多被 `qc_deployNN.sh` 的拷贝清单与测试引用；`dw4i2_v2timer_deckcap05` 只出现在 deploy18/19 与 `test_ab_queue`。
- 删除前必须查：队列里的 `job['recipe']` 存的是**绝对路径**；deploy 脚本的 `FILES` 清单；`python/test_ab_queue.py`。
- 建议：保留三份，其余移入归档目录或随旧 deploy 一起删除；先把 deploy 清单里的死引用清掉。

**迁移代价**
- 低，但顺序不能反：先清引用与队列，再删文件。

### 9. 观测编码 Python / C++ 两份（有意保留）

- Python：`pvz_value_net.featurize/encode_frame`、`pvz_rollout_policy.packet_features`、`pvz_student_observation.StudentHistory`、`pvz_token_observation`。
- C++：`src/EnvironmentValueNet.{h,cpp}`、`src/EnvironmentRollout.{h,cpp}`、`src/EnvironmentTokens.cpp`。
- 用途：C++ 整局 rollout 与原生根搜索必须与 Python 训练侧逐位一致；等价测试（`test_cpp_rollout`、`test_identity_cpp`、`test_value_encoder`、`test_sleep/timer/bungee_inputs`）就是契约。
- 结论：这是性能所需的双实现，不是“双入口”。保留两份，但任何一侧改动必须同步另一侧并跑对应等价测试；不要为了“单一实现”删掉 C++ 路径。

**迁移代价**
- 不迁移；维护成本即测试成本。

### 10. 场景文本渲染三份

- `scripts/llmplay/scene.py`（`header` / `grid` / `rows` / `cards` / `events` / `legend` / `render` / `timeline`）。
- `scripts/episode_query.py:_fmt_scene`（生产查询）。
- `scripts/render_episode.py:render_text`（训练文本报告）。
- 建议：GPT 包直接调用共享渲染器；先把 `scene.py` 的独有段落（卡组/开局面板）合并进共享模块，再删副本。

**迁移代价**
- 低到中，纯展示，不影响存档。

### 11. 任务生成器重复与 llmplay 分支家族

- 生成器：`make_day_panel` / `zone_panel` / `make_fog_user_panel` / `make_fog_user_v2_panel` / `make_roof_panel` / `make_multiplier_tasks` / `unified_panel` / `day_orig_tasks`，各自拼任务 dict；`unified_panel._task` 与 `zone_panel.task_document` 是同一个 schema 的两份写法。
- 分支：`llmplay`、`llmplay2`、`llmplay3`、`llmplay4`、`llmplay5`、`llmplay-decks`；旧 `make_llmplay_tasks.py` 拒绝运行却仍与 `make_llmplay3_tasks.py` 并存。
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
4. **统一 schema** 与必备字段；search_play 补 schema。
5. **门禁**：v6 标记 legacy，`select_student` 预测对齐当前 gate 版本。
6. **部署与配方**：参数化启动器；清引用后再归档旧 recipe / 旧 deploy。
7. 最后才考虑 **`llmplay.Game` 录制语义** 与 **场景渲染** 的合并；这两项改动面大、收益是维护性。