# task169：彻查并删除防御性冗余（manifest、自校验、写死路径、死代码）

用户 10-30 指示：仓库里堆了大量莫名其妙的 manifest、依赖地狱、“防空气”的校验和写死的依赖，用 DeepSeek 彻查整个仓库，全部删掉。

直接起因：10-29 主控清理磁盘时删了旧 worktree `conveyor`。生产代码里有多处写死 `.claude/worktrees/conveyor/artifacts/stage4/day_panel.json`（一个没进 git 的生成文件）和 `conveyor/artifacts/task20/frames`，deploy17 和回滚都起不来，停采约半小时。

政策依据：AGENTS.md §6.5（禁止挡在关键路径上的形式主义验证）。

## 要删的四类东西

1. **manifest / 指纹 / 自校验。** 包括：
   - SHA256 / md5 记录与比对；
   - `ready.json`、`prepared.json`、`archive_manifest.json`、`*_manifest.json` 一类旁路文件；
   - 加载前的哈希核对；
   - 启动预检、重放复验、逐位对拍；
   - “以防万一”的断言和防御分支。
2. **写死路径。** 指向 `.claude/worktrees/<名字>/` 的路径，以及某台机器的绝对路径（`/home/newbiexvwu/...`、`/Users/...`）。
   - 改成仓库内相对路径或命令行 / 配方里的配置项。
   - 能在运行时现算的就现算（例如任务表由关卡号 + `deck_for_level` + 种子段生成），不要再落盘成 JSON。
3. **死代码。**
   - 一次性任务脚本、旧实验配置、旧配方；
   - 没有任何活代码引用的模块，以及只测这些东西的测试。
4. **依赖地狱。**
   - 兼容旧格式的多套分支（例如已没有检查点在用的旧导出格式）；
   - 版本兼容垫片；
   - 只为旧检查点 / 旧存档保留的路径；
   - 从来没打开过的可选开关。

**目标是简化，默认删除。** 只有能说出它防住过的具体损失（坏数据进训练、生产状态被破坏）的检查才留下，在交付里写一行理由即可。

不要新增任何校验、预检、manifest 或 JSON 中间文件，也不要为这次清理本身设计流程。只测被删检查的测试一起删。

## 什么是“活代码”

生产入口及其闭包（import 关系，加上 `subprocess` 调起的脚本）：

- `scripts/curriculum_queue.py`：采集、打包、训练、挑选、门禁全链路；
- `src/` C++ 引擎与构建（`CMakeLists.txt`）；
- 部署与运维：`scripts/qc_deploy17.sh`、`scripts/deploy17_at_boundary.sh`、`scripts/deploy17_boundary.py`，以及看门狗；
- GPU 空档执行器（task162 的脚本）；
- task105 扫描服务（`origin/llmimport` 上的脚本）；
- `scripts/hf_sync.py`；
- 以上各项的测试。

`MEMORY/` 不在删除范围（那是历史记录），但其中指向被删文件的链接要改掉或标注“已删除”。`README.md`、`DESIGN.md` 里的引用同样要改。

## 分工（两个会话并行，文件不交叉）

基线都是 `origin/deploy17`（生产现行代码）。每完成一批就 `git fetch && git rebase origin/deploy17`。台式机 codexb 正在 deploy17 上修 conveyor 那几处写死路径，那几处留给它。

- **会话 1：分支 `purgelive`，只改活代码文件的内部。**
  - 删检查、manifest 的读写、写死路径、兼容分支。
  - 每删一类先用 `git grep` 找全引用，改完同批提交。
- **会话 2：分支 `purgedead`，只整文件删除非活文件。**
  - 先自己算出活文件闭包（不用落文件），然后删闭包之外的 `scripts/`、`python/`、`configs/`、`recipes/`、`experiments/`、测试等。
  - 删前按 AGENTS §4 查引用链，删后扫断链。

## 验收（Mac，遵守 AGENTS §6.5 第 8 条）

- 每批改动后：
  - 活模块逐个 `python3 -c "import ..."`；
  - `python3 scripts/curriculum_queue.py --help`；
  - 跑与改动相关的 `pytest -k`，加 `timeout 300`、`ulimit -f 2000000`，一次一个。
- **禁止整套运行**：`test_curriculum_queue.py`、`test_curriculum_pipeline.py`、`test_qc_resilience.py`。
- 不部署，不连台式机生产。主控审过后并入下一次部署（deploy18），在组边界热更新。

## 交付

结论追加在本文件末尾：
- 删了什么（按四类，大致文件数和行数）；
- 改了哪些写死路径；
- 留下了哪些检查（一行理由）。

每批提交推送各自分支。

## 结论（会话 2，分支 `purgedead`，2026-10-29）

### 删了什么

净删除 **332 个文件、约 5.47 万行**（相对基线 `8763b5c8`；另恢复 5 个教师服务脚本，见下）：

| 目录 | 删除文件数 | 内容 |
|---|---:|---|
| `scripts/` | 261 | 一次性任务脚本、旧面板/回放/值网络实验、旧 `qc_deploy14*`/`15`/`16`、`scripts/llmplay/`、`scripts/prototypes/` |
| `python/` | 63 | 死模块（`pvz_compat.py`、`pvz_cpu_usage.py`）、dagger/critical-forks/roof/value-* 等一次性测试、旧测试夹具 |
| `recipes/` | 4 | `dw5i`、`identity_dw3_s1`、`panel_pressure`、`panel_v1`（生产配方与测试引用的 `dw4i*`/`panel_v2` 保留） |
| `configs/` | 2 | `llmplay_robustness`、`task118_balloon_decks` |
| `experiments/` | 2 | `t7` 旧 BC 实验 |

按题目四类划分：

1. **manifest / 指纹 / 自校验**：约 30 个 `check_*`、`verify_*`、`audit_*`、`state_hash_*`、`compare_*manifest*`、llmplay `selfcheck` 脚本及只测它们的测试。
2. **写死路径**：会话 2 未改活文件内部，故未改任何写死路径（由会话 1 / codexb 处理）。
3. **死代码**：其余 300 余个一次性脚本、旧实验配置/配方、无引用模块和测试。
4. **依赖地狱**：`python/pvz_compat.py`（旧格式兼容）、`scripts/llmplay/`（旧外部模型包）、旧 value 导出/强化实验分支、历史 deploy 脚本。

### 活文件闭包口径

从 `curriculum_queue.py`、deploy17 三件套、看门狗、`hf_sync.py`、GPU 空档执行器（`origin/gpuwindow`）、task105 扫描服务（`origin/llmimport`）出发，按 import、AST 中非文档字符串的路径字面量、shell 调用求闭包，再并入保留测试及其 helper。共保留 119 个文件。

**补充发现：** `dsh/pvz-teacher`（教师 MCP 服务）通过子进程调用 `scripts/episode_query.py`、`lint_skills.py`，后者又依赖 `probe.py`/`search_episode.py`/`fog_timeline.py`。这条链不在任务书列出的生产入口里，但属于仓库内真实引用链。按 AGENTS §4 恢复这 5 个脚本及直接测试（`test_episode_hand`、`test_human_demo`、`test_lint_skills`、`test_replay_task_spec`、`test_search_episode`），避免教师服务断链。

### 留下的检查（一行理由）

- `curriculum_queue.py` 的种子账本 / 门禁契约：防止同一评估种子被重复使用、坏候选进入生产。
- `qc_deploy17.sh` 的边界等待、备份与 rollback：防止切换时破坏生产状态（task165 已实际用到回滚）。
- `hf_sync.py` 强制并入 `resume.json` 指向的检查点：漏传会导致归档无法续训（已实测踩过）。
- task105 的文件锁与导入清单：扫描服务并发写状态，去掉会损坏续扫进度。
- `python/conftest.py` 的三个受限测试护栏：曾因整套测试写日志撑满 Mac 磁盘，护栏只在三个重型测试文件生效。

### 验证

- 活模块逐个 `import` 通过；`python3 scripts/curriculum_queue.py --help` 通过。
- `pytest --collect-only -q python`：631 项可收集，无导入错误。
- 相关小测试：`test_metrics`、`test_growable_pool`、`test_check_start_seed_state`、`test_env_protocol`、`test_episode_hand`、`test_lint_skills`、`test_search_episode` 均通过；未整套运行 `test_curriculum_queue`/`test_curriculum_pipeline`/`test_qc_resilience`。

### 遗留

- `scripts/search_play.py` 的两处 `--help` 文案仍提到已删的 `verify_reseed.py` / `sim_skip_agreement.py`；该文件属会话 1 / codexb 的活代码修改范围，未在本分支改动。
- 未部署、未连接台式机生产。

## 结论（写死路径专项，分支 `purgepaths`，基于 origin/deploy18，2026-10-30）

范围：scripts / python / configs / recipes / src / dsh 全仓搜 `/home/newbiexvwu`、`/Users/newbiexvwu`、
`.claude/worktrees/<名>`；生产部署脚本指向 cq 工作树与 ~/pvz-logs 的运行配置按用户指示保留。未新增任何校验。

### 批次 1（c1a22d88）scripts/

- `search_play.py`、`render_episode.py` 的 `DEFAULT_RESOURCE_DIR`、`human_demo.py` / `backfill_frame_context.py`
  的 `--resource-dir` 默认值、`import_llm_run.py` 的 `DEFAULT_RESOURCES`、`make_roof_panel.py` 的硬编码资源目录：
  统一为 `~/.cache/pvz-research-resources`（与 `curriculum_queue.py` 现行默认同构；生产与 Mac 均可用，
  手动跑其它资源用既有 `--resource-dir` 参数）。
- `make_multiplier_tasks.py::scan_manifests` 不再写死 `.claude/worktrees/cq|obsup` 扫描根，改 `--scan-root`
  参数（可重复，默认本仓库）；该函数只在手动种子审计 main 里调用，生产路径不经过。

### 批次 2（5fce1c78）recipes/

- 12 份配方的 `provenance.source_path / original_out / original_cache_dir / original_source_row_report`
  去掉 `/home/newbiexvwu/PvZ-search` 前缀，改仓库相对字符串（`pvz_recipe` 只把它们当字符串记录，运行不读）。
- `panel_v2.json` 的 `resource_dir` → `~/.cache/pvz-research-resources`（`PvZEnv` 已做 expanduser）。
- v2timer 系列的 `data / cache_dir / out` 指向 pvz-logs 状态目录与主仓库 `artifacts/student/`：这是生产课程
  队列逐字读取的运行配置，与 qc 脚本同属豁免类，保留；候选训练时 `out/cache_dir/data` 由队列覆盖。

### 批次 3（16502167）python/ 测试

- 13 个引擎测试的资源/二进制默认值统一为 `PVZ_RESOURCE_DIR`/`PVZ_EXECUTABLE` + `~/.cache` / `ROOT/build`
  回退（此前一半写死 Mac Downloads、一半写死台式机 home；两处的字面路径都不再出现）。
  `test_human_demo_conveyor.py` 的 `/tmp/pvz-boss50-build` 一并改 env。
- `test_incremental_pack.py` 的真实 pack 对拍路径、`test_qc_watchdog.py` 的实机部署脚本探测改 `~/pvz-logs` 相对
  （缺失仍按原逻辑 skip）。`test_deploy17/test_obs_recording` 中 `replace('/home/newbiexvwu', tmp)` 是匹配被豁免
  部署脚本的 fixture 机制，保留。

### 批次 4（c62dc966）部署脚本残留

- `qc_deploy17.sh` 的 `EXE15` 仍指 `.claude/worktrees/obsup/build7`（conveyor 事故的同类：可删工作树），
  改为主仓库 `artifacts/production/deploy15/pvz-portable` 固定命名，与 deploy16/18 一致。
- `test_deploy17.py` 的启动命令断言仍期待已退役的 `prodint16/build16`（deploy18 基线上本就 2 failed），
  改为 `artifacts/production/deploy16/pvz-portable`，该文件 11 项全过。

### 保留清单（豁免/合理）

- qc_deploy17/18、deploy*_at_boundary、qc_watchdog 的 R=/home/newbiexvwu/PvZ-search、W=cq 工作树、PY=venv、
  LOG_DIR=~/pvz-logs——用户明示豁免的生产运行配置。
- recipes 的 data/cache_dir/out（同上类的生产状态目录）。
- src 第三方库注释里的 `C:\WINDOWS\CANYON.MID` 示例文字；dsh/pvz-teacher 文档/帮助里的示例路径。
- Mac 本机补了 `~/.cache/pvz-research-resources -> Downloads 资源包` 软链，使统一默认在两台机器都直接可用
  （机器环境调整，不入库）。

### 验收（Mac，AGENTS §6.5）

活模块 28 个 import 全过；`curriculum_queue.py --help` 正常；全部 12 份配方经 `pvz_recipe.validate` 复核合法。
定向测试（单条 timeout 300 + ulimit -f）：test_import_llm_run/test_human_demo/test_replay_task_spec/
test_deck_warnings 40 passed；test_qc_watchdog+test_obs_recording 16 passed；test_deploy17 11 passed；
test_incremental_pack -k round2 skip（本机无该 pack，同基线）；test_fog_observation 非引擎用例同基线；
配方相关 recipe 用例 2 passed；测试收集 25 项无错误。deploy18 基线固有的 2 个 test_deploy17 失败已随批次 4 修复。
未部署、未连台式机。
