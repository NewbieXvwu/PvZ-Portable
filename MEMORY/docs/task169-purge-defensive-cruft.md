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

## 结论（C++ 环境层，分支 `purgecpp`，2026-10-30）

用户 10-30 追加：src/ 环境 / 搜索 / 导出层同类清理，基于 origin/deploy17，不碰游戏本体逻辑。

### 先查格式：现存检查点用哪一种

扫描了本机全部 50 个 `model.pt`（Mac artifacts，与生产同源同步）按现行导出规则会写出的格式：

| 格式 | 检查点 | 状态 |
|---|---|---|
| PVZRN006 | dw4i_s1、dw5i_s1、identity_dw3_s1（identity 头） | 生产默认学生 |
| PVZRN005 | v33、v34 | 10-21 前的历史对照 |
| PVZRN004 | v20–v32、d17v2lw2s1 等 v2lite/v2 | 历史 |
| PVZRN002 | v1–v19、rollout_policy、value_net | 历史 |

deploy17 现行配方（dw4i2_v2timer、dw4i2_v2timer_history10 等）全部 `packet_head=identity`，
`export_student` 必然产出 PVZRN006；002–005 只被历史文件本身“在用”，没有任何活代码路径
产出或读取它们（旧 `.pvzrn` 仅存在于已废弃 worktree 的临时产物）。因此按“单格式”收敛。

### 批次 1（a333e1d6）：导出格式收敛为单一 PVZRN006

- `src/EnvironmentRollout.cpp::Load`：删 002–005 的条件字段读取与版本枚举，头字段无条件读满 7 项。
- `python/pvz_rollout_policy.py::export_student`：格式号选择表达式删除，恒定写 006；
  context/observation/row_pool/packet_head/embedding 全部无条件写（slot 头 embedding=width，
  与旧 005 二进制语义等价）。旧检查点重新导出后仍可在 C++ 里推演（obs 1–9 分支保留）。
- `python/pvz_env.py::load_rollout_net`：删五路版本判定与 `rollout_version>=2..6` 老二进制探测。
- `python/pvz_compat.py`：删 `pvzrn-versions` 登记（连带 `LEGACY` 标记）。
- 测试：`test_rowpool_aux.py`、`test_timer_inputs.py` 的头部字节断言改为 006 全字段。

### 批次 2（a32c467e）：调试 / 自校验 / 计时仪器 / 老二进制探针

删（C++ 约 240 行 + Python 约 60 行）：

- `STATE_BYTES` 命令 + `WriteEnvironmentStateBytes`（main.cpp 注释自述“调试工具”，
  唯一用户 `state_hash_determinism.py` 及测试 `test_state_hash_determinism.py` 属哈希复验，整文件删除）。
- `ENV_PROFILE` 命令 + 每命令 chrono 累计（actionSeconds/observationSeconds/controlSeconds）
  + `EnvironmentValueNet` 的 `featureSeconds/forwardSeconds/convolutionSeconds/calls` 仪器成员
  与三处 `std::chrono` 热点计时——唯一消费者是 purgedead 已删的 perf 诊断脚本。
- `BENCH_SNAPSHOT` 命令（分段计时 bench，用户 hotpath 诊断已删）。
- `SCREENSHOT` 命令 + `LawnApp::EnvironmentScreenshot`（task66 校准专用，脚本已删）。
- `pvz_env.py` 三处“老二进制能力探测”垫片：`_rollout_version`、`_sparse_tracks_version` +
  `require_sparse_tracks`、`_branch_observe_supported`（探测失败回退旧命令）。现二进制恒定
  广告这些能力，探测为纯死分支；`BRANCH_OBSERVE` 直接使用。ROLLOUT 响应里的
  `simulation_seconds/network_seconds/net_calls` 保留——search_play 的 rollout_metrics 与
  挑选统计真实消费。
- C++ ready 握手仍发 `rollout_version/sparse_tracks_version` 字段：llmimport、gpuwindow 分支的
  旧包装会读它们，删字段会让那两条活分支误判（跨分支兼容面，等 deploy18 三包合一后再删）。

### 留下的检查（一行理由）

- `STATE_HASH`/`StableEnvironmentHash`：search_play 与 llmplay 把它写进每份存档证据层并在重放时
  对比，删除破坏既有存档合同（task105 服务现行工作流）。
- `CRITIC_INPUTS`：活测试 `test_fog_observation.py` 用它查询引擎波次名单验证倍率缩放，是引擎状态
  查询接口而非自校验。
- `Load` 里的权重 `isfinite` 扫描 + 尾部 eof 检查：坏权重进推演会把整天采集变成废局（真实损失），
  成本每次加载约 1 ms、不在每 tick 热路径。
- obs 编码器 1–9 分支：LEGACY(obs-versions) 声明面板仍在对照 v34/v2lite 检查点，属 Python 观测层
  同一决定，不在“导出格式”范围内。

### 验收（Mac，AGENTS §6.5）

独立目录 `build169` Release + LTO 编译通过（两次）。定向测试（均 timeout 300、ulimit -f、单条跑）：
`test_env_protocol` 35 passed；`test_engine_value` 3 passed（新二进制）；`test_cpp_rollout`
logits_width(48 subtests)+v2timer_continuation+shovel；`test_sleep_inputs`/`test_bungee_inputs`
native parity 各 1 passed；`test_rowpool_aux`/`test_timer_inputs`/`test_identity_packets`/
`test_compat_registry` 全过；`curriculum_queue.py --help` 正常；活模块 import 正常。
`git grep` 核对 llmimport/gpuwindow 两分支：被删命令只出现在其非闭包诊断脚本与方法定义中，
llmplay 服务与 GPU 执行器不经过它们。

### 遗留

- 未部署；生产二进制仍为 prodint16/build16，本分支改动在 deploy18 合入后于组边界热更新。
- ready 握手字段（rollout_version/sparse_tracks_version）待三包（deploy18 = deploy17+purgelive+
  purgedead+purgecpp）合一、旧包装退役后删除。
