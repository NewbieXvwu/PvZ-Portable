# task38（主控 2026-10-23 08:20 下发给台式机 Codex 第三会话 codex3）：训练配方文件化 + 统一运行指标日志

先读 `AGENTS.md`、`MEMORY/docs/hotpath-audit.md` §6（开关清单）、`MEMORY/docs/student-bottleneck-plan.md` 末尾 10-22/10-23 的几条、`scripts/train_rollout_policy.py`、`scripts/unified_panel.py`、`scripts/curriculum_queue.py`。

## 背景
1. **开关膨胀**：scripts 里 732 个命令行参数（`train_rollout_policy` 34 个）。每次训练靠手拼长命令行（`scripts/run_dw4_study.py`、`run_identity_validation.py` 等里各写一份），配方无法一眼对比，也留不下“这一版到底用了什么”。
2. **看不见运行状态**：10-22 晚 dw5i 训练只有 8% 时间在跑（磁盘狂读、GPU 0%），过了几小时才被代理事后发现；WSL 被内存挤死前也没有任何记录。

## 要做的（工作树 `git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/recipe -b recipe origin/pvz-env`）
1. **配方文件**：`recipes/<名>.json`（进 git，KB 级），字段 = 训练需要的全部设置（数据来源清单 / 行权重 / 模型结构 / 优化器 / epoch / 种子 / 验证集划分 / 输出位置）；
   `train_rollout_policy.py --recipe recipes/x.json` 读它，**命令行只保留 `--recipe` 与少数运维参数**（输出目录、设备、线程数）。其余训练参数从命令行删除（按 task33 §6 的批准清单与例外；配方里出现的字段才是唯一来源）。
   把 dw4i、dw5i、identity_dw3_s1 的实际配置反写成三个配方（从它们的 `config.json` / 运行脚本取，逐字段核对），验证用配方复训的前 200 步损失与原训练一致到 1e-6（同数据同种子）。
   `scripts/run_*study*.py` 等手拼命令的地方改为引用配方；面板脚本同理把评估设置写成 `recipes/panel_v2.json` 之类。配方文件的 schema 写在 `python/pvz_recipe.py`（校验 + 默认值 + 版本号），未知字段报错。
2. **统一运行指标日志**：`python/pvz_metrics.py`——一个后台线程每 30 s 往 `~/pvz-logs/metrics/<进程名>-<pid>.jsonl` 追加一行：时间、本进程与子进程 RSS、系统 MemAvailable、`/proc/pressure/memory|io`、本进程累计读写字节（`/proc/self/io`）、
   CPU 利用率、GPU 利用率与显存（有 `nvidia-smi` / NVML 才记）、以及调用方传入的进度计数（训练：step / 行每秒；采集：局数 / 决策每秒）。训练、采集 worker 汇总、课程控制器、面板都接上。
   再写 `scripts/metrics_report.py`：读这些 jsonl，给出“训练有效时间占比（GPU 利用率 > 50% 的时间比例）/ 磁盘读写 / 内存最低点”等摘要；**训练有效时间占比 < 50% 持续 10 分钟就在日志里写 WARN**（主控巡检时 grep）。
   指标日志按天轮转、保留 7 天。
3. 报告 `MEMORY/docs/recipes-metrics.md`：配方 schema、迁移前后命令对照、指标字段说明、示例摘要。

## 规则
- 不动 tmux 队列与别的代理会话；不改别的工作树与主工作区；最多 2 个你自己的游戏进程，训练对拍在 GPU 空闲时做（`pgrep -f "train_rollout_policy|train_value"` 为空）。
- 分支 `recipe` 提交，可以 push；不推 pvz-env、不合并。不加新开关（`--recipe` 是替换几十个参数的唯一入口）；不写字面密钥；不 `git add -f`；大文件不进 git。只用绝对路径。
- 每完成一小步就提交并在本文末尾“结果”里记一行；疑问写“问题”，按最佳判断继续。

## 结果
- 2026-10-23：去掉对拍工具人为写死的单进程预处理，恢复三份历史配方的8进程（池退出后才加载训练主体）；2进程预处理+两侧200步余弦调度回归通过。主动终止串行扫描半成品，dw4i已在parity_parallel按8进程重跑。
- 2026-10-23：查明合并带来的缓存全失效：obs模块/packet_features只新增注释却改变源码哈希；恢复观测历史原文并将packet标记移到函数外，v1/v2lite/v2/packet历史签名与缓存命中回归通过（18项，1环境跳过）。主动终止错误签名的dw4i任务，产物作废，修复版已在parity_frozen重跑。
- 2026-10-23：补充历史完整语料验收（dw4i 13643901、dw5i 15788614、identity 11051782行；前两者逐来源与训练/验证划分也须一致）；缺行/划分变化拒绝测试及带原长度余弦调度的200步回归通过。dw4i现有进程继续补建缓存，尚未进入训练；完成后独立补跑语料验收，不重复启动。
- 2026-10-23：合并后重新跑2-worker/3局真实采集（3/3，3452决策，指标末行逐项相同），v2配方单任务面板1/1且正确标记complete=false，证据task38-restart-smoke.json；全量dw4i对拍仍在装载，不计作通过。
- 2026-10-23：补迁移新合入的dw5训练/参考面板调用方，面板版本写入配方；修复对拍回调冲突并适配紧凑训练包（历史侧恢复原张量类型，原损失/优化器不变），带来源行报告的200步回归误差0，另31项迁移/课程/面板回归通过；dw4i完整语料已重启（尚未出200步结论）。
- 2026-10-23：重启恢复后已合并 origin/pvz-env（29fd6896），优先完成课程 --recipe 接入并删除全部 --pack-*；三份历史配方及辅助/上下文变体的实际 pack CLI 与 trainer.load 请求逐键一致。相关57项测试通过（含候选补齐回归修复）；旧dw4i对拍实际因回调同名失败、无200步结果，全部待重跑。
- 2026-10-23：按历史计划SHA找到三份原始训练脚本，装载函数、batch_loss与去观测后的epoch循环均与对拍基线AST完全相同，KB证据写入task38-source-audit.json；对拍工具增加复用前的完整装载参数一致性检查。
- 2026-10-23：为完整语料对拍补充历史缓存私有复制复用（原子不覆盖、独立inode、摘要照常校验），已有dw4i进程继续推进；真实装载满10分钟已自动产生WARN，确认告警并非仅测试模拟。
- 2026-10-23：采集进度改为每任务独占无锁共享槽，避免 worker 被 OOM 杀死时留下计数锁；2 worker 真实冒烟完成2局，2240决策与指标末行逐项一致（同时自有游戏进程未超过2）。
- 2026-10-23：新增 recipes-metrics.md（schema/命令迁移/指标字段/真实摘要）与完整语料200步对拍工具；三份历史命令各34字段逐项核对通过；指标扩展到6项测试，覆盖真实子进程RSS与WARN落盘。dw4i对拍进程仍在全量装载，未重复启动。
- 2026-10-23：study/identity/conveyor/dilution 等训练调用方改为引用冻结配方，面板改为 recipe 入口并新增 v1/v2/pressure 配方；schema 5 项、面板 7 项、数据行 6 项通过；v2 单任务真实冒烟 1/1，结果标为 incomplete。
- 2026-10-23：新增严格版本化配方 schema，反写 dw4i（170 标签）、dw5i（281）、identity_dw3_s1（82）；训练 CLI 收敛为 recipe/out/device/threads/precompute-workers，保存 resolved recipe；schema 4 项与 identity 训练 5 项测试通过。200 步历史对拍及调用方迁移待做。
- 2026-10-23：指标接入学生/价值训练、采集 worker 共享进度汇总、课程控制器和统一面板；单 worker 真实冒烟 1 局完成，1086 个决策与指标计数一致；metrics 4 项、课程控制器 11 项测试通过。
- 2026-10-23：完成独立运行指标模块与 metrics_report（30 秒采样、进程树 RSS/CPU、PSI、IO、可选 GPU、进度速率、10 分钟 WARN、按日轮转/7 天保留）；4 项测试通过，尚待各入口接入。

- 2026-10-23：接手 Codex 中断的 dw4i 完整语料对拍（原进程仍在装载，未重启）；预先按对拍工具的同一函数把 dw5i（215481 文件/7.5 GB）与 identity（114813 文件/0.86 GB）的历史缓存复制进本工作树私有缓存，两份配方起跑时不必重算；配方/指标/课程/面板/identity 训练 42 项回归全绿。

- 2026-10-23：dw4i 完整语料 200 步对拍**通过，误差 0**（13643901 行、48816 档案、两侧 source_rows 报告 SHA 相同）。同时查清验收口径缺陷：同一份代码在同一 GPU 上重跑，第 4 步就出现 9.5e-7 漂移、第 12 步到 7.5e-4，普通 CUDA 内核不可复现，1e-6 判据在非确定性后端上没有意义；对拍改为确定性内核（`CUBLAS_WORKSPACE_CONFIG=:4096:8` + `use_deterministic_algorithms`），同代码重跑误差 0。另加已发布训练包复用（meta 与本次 options 逐项相等才复用），重启不再重建 13.6M 行。

- 2026-10-23：dw5i 完整语料 200 步对拍**通过，误差 0**（15788614 行、56784 档案、两侧 source_rows 报告 SHA 相同）；identity 对拍已在上一步通过后自动接续启动。

- 2026-10-23：identity_dw3_s1 完整语料 200 步对拍**通过，误差 0**（11051782 行、38271 档案）；三份历史配方（dw4i/dw5i/identity）现已全部按完整语料、原余弦长度、同种子、确定性内核验证一致，本任务书的对拍验收项完成。

- 2026-10-23 **[close]**：合并 `origin/pvz-env`（`--no-ff`，合并提交 `ecf2dddb`，合并基 `29fd6896`）。
  冲突两处：`MEMORY/2026-10-23.md` 双侧保留；`scripts/unified_panel.py` 以本分支的 `panel_args` 配方入口为准（对方那段是同一 CLI 的旧 argparse 版本，已被配方入口取代）。
  自动合并处逐项核对两边功能都在：课程 `DEFAULT_VALUE` 已是 v4、`--pack-*` 仍不存在、包参数仍只来自配方；`train_rollout_policy` 同时保留配方 CLI 与 x86 `nan_to_num(0)` 修复；`search_play` 保留 `--leaf-probe` 与 `encode_frame(..., encoder_version)` 的编码 v2。
  构建引擎（`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -G Ninja`、`cmake --build build -j8`）后跑
  `PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources python -m pytest python -q`：**339 passed / 133 subtests passed / 0 failed / 0 skipped**。
  未建引擎时同一命令是 5 failed / 314 passed / 20 skipped，5 项全是 `test_cpp_rollout` 找不到 `build/pvz-portable`，属环境前置，建好后全过。
  课程控制器 dry run（`--manifest .../cq/artifacts/stage4/multiplier_panel.json --round-games 768 --games-per-group 48 --dry-run`，16 组 / 768 局）打印的 pack 命令与 `recipes/dw4i.json` 逐项相同，唯一额外参数是控制器自己的 `--prune`：
  `.../scripts/pack_training_rows.py --data /home/newbiexvwu/PvZ-search/artifacts/student/frames_v2 --labels <186 个标签> --wait-keep 0.1 --obs v2 --packet-head identity --aux-horizon 0 --cells --label-weight <33 个 z13–z16 标签=1.5> --prune`。
  dry run 现在会打印这条命令（此前只打印分配表），便于在一轮开跑前核对行参数。已 push `origin/recipe`。

- 2026-10-23 **[fix]**（按 17:50 审查）：
  1. **配方路径改仓库相对**：`recipes/*.json` 的 `data`/`out`/`cache_dir`/`read_cache_dir`/`manifest`/`value`/`executable` 等全部改成相对仓库根的写法，`pvz_recipe.load_recipe` 载入时解析成绝对路径（新增 `repository_root()` + `resolve_paths()`）。仓库根 = 当前 checkout；从 `<checkout>/.claude/worktrees/<名>` 里的工作树运行时解析到它所属的那个 checkout，所以本工作树删掉后配方仍有效，Mac 上同一条相对路径也成立。训练配方 `out` 改为正式产物位置（`artifacts/student/dw4i_s1`、`dw5i_s1`、`identity_dw3_s1`），对拍/试跑用 `--out` 覆盖；`cache_dir` 改为 `artifacts/student/recipe_cache/<名>`。唯一保留绝对值的是面板配方的 `resource_dir`（游戏资源在仓库外的本机缓存，跨机器本来就要改）；`provenance.original_*` 按原文保留。
  2. **测试平台假设**：`test_metrics.py` 的两项 `/proc` 测试加 `skipUnless(/proc/meminfo)`；`test_recipe.py::test_full_200_step_observer_with_source_row_report` 固定 `device='cpu'`（测的是对拍逻辑，确定性算子没有 MPS 实现）；`test_historical_recipes_resolve_all_training_settings` 改为断言配方文件里没有绝对路径、解析结果相对 `repository_root()`。原先 "相对路径必须报错" 的那条用例改成空路径报错。
  验证：建好引擎后 `PVZ_RESOURCE_DIR=... python -m pytest python -q` → **339 passed / 133 subtests passed / 0 failed / 0 skipped**（本机 Linux）。课程 dry run 的 pack 命令与 `recipes/dw4i.json` 仍逐项相同（`--data` 解析回同一个 frames 树，唯一额外参数仍是控制器自己的 `--prune`）。已 push `origin/recipe`（`fa59d68a`）。

## 问题
- 本分支缺 hotpath-audit.md；已 fetch 并完整读取 origin/audit 对应报告（含 §6），依据 task33 批准清单删除训练 EMA/全候选软目标入口，保留 cell soft temperature、width 与旧模型元数据。退役前标记 retire-recipe-training-flags-20261023。

## 主控追加（2026-10-23 10:10）：课程控制器的训练包参数改从配方取（优先做）
- pvz-env 合进了 task33 的打包装载（b95d2cf7）。`scripts/curriculum_queue.py` 每轮结束会调 `pack_training_rows.py` 打训练包，包参数来自 7 个新开关 `--pack-wait-keep/--pack-obs/--pack-packet-head/--pack-aux-horizon/--pack-cells/--pack-context/--pack-label-weight`，**默认是 v1 / 槽位头 / wait-keep 1.0**，与现行 dw4i 配方（v2 / 身份头 / 0.1 / 按行汇总…）不一致——照默认打出来的包，训练根本用不上（`train_rollout_policy.py` 要求包参数逐项相同）。
- 改法：`git merge origin/pvz-env` 后，课程控制器只收一个 `--recipe recipes/<名>.json`（默认指向现行采集学生对应的配方），包参数全部从配方解析（与训练用同一个解析函数，不复制字段表）；删掉这 7 个 `--pack-*` 开关。单测：配方 → `pack_command` 的参数与 `train_rollout_policy` 对同一配方要求的包键逐项相同。
- 这条做完再继续原来的“调用方迁移”。

## 主控收尾（2026-10-23 17:20）
三份配方对拍都过了（误差 0），task38 验收完成。主控已删除 `artifacts/task38/parity_parallel` 与 `artifacts/task38/cache`：33 GB，可再生，C 盘紧张。
最后一步：
1. `git -C /home/newbiexvwu/PvZ-search/.claude/worktrees/recipe merge --no-ff origin/pvz-env`，解冲突。pvz-env 新合了 v4（`DEFAULT_VALUE`→v4、`search_play.py` 编码 v2 / `--leaf-probe`）、task35、task42 和 `train_rollout_policy.py` 的 NaN 修复，两边功能都留。`curriculum_queue.py` 以本分支"打包参数从配方取"为准，同时保住主线的 v4 默认值。
2. 用 `PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources` 跑 `python -m pytest python -q`，要求全过。再跑一次课程控制器 dry run，打印出的 pack 命令要与 qc 现行配方一致。
3. `git push origin recipe`，在"结果"写 `[close]` 一行：合并提交号、测试数字、dry run 的 pack 命令。不碰 tmux `qc`，切换由主控做。

## 主控审查（2026-10-23 17:50）：合进主线前还差两处
主控在 Mac 上试合 recipe，pytest 结果是 4 failed / 310 passed，合并已撤回。要修的是两件事：

1. **配方里写死了本工作树的绝对路径**：`recipes/*.json` 的 `out`、`cache_dir`（以及 `data`、`init`、`row_weights` 等 `PATH_FIELDS`）都指向 `/home/newbiexvwu/PvZ-search/.claude/worktrees/recipe/artifacts/task38/...`。
   - 这个工作树之后会删；而且 `artifacts/task38/cache` 主控 17:20 已经删了。
   - 改法：配方里的路径一律写**仓库相对路径**（如 `artifacts/student/dw4i_s1`），由 `pvz_recipe` 在载入时按仓库根解析成绝对路径。只有一种写法，不加开关。
   - `out` 的默认值应是正式产物位置；对拍用的输出由 `--out` 覆盖。`original_*` 这类出处字段保留原文。
   - `test_historical_recipes_resolve_all_training_settings` 要在 Mac 上也过。
2. **测试的平台假设**（Mac 上失败；同类测试主线的做法是按能力跳过，见 `python/test_oom_stress_test.py`）：
   - `test_metrics.py` 的两项依赖 `/proc`：用 `skipUnless(Path('/proc/meminfo').exists())`，或者让 `pvz_metrics` 在没有 `/proc` 时给出明确的 None，测试按能力断言。
   - `test_recipe.py::test_full_200_step_observer_with_source_row_report` 在 Mac 上走了 MPS，确定性算子不支持。测试里固定 `device='cpu'`，因为它测的是对拍逻辑，不是设备。

修完后 `pytest python -q` 要求全过，再 `git push origin recipe`，在"结果"写 `[fix]` 一行。主控在 Mac 上复测后合并。
