# task20（主控 2026-10-21 下发给台式机 Codex，额度用完由 CodeBuddy 接手）：原版传送带关（1-10）

先读仓库根目录 `AGENTS.md`（优先级最高，尤其 §8 卡组与地形），再读 `MEMORY/docs/student-bottleneck-plan.md` 末尾 10-21 的几条
（L10 与原版的区别、L10 变体不保留）、`MEMORY/docs/task19-identity-packets.md`（按身份出卡，已合进 pvz-env 81919bc0）。

## 背景
现在面板上的 L10 不是原版 1-10：`LawnApp::EnvironmentReset`（`src/LawnApp.cpp` 约 1411 行）不论关卡都把任务卡组写进卡槽并
`UpdateWidth(deck.size())`，于是开局就有我们的 7 张卡、传送带容量 7。之后引擎照原版继续发卡（`Challenge::UpdateConveyorBelt`，
权重表、按张数变化的发卡间隔都是引擎自己的）。原版是开局空、容量 10。这个变体没有独立价值，**原版开局做好后它停止新采集与评估**，
已有档案只当训练数据（档案里逐帧的真实手牌照样能学）。

task19 的 identity 头按植物身份出卡、逐槽读“种类 / 可用 / 冷却 / 买得起”，传送带上同一种卡有几张、供应池比 10 种多，都不受影响。
剩下要做的是让环境、观测与评估对准原版。都不是手写规则：只改环境保真度与观测，策略仍全部从老师数据学。

## 要做的
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/conveyor -b conveyor origin/pvz-env`，只在这里改代码、单独编译
（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`）。

1. **原版开局选项**：任务字段（名字自定，如 `"conveyor_start": "original"`）。传送带关（`HasConveyorBeltSeedBank()`）设了它时，reset 不写卡组、
   不 `UpdateWidth`，卡槽保持引擎原版初始化（容量 10、开局空，之后由引擎发卡）；不设时行为与现在**逐位一致**（旧任务、旧档案重放不变，
   用同一种子对比现有 `day_l10_eval_d7` 的若干局 `games.jsonl` 证明）。非传送带关设了它直接报错。
   `meta.json` 照常记录任务字段；`episode_query.py whatif` 能重放新任务的档案。
2. **观测核对**（只核对、补缺，不改 slot 模型的输入）：
   - 传送带卡的“买得起 / 可用”：传送带关种植免费。确认 `packet_features` 的 affordable 与 active 在传送带卡上和引擎
     `CanPickUp` / 合法动作一致；不一致就从引擎取（例如帧里给出传送带标志或实际花费），不要在 Python 里写“传送带 = 免费”的特判。
   - 传送带上的卡在滑动（位置 / 是否已到可点击处）：查引擎何时允许点它，若到位前不可点，观测里要能看出来（同样从引擎取）。
   - 当前张数已能由 `packet_features` 的非空槽数得出；不要另加手工特征。
3. **任务族**（DESIGN §14.7 登记，AGENTS §8 第 4 条）：`day_l10_orig_eval`（评估种子 30000–31023，与面板同段）与
   `day_l10_orig_train`（训练种子另起一段，如 1880000–1888191；1840000–1878191 已被 task19 的 `day_l*_train_d7p` 占用），写进 `scripts/make_day_panel.py` 生成的清单。
   卡组字段：原版开局不用卡组，清单里按引擎实际行为写清楚（如 `deck: []` 或省略），并让 `deck_for_level` / 校验接受这种任务。
4. **老师与学生能跑**：`search_play.py` 在 `day_l10_orig_eval` 上各跑 2 局冒烟（学生 v34：`~/PvZ-search/artifacts/student/v34/model.pt`；
   identity 学生：`~/PvZ-search/.claude/worktrees/identity/artifacts/task19/identity_dw3_s1/model.pt`；老师 = 学生 v34 + 搜索，
   参数照 `~/pvz-logs/queue100.sh` 里采集轮的写法）。开局空槽、等卡期间搜索 / 学生不能出错。
5. **基线数字**（≤4 个游戏进程，台式机 q100 队列在跑，别抢它）：`day_l10_orig_eval` 上 v34 与 identity 学生各 256 局，老师 64 局；
   写进 `MEMORY/docs/day-panel.md` 新的一节。
6. **小规模 DAgger 试验**（GPU 空闲时，先 `pgrep -f train_rollout_policy` 确认没人在训）：用老师（identity 学生 + 搜索）在
   `day_l10_orig_train` 采 2 轮各 128 局，加进 task19 的 identity 训练数据（`--labels` 照 `scripts/run_identity_validation.py`）训 1 个种子，
   评 `day_l10_orig_eval` 256 局与 7 卡面板 L7–L9（`day_l*_eval_d7`，各 256 局），看原版 1-10 涨多少、别的关掉多少。

## 规则
- **不要动任何 tmux 队列（q100 等）、不要改 `~/PvZ-search/.claude/worktrees/rowpool`、`identity` 与 `~/PvZ-search` 主工作区**；同时最多 4 个游戏进程。
- 分支 `conveyor` 提交，可以 `git push origin conveyor`；不推 pvz-env、不合并。
- 不改默认值；不写手工规则；不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；不 `git add -f`；`.pt/.npz/.log` 不进 git。
- 每完成一小步就提交，并在本文末尾“结果”里记一行做到哪；疑问写“问题”一节，按最佳判断继续，不要停下来等。
  你额度用完时台式机的 `agent_switch.sh` 会让 CodeBuddy（DeepSeek）接手、额度恢复后交回你，所以进度一定要落在提交和“结果”里。

## 结果

- 2026-10-21，1完成（原版开局选项）：`RESET_V2` 新增可选尾标记 `conveyor_original`（旧任务报文逐字节不变），传送带关设了它时 reset 不写卡组、不 `UpdateWidth`，保留 `Board::InitLevel` 给的 10 格空传送带、之后由 `Challenge::UpdateConveyorBelt` 发卡；非传送带关设了直接拒绝。空卡组在报文里写 `-`。观测只在原版开局任务里多一个 `conveyor_start` 键（旧观测不变）；Python `TaskSpec.conveyor_start`、空卡组 `profile_for_deck`（6 槽）、`packet_features` 按引擎标志把传送带卡视为买得起、`training_task_kwargs`/`collect`/清单校验全部接通。实测：L10 开局 10 个空槽、tick 240 出现首卡、可堆满 10 张；L7 设了被拒；dw3s1 在 `day_l10_eval_d7` 的 30000–30007 八局与既有 `dw3s1_l10_d7_256/games.jsonl` 除 `seconds` 外逐字段相同；27 项既有测试通过。
- 2026-10-21，2完成（观测核对）：传送带卡的 `active` 由 `SeedPacket::AddSeed` 直接置 true（SeedPacket.cpp:1010-1035），不经过 `SetPacketType` 的冷却分支；`CanPickUp`（SeedPacket.cpp:686-720）在传送带关跳过阳光检查，合法动作 = `CanPickUp` ∧ `CanPlantAt`（LawnApp.cpp:1726-1741）。所以观测里的标称 cost / affordable 与引擎合法性**不一致**：已在观测里由引擎回 `conveyor_start`，`packet_features` 只在原版开局任务上按它判买得起（旧任务逐字节不变，见 6 项新测试）。滑动：`CanPickUp` 不读 `mOffsetX`，只有 `MouseHitTest` 读（SeedPacket.cpp:898-917）；实测首卡出现那一帧 `legal_actions` 已非空，所以滑动不影响可点性，**不需要**新特征。张数仍由非空槽数得出，未加手工特征。

- 2026-10-21，3完成（任务族）：`scripts/make_day_panel.py` 生成 `day_l10_orig_eval`（deck `[]`、`conveyor_start: original`、种子 30000–31023）与 `day_l10_orig_train`（同字段、种子 1880000–1888191，避开 task19 的 184–187 万段）；旧 `day_l10_eval` / `day_l10_eval_d7` 保留不动。`deck_for_level(level, conveyor_start="original")` 返回空卡组，`conveyor_belt_level()` 从 `IsMiniBossLevel` 映射 10/20/30；清单校验接受空卡组与 `wave_cap=None`。

- 2026-10-21，4完成（冒烟）：[6 局证据](task20-smoke.json)。`day_l10_orig_eval` 种子 30000/30001：v34 学生 2/2 胜（855/966 决策）、identity 学生 2/2 胜（773/934）、老师（v34+搜索，queue100 参数）2/2 胜（863/779 决策，193/196 次搜索）。开局空槽、等卡期间没有报错；老师与学生都能在只有空槽的局面下等待。

- 2026-10-21，5部分完成（基线，`day_l10_orig_eval`）：v34 学生 **244/256**（95.3%，均 867.5 决策）、identity 学生 **247/256**（96.5%，均 853.8），各 4 worker、种子 30000–30255；老师 64 局交接时仍在跑（20/64，见“问题”）。
- 2026-10-21，交接状态：步骤 1–4 已全部提交（`f0132d8` 原版开局、`7c1d153` 观测核对+任务族+6 局冒烟、`836c55f` DESIGN §14.8），工作区干净；本文件的数值/冒烟证据在 `task20-smoke.json`。
- 2026-10-21，6未开始：DAgger 采集（2×128）、训练、评估都没有启动；训练数据镜像 `artifacts/task20/frames_v2`（对主工作区 `frames_v2` 的符号链接，未写主树）已备好。

- 2026-10-21，复核后预检：修正后的identity学生原版2局2胜，2份真实采集meta均保留空deck/原版模式/模型SHA，两个8动作前缀实际重放均保持10空槽开局。冻结[实验计划](task20-study-plan.json)：老师基线64局beta=1，两轮各128局beta=0；同一冻结identity老师，合并原82标签后从头训seed1原12epoch，再评原版256与七卡L7–L9各256。最多4个自有游戏，2/3GiB内存暂停/恢复，训练须无其他训练且14GiB稳定30秒。

## 问题

- 2026-10-21 接管复核更正：原 C++ identity 输入没有使用原版标志，affordable/global-ready仍按阳光计算；原search_play/collect meta及whatif未传conveyor_start，静态清单校验仍拒绝空deck/None wave_cap。已只在conveyor修复，31项目标检查通过（含40个真实原版局面Python/C++一致、回放传字段与清单契约），旧slot输入/旧任务episode字段不变。原identity 247/256基线须重评对齐后的输入；保留旧文件，不混作新基线。
- 2026-10-21 老师运行状态更正：833286及本树老师进程均已不在，原文件保留24个完整种子30000–30023，停止原因不确定；原作业dagger-beta=0，仅老师标注、学生执行。主控已明确老师64局必须beta=1（搜索执行），DAgger两轮beta=0（学生执行、老师标注），二者分目录重跑，不混比。
- 老师基线（`artifacts/task20/base_teacher`，PID 833286，4 worker）交接时仍在运行，只写了 20/64 局；接手方先确认它是否继续，`games.jsonl` 是增量写的、可续跑。
- 下一步建议：跑完老师 64 局并补进 `MEMORY/docs/day-panel.md` 新一节 → 用 identity+v34 老师（queue100 参数）在 `day_l10_orig_train` 采 2 轮各 128 局（offset 0/128，label `d10orig1/2`）→ 训 1 个 identity 种子 → 评 `day_l10_orig_eval` 256 局与 `day_l7/8/9_eval_d7` 各 256 局。
- 采集时 `--frames` 请写本工作树（`artifacts/task20/frames`），再硬链进 `artifacts/task20/frames_v2`，不要写主工作区。
