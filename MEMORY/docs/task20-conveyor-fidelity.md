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

## 问题
