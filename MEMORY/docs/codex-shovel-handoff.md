# 交接书：铲子进动作空间，第一阶段（老师侧）——计划第 11 项

写给接手的 Codex（2026-10-18，主控 Claude 起草）。先读仓库根目录 `AGENTS.md`（体积与交付规则，**优先级最高**），再读本文。

## 0. 开工前

1. 确认你在哪台机器上，按下表取路径（`artifacts/` 下都是本地文件，不在 git 里）：

   | 用途 | Mac（`~/PvZAgent`） | 台式机 WSL（`~/PvZ-search`） |
   |---|---|---|
   | 任务清单 | `artifacts/stage4/day_l7_hard.json` | 同左 |
   | 叶子价值网络 v3 | `artifacts/hard_mac/value_v3/model.pt` | `artifacts/value_net/v3/model.pt` |
   | 选格子静态价值网络 | `artifacts/hard_mac/value_branch1/model.pt` | `artifacts/hard/value_branch1/model.pt` |
   | 游戏资源目录 `--resource-dir` | 默认值即可 | `~/.cache/pvz-research-resources` |

   学生 v23（当前最好）不一定在本地：用 `huggingface_hub.hf_hub_download(os.environ["PVZ_HF_REPO"], "transfer/v23_model.pt", token=os.environ["HF_TOKEN"])` 下载到 `artifacts/student/v23/model.pt`。
   **凭据只用环境变量名，不要打印、不要写进任何文件。**
2. 在仓库里新建 worktree 和分支：`git fetch origin && git worktree add .claude/worktrees/shovel -b shovel origin/pvz-env`，之后所有改动都在这个 worktree 里做。
   **只在 `shovel` 分支本地提交；不 push，不合并。** 工作区里的 `artifacts/` 是空的，上表的文件用主目录的绝对路径引用。
3. C++ 有改动时，在 worktree 里单独编译（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j3`）。
   **不要重编主目录的 `build/`**：两台机器上都有正式采集在用它。
4. **资源纪律**：两台机器都在跑正式实验。你同时最多用 4 个游戏进程（`--workers 4` 以内）；台式机上不要碰 GPU。

## 1. 背景

- 用户看 v20 的一局（`MEMORY/docs/teacher-diag-v20-z15-30000.md`），发现模型从不用铲子，后期把双发、寒冰种在最前排被吃。
- 核实结果：老师和学生的动作空间里都没有铲子。
  - 老师 `scripts/search_play.py` 的候选只有两类："每张卡最好的 `--per-card` 个格子"和"等待"，代码在 `by_card` / `candidates` 附近。
  - 学生 `python/pvz_rollout_policy.py` 的 `act()` 也只输出种植或等待。
- 环境本身支持铲子：动作写法是 `{"type": "shovel", "row": r, "col": c}`，`scripts/episode_query.py` 里写作 `shovel:row:col`。
- **项目原则：不写手工规则或公式。** 比如"双发在前排就铲"这种规则不能写。挑哪些候选、值不值得铲，都由价值网络打分或续走比较来决定。
- 当前对局压力是 1.7 倍：任务 `day_l7_z17_train` / `day_l7_z17_eval`，v23 在 1.7 倍下 1024 局赢 559 局。

## 2. 要做（第一阶段：只改老师，验证铲子对老师有没有用）

1. **`search_play.py` 加一个开关，把铲子动作加进候选。** 新参数例如 `--shovel-candidates k`，默认 0 表示关闭；关闭时旧行为必须逐位不变。
   - 候选来源要和种植一致：用静态价值网络（`--static-value`）给"铲掉之后"的局面打分，取前 k 个，再交给续走比较。
   - 续走里学生接手后不会用铲子，但可能在铲空的格子上重新种。"铲掉前排、往后挪"的价值正是从这里来，续走能看到。
   - C++ 续走路径（`--cpp-rollouts`、分支快照 `branch_snapshot`）要能处理根动作是铲子的情况。如果接口只接受种植动作，就扩展它，并补测试。
2. **在推翻统计里，单独记录"老师选了铲子"的次数。**
3. **验收**：
   - 开关关闭时，取同样的若干个种子，与旧代码逐局一致：胜负、决策数、推翻数都相同。
   - 老师用 v23 续走，在 1.7 倍评估种子 30000–30127 上打 128 局，铲子开和关配对比较。报告：
     - 两边胜局数、各自独胜局数、McNemar p 值；
     - 每局老师选铲子的次数；
     - 每局搜索耗时的变化。
   - 老师参数：`--verify-samples 9 --value <叶子 v3> --static-value <branch1>`，其余用默认值。
   - 抽 2–3 个老师选了铲子的局面，用 `scripts/episode_query.py`（frame / whatif）讲清楚：它铲了什么，之后发生了什么。用人话写。
4. **第二阶段先不做**（学生加铲子输出头、C++ 学生同步），只在最终报告里给设计建议：
   - 学生怎么表示铲子，例如多一个"铲子"卡槽，格子头复用；
   - 旧标签怎么兼容。
5. **文档**（中文、说人话）：
   - `MEMORY/2026-10-18.md` 追加记录；
   - `learn/06-expert-iteration.md` 末尾加一小节，面向 ML 初学者，讲为什么动作空间里没有的动作，模仿学习永远学不会。
   - 每完成一节就提交一次。

## 3. 交付

最终回复用中文，包括：
- 改了什么，提交列表；
- 一致性验收结果；
- 128 局配对结果；
- 铲子案例；
- 第二阶段建议。

同时把这份报告写进 `MEMORY/docs/codex-shovel-report.md` 并提交，主控会从分支上读。
