# task22（主控 2026-10-22 下发给台式机 CodeBuddy 第二条线）：白天泳池 L21–L30 原版难度评估面板

先读仓库根目录 `AGENTS.md`（优先级最高，尤其 §8 卡组与地形：水路必须有睡莲 16 或香蒲 43，升级植物要声明所有权且只能种在底座上），
再读 `MEMORY/docs/task21-night-panel.md` 与 `MEMORY/docs/night-panel.md`（夜间面板，本任务照它的做法和教训）、DESIGN §14.9。

## 背景
加宽的第三个区。夜间的教训：(1) 老师的推演要用按植物身份出卡的 identity 学生，slot 头学生换了卡组会错位；
(2) 老师也打不过时先查卡组（夜间原卡组老师 3/80，加小喷菇后 58/80）。泳池有 6 行，中间两行是水，
`deck_for_level` 给的是 `[0,1,2,3,4,5,16,43]`（白天 6 卡 + 睡莲 + 香蒲）。

## 要做的
工作区：`~/PvZ-search/.claude/worktrees/pool`（分支 `pool`，主控已从 origin/pvz-env 建好，含 task21 的 `make_night_panel.py`），
只在这里改代码、单独编译（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`）。

1. **清单**：照 `scripts/make_night_panel.py`（能泛化就泛化成按区生成，别复制一份），生成 `artifacts/stage4/pool_panel.json`（不进 git）。
   先查清每一关（21–30）：地形（`pvz_constants.background_for_level`）、特殊玩法（3-5 / 3-10 是否小游戏或传送带）、波数、
   卡组解锁与所有权（`DECK_OPTIONS`、`profile_for_deck`）。特殊关照 task21 的方式处理并写清。
2. **卡组候选**（三套，同种子 30000–31023，只换卡组，任务族名互不覆盖）：
   - `pool_lN_eval`：`deck_for_level` 的 `[0,1,2,3,4,5,16,43]`（对照）
   - `pool_lN_eval_p9`：`[0,1,2,3,4,5,7,16,43]`（学生训练卡组 + 睡莲 + 香蒲）
   - `pool_lN_eval_p9k`：`[0,1,2,3,4,5,7,16,19]`（学生训练卡组 + 睡莲 + 缠绕水草 19）
   每套都按 AGENTS §8 校验地形覆盖与升级植物所有权；不合法就写进“问题”并跳过那套。
3. **跑数**（同时最多 2 个游戏进程）：普通关里挑 5 关（避开特殊关），三套卡组 × 老师 16 局 + identity 学生 64 局。
   老师 = identity 学生 + 搜索，参数照 `~/pvz-logs/queue101.sh` 里的 `run()`（但老师基线用 `--dagger-beta 1`：搜索实际执行；`--rollout-policy` 用
   `~/PvZ-search/artifacts/student/identity_dw3_s1/model.pt`，CPU，不开 CUDA）。
   Python `~/.venvs/ml/bin/python3`，资源目录 `~/.cache/pvz-research-resources`。
4. DESIGN 在 §14.9 之后登记新任务族（§14.10）。

## 交付
- `MEMORY/docs/pool-panel.md`：面板定义、特殊关、三套卡组 × 5 关的老师 / 学生表，写清哪套卡组老师赢得最多、学生离老师多远。
- 分支 `pool` 提交，可以 `git push origin pool`；不推 pvz-env、不合并。最后选哪套卡组由主控定。

## 规则
- **不要动任何 tmux 队列（q101 等）与别的代理的会话；不要改别的工作树与 `~/PvZ-search` 主工作区。**
- 不训模型；不改默认值；不改 `deck_for_level`；不写手工规则；不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。
- 只用绝对路径。每完成一小步就提交，并在本文末尾“结果”里记一行做到哪；疑问写“问题”一节，按最佳判断继续，不要停下来等。

## 结果

- 2026-10-22 第一步（清单）：`make_night_panel.py` 的通用逻辑抽到 `scripts/zone_panel.py`（按区生成：`Zone` + `Family`），`make_night_panel.py` 变薄包装且 `--combined` 输出与 task21 逐字节相同（cmp 通过）；新增 `scripts/make_pool_panel.py`，生成 `artifacts/stage4/pool_panel.json`（不进 git）：27 个任务（L21–24、26–30 × 3 套卡组，各 1024 种子 30000–31023）。L25（3-5）源码核对是 **Little Trouble**（`IsLittleTroubleLevel`，传送带 + 僵尸点数×4 + 半尺寸 1/4 血），默认排除；L30（3-10）是 mini-boss 传送带，收进来并标注。三套卡组均过 AGENTS §8 核对（水路有睡莲 16、香蒲 43 由 `profile_for_deck` 声明所有权、长度 9、无重复）。

- 2026-10-22 第二步（学生）：identity 学生 64 局/关 × 5 关（L21–24、26）× 3 套卡组 = 960 局。对照 0/320、p9 18/320（5.6%）、p9k 19/320（5.9%）；平均到达波数 11.0–11.5（对照）对 13.1–16.6（p9/p9k）。报告 `MEMORY/docs/pool-panel.md` 已写面板定义、特殊关、卡组核对与学生表（老师表当时未跑完，第四步已补）。

- 2026-10-22 第三步（登记）：DESIGN §14.11 登记三个任务族（`pool_l*_eval` / `_p9` / `_p9k`），写清地形、种子、特殊关（3-5 Little Trouble 不收、3-10 mini-boss 传送带收）、卡组核对与“老师用 identity 推演”。

- 2026-10-22 第四步（老师）：identity 推演 + 搜索（`--dagger-beta 1`，`--workers 2`）16 局/关 × 5 关 × 3 套 = 240 局。对照 13/80（16.3%）、p9 23/80（28.8%）、p9k 25/80（31.3%）。**p9k 最多但 p9/p9k 在噪声内**（差 2 局，区间大幅重叠），两套 p9 系都明显好于对照；学生 5.6–5.9% 对老师 29–31%，差约 23–25pp。40 波的 L24 是师生共同天花板（老师 0/1/4，学生 0/0/2）。报告 `MEMORY/docs/pool-panel.md` 已完成（§4.2 老师表、§4.3 结论）。

- 2026-10-22 第二步（主控追加，定 p9k）：`zone_panel.Family` 加 `exclude_levels`（训练族可按族排关）；`make_pool_panel.py --combined` 现在写 **35 个任务** = eval / eval_p9 / eval_p9k 各 9 关 + `pool_lN_train_p9k` 8 关（L21–24、26–29；L30 mini-boss 与 L25 不收），训练种子 `2000000+(N−21)·10000` 起每关 8192（2000000–2088191）。自查：训练段与评估段交集 0、任务内无重复、与 1500000/1600000/1800000–1878191/1900000–1998191 各段不重叠；夜间 `--combined` 回归仍逐字节相同。清单已拷到 `~/PvZ-search/artifacts/stage4/pool_panel.json`（cmp 相同）。DESIGN §14.11 已登记 p9k 决定与训练族，报告加 §7。

## 问题

- **解锁列会误报香蒲 43**：`LawnApp::GetAwardSeedForLevel` 不含升级植物，所以对照/p9 的 43 在 L21–L30 一律显示“未解锁”。
  这不是运行问题（所有权由 `pvz_tasks.profile_for_deck` 声明、`playthrough = 2` 时 `GetSeedsAvailable()` 返回 49），
  但报告里必须解释，否则会被读成非法卡组。p9k 的 19 在 L21–L22 也显示“未解锁”，同理只是首轮奖励列表的边界。
- **L25（3-5）没有卡组语义**：Little Trouble 是传送带 + 全僵尸点数 ×4 + 半尺寸 1/4 血，与 2-5 打地鼠同类，
  本面板不收；要评估它需要另开“传送带 / 小游戏”任务族（与 task20 的 conveyor 线同源）。
- **p9 与 p9k 的 2 局差不可判**：每关 16 局的老师样本，80 局总量下 23/80 与 25/80 的 95% 区间大幅重叠。
  如果主控要在两者之间选，建议要么按“哪套更适合采集”定性选，要么把两套都在更多关卡/种子上再跑一轮；
  本任务按任务书停在 16 局，不自行加量。

## 第二步（主控 2026-10-22 09:40 追加）：定泳池卡组 p9k，准备采集
老师 control 13/80、p9 23/80、p9k 25/80。泳池对老师也难，主因多半是老师的推演学生没见过泳池；夜间的经验是一轮采集 + 训练后学生就超过了
当时量到的老师水平（老师随学生一起变强）。所以不再比卡组，**泳池卡组定为 p9k `[0,1,2,3,4,5,7,16,19]`**，直接开始采集。
1. 生成器加训练任务族 `pool_lN_train_p9k`（普通泳池关，特殊关照你第一步的结论处理），字段同 `pool_lN_eval_p9k`，
   训练种子 `2000000 + (N−21)·10000` 起每关 8192 个（生成时自查与所有已有段不重叠），DESIGN §14.11 登记。
2. 重新生成泳池清单（含 eval / eval_p9 / eval_p9k / train_p9k），**另拷一份到 `~/PvZ-search/artifacts/stage4/pool_panel.json`**（主控的采集队列读这份）。
3. 提交、`git push origin pool`，在“结果”记一行，然后停下。
