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

## 问题
