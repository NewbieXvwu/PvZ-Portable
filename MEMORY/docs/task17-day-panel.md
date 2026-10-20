# task17（主控 2026-10-20 下发给 Mac CodeBuddy / DeepSeek）：白天 L1–L10 原版难度评估面板

先读仓库根目录 `AGENTS.md`（优先级最高，尤其 §8 卡组与地形），再读本文。

## 背景
1-7 高难攻坚按停手规则结束：连续三组实验（EMA、按行汇总、辅助预测头），z20 两种子平均都没超过 v32 + 30。
按行汇总进了配方（已合进 pvz-env，d6c8e5eb），其余否决。下一阶段是“加宽”：**一个模型、原版难度、按地形配卡组，打通冒险模式**，
先从白天 L1–L10 开始。第一步要知道现在站在哪：现役学生和老师在每一关原版难度下各赢多少。

## 要做的
工作区：`git -C /Users/newbiexvwu/PvZAgent worktree add .claude/worktrees/daypanel -b daypanel origin/pvz-env`，只在这里改代码；
单独编译 `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`。
主仓库工作区和 `build/` 不许碰（Mac 正式采集在用）。同时最多 4 个游戏进程。不要 `cd` 进别的工作树。

1. **清单** `artifacts/stage4/day_panel.json`（不进 git；生成脚本进 git，如 `scripts/make_day_panel.py`）：
   每关一个评估任务 `day_lN_eval`，原版难度（倍率 1.0、不限波），卡组用 `scripts/pvz_tasks.py` 的 `deck_for_level`（不要另写），
   其余字段照 `artifacts/stage4/day_l7_hard.json` 里 `day_l7_*_eval` 的写法，先弄清 `playthrough`、`sun_start` 等字段的含义并在报告里写明。
   种子用一段与训练不重叠的评估段（与 `day_l7_*_eval` 同一段即可），每关 1024 个。
   特殊关：1-5（坚果保龄球）不收；1-10（传送带）先查环境能不能跑，能跑就收、不能跑写进报告。
   先核对每关的卡组是否只含该关已解锁的植物、地形是否都是白天草地；有不一致写进报告，不要自己改 `deck_for_level`。
2. **评估脚本**（可复用 `scripts/search_play.py` 的纯学生快速路径，或写薄包装 `scripts/eval_panel.py`）：
   给一个学生模型，对面板每关跑 N 局，输出每关胜局数、平均丢车数、平均到达波数，汇总成一个 json 和一张表。
3. **跑数**（本机最多 4 进程；用户 Mac 插上电之前先只做第 1、2 步和 2 局冒烟）：
   - 学生 `v32`、`d35rows2`（从 HF `transfer/` 取，没有就写进问题一节）：每关 256 局；
   - 老师（学生 v32 + 搜索，参数照 `artifacts/hard_mac/round15_flags`）：每关 32 局，只为知道老师在各关的上限。
4. 跑现有测试（`python3 -m pytest python -q`；本机若缺 pytest 就 `python3 -m pip install --user pytest`），全部通过。
   顺带确认 pvz-env 上刚合并的 rowpool（d6c8e5eb）测试全过。

## 交付
- 分支 `daypanel` 本地提交，可以 `git push origin daypanel`；不推 pvz-env、不合并。
- 结果写 `MEMORY/docs/day-panel.md`：每关一行（关卡、卡组、学生 v32 / d35rows2 胜局 / 256、老师胜局 / 32、丢车、到达波数），
  加一段“哪几关学生明显不如老师”（那就是下一步采集的重点）。
- 有疑问写在本文末尾“问题”一节，按最佳判断继续，不要停下来等。

## 不要做的
- 不训模型；不改 `deck_for_level`、不改默认值；不写手工规则。
- 不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；不 `git add -f`；`.pt/.npz/.log` 不进 git。
