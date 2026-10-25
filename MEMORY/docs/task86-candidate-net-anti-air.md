# task86：静态候选网漏掉对空植物——用“强加全部合法格”的老师标签重训候选网

> 主控 2026-10-25 写。执行者：Mac Codex（plus2 号，tmux `codex-plus2`）。工作树沿用 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/conveyor2`（先 `git merge origin/pvz-env`）。Mac 上每个任务最多 6 个游戏进程。

## 证据
- **task75c**：L40 原生 0.5 倍，候选强加全部合法格后老师 **6/16 → 15/16**，多种了仙人掌 298 张、三叶草 97 张。现行静态候选网（branch1）在气球关几乎不提议对空植物，属于**候选遗漏**。
- 雾夜 L39 在多套卡组下也是气球破防（task74b / task79），很可能是同一个缺口。
- 屋顶（task76）强加候选只有很小的增益（200 对：gained 5 / lost 1），瓶颈不在候选。
- L30 强加候选后仍是 0/16，属于叶值或学习覆盖问题，不在本任务范围。

## 要做的
1. **复核 L39**：雾夜 L39 用 `fog79_l39_snow_blover_v2` 与 `fog79_l39_snow_cactus_v2` 两族，学生 r11，生产老师参数。照 task75c 的方法，同种子配对 16 局，跑“现行候选”与“强加全部合法格”两组。报胜率差与气球破防数。
2. **数据**：用“强加全部合法格”的老师在候选遗漏关（L40 0.5 倍、L39，以及第 1 步确认的其它关）上采集决策级数据。标签取老师最终选择的动作，记录它是否在原候选集里。种子段要独立并做机械核对。同时保留原有多区数据，防止旧区退化。
3. **重训候选网**：沿用 branch1 的结构与训练脚本，新数据与原数据混合，混合比例写在预登记里。评估分三项：
   - **候选召回**：老师最终动作落在候选集里的比例，按区报，对比 branch1；
   - **配对胜率**：L40 0.5 倍、L39 各 32 局，新网 vs branch1；
   - **旧区不退化**：day / night / pool / fog L31 / roof L41 各 32 局，配对比较，非劣界 −3pp。
4. 预登记写在动手训练之前。结论写 `MEMORY/docs/candidate-net-anti-air.md`，`git push origin conveyor2`。新网只交付路径与 SHA，是否上生产由主控定。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。不改生产默认，不加开关。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；**不手写打法规则或公式**，也不写“看到气球就加仙人掌”之类的候选规则。改进只能来自数据和训练。
- 模型与资源路径（只读）：
  - 资源：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`
  - branch1：`/Users/newbiexvwu/PvZAgent/artifacts/hard/value_branch1/model.pt`
  - 叶值 v5：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/demo/artifacts/hard/value_v5/model.pt`
  - 学生 r11：`/Users/newbiexvwu/PvZAgent/.claude/worktrees/fog36/artifacts/task79/models/r11/model.pt`
  - Python（带 numpy）：`/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3`
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

## 问题

## 主控追加（2026-10-26 00:30，task86b：训练入口就在仓库里）
“没有静态候选网训练器”这个判断不对。branch1 本身就是分叉价值网，训练链路见 DESIGN.md §14 表格 B 行“第 2 轮”：`value_ranking_check.py --dump` 生成分叉数据，再用 `train_value_branches.py` 训练。
- `--dump` 里的 **kind 1** 行，正是“某张牌种在某格之后那一刻的局面 + 该候选全部未来的平均结果”。这就是 search_play `--static-value` 在挑格子时打分的对象，也就是候选网的训练目标。
- **kind 0** 是 3000 tick 叶子，是 v2 系叶值的目标。
- branch1 的原始语料不在了。这不是阻断：branch1 当年只用白天 z13 / z15 的数据训练，没见过泳池、雾夜、屋顶。正好借这次用现行学生 r11 重新生成多区语料。

做法：
1. **语料**。用 `value_ranking_check.py --values none --dump-shards <目录> --cells 2 --random-cells 2`，学生 r11。
   - 关卡：
     - 新数据：L40 0.5 倍、两族 L39；
     - 旧区：day L7、night、pool L21–L25、fog L31、roof L41。各取一个现行训练族。
   - 每关 32 个训练种子。种子段与全部现有段不相交，要机械核对。
   - 留出 8 个种子，只用来算成对一致率。
   - `--random-cells` 是为了让对空植物种在非学生首选格的分叉也能进入数据。这就是候选遗漏的直接解法，不写任何规则。
2. **训练**。`train_value_branches.py --init <branch1> --data <全部分片> --compare branch1=<branch1>`，宽度和 context 与 branch1 一致。报每区成对一致率，新网对 branch1。原先 25% 新 / 75% 旧的混合比例作废，改为按关等量，并把这一改动写进预登记。
3. **评估**。不变：L40 0.5 倍与两族 L39 各 32 局配对；五个旧区各 32 局，非劣界 −3pp。老师只换 `--static-value`，叶值 v5、学生 r11 不变。
4. 正在跑的 `task86_collect.py` 跑完就停，不用丢弃，它的结果作为召回对照。Mac 上每个任务最多 6 个游戏进程。

## 主控追加（2026-10-26 04:00，task86c：数据放大 8 倍，以格点一致率为早停）
task86b 的结果是：格点成对一致率从 0.698 掉到 0.508，L40 也退步。我的判断是两个原因：
- kind 1 只有 3,973 行，太少；
- 从 branch1 起步后用全量学习率训了 40 个 epoch，把原有的格点排序冲掉了，叶值一致率反而上升，说明训练目标被 kind 0 主导。

做法：
1. **语料放大**。kind 1 至少 30,000 行，八个区按关等量。
   - 稀疏区（白天、夜间、泳池、L40）把 `position-rate` 提到 0.01，samples 至少 4；
   - `--cells 2 --random-cells 2` 不变；
   - 种子段新开，机械核对；
   - 分片落在 `artifacts/task86c/`（ignored），不要放在 `/tmp`。
2. **只训格点**。branch1 在生产里只用来挑格子，所以训练只用 kind 1 行；如果 `train_value_branches.py` 不支持按 kind 过滤，就加一个参数，并配测试。
   - 从 branch1 起步，学习率取原值的 1/10；
   - 每个 epoch 在留出种子上算格点成对一致率，取最好的 epoch；
   - 要同时报告新网与 branch1 在每个区上的格点一致率。
3. **晋级条件**（训练前写进预登记）：
   - 留出的格点一致率总体 ≥ branch1，且没有哪个区低于 branch1 超过 0.02；
   - 达到后才做 32 局配对：L40 0.5 倍、两族 L39，以及五个旧区非劣。
   - 不达标就停，写结论。
4. Mac 上每个任务最多 6 个游戏进程。结论写 `MEMORY/docs/candidate-net-anti-air.md`，`git push origin conveyor2`。
