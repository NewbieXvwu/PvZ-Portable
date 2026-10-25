# task86：静态候选网漏掉对空植物——用“强加全部合法格”的老师标签重训候选网

> 主控 2026-10-25 写。执行者：Mac Codex（plus2 号，tmux `codex-plus2`）。工作树沿用 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/conveyor2`（先 `git merge origin/pvz-env`）。每个任务最多 2 个游戏进程，其他任务进程不计入本任务限制。

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
- 预登记提交 `e18f8f14` 并已推送 `origin/conveyor2`。
- L39 仙人掌族现行候选与全合法候选均完成 16/16，胜率均为 0/16；气球终局数为 16 对 15。
- L39 三叶草族现行候选与全合法候选均完成 16/16，胜率为 0/16 对 8/16；气球终局数为 14 对 6，配对差 +50 个百分点。
- L39 逐决策原候选召回率为仙人掌 2431/2483、三叶草 3916/4014；全合法候选追加和最终动作已完成机械核对。
- L40 的全合法数据采集已在后续复测中完成；候选网重训和后续 32 局评估仍未执行，详见下方“继续执行：全合法采集复测”。

## 问题
- 仓库没有静态候选网训练器或 branch1 原始数据路径，现有 `train_value_branches.py` 的目标是叶值预测，不能直接替代候选网重训。

## 预登记（训练前）
- 模型沿用 `ValueNet(width=64, context=True)` 与 `scripts/train_value_branches.py`，不改生产默认；新数据按 25% 抽样，旧 branch1 数据占 75%，以旧数据为基线保留多区覆盖。
- 新数据只纳入强加全部合法格运行中未拒绝的老师最终种植动作，并保留 `seed`、关卡区、原候选集、全合法集和 `teacher_action_in_original`；种子按任务族独立分块，训练/留出按种子划分，留出种子不参与训练。
- 召回率按 day、night、pool、fog、roof 分区比较 branch1 与新网；配对胜率使用同种子 32 局，L40 0.5 倍与两族 L39 为主检验，L31 雾夜和 L41 屋顶为旧区回归检验，非劣界为 -3 个百分点。
- 只有在决策级数据完成机械审计、旧数据路径可读且不含检查点或日志后才训练；缺少上述条件时记录为未完成，不修改生产路径。

## 2026-10-25 路径复核

- 复核 `/Users/newbiexvwu/PvZAgent` 及全部工作树：指定的 `artifacts/hard/value_branch1/model.pt` 只有模型权重，没有原始训练样本；HF `realnewbiexvwu/pvz-agent-artifacts` 只有 `transfer/` 下的示范归档和 `value_v5.pt`，没有 branch1 语料。
- 现有 `scripts/train_value_branches.py` 接受 `value_ranking_check.py --dump` 的叶值分叉数据，训练目标是 `won`，不能消费 `teacher_action_in_original` 或候选集合标签。
- 历史提交 `2f63f10e` 的候选覆盖实验是“给现有候选补少量动作”的覆盖提议器，模型目标和 task86 的静态候选网重训契约不同；该实现已在 `e6a9ba4d` 删除，不能直接当作 branch1 重训器。
- 因此 task86 的 L40/L39 数据混训、候选召回对比、新网训练和 32 局评估保持未执行，生产默认与现有 branch1 权重不变。

## 继续执行：全合法采集复测

- 按“每个任务最多 2 个游戏进程”重新采集决策档案，L39 仙人掌、L39 三叶草、L40 0.5 倍各 16 局，三批均使用 `--workers 2`，没有修改生产代码或默认候选路径。
- L39 仙人掌全合法候选为 0/16，平均末波 20.38，15/16 局末帧仍有气球；L39 三叶草为 8/16，平均末波 26.88，6/16 局末帧仍有气球。
- L40 0.5 倍全合法候选为 12/16，平均末波 28.81，4/16 局末帧仍有气球；现行候选基线为 6/16，配对区间外的方向性增益为 +37.5 个百分点。
- 新档案包含 L39 仙人掌 2484、L39 三叶草 4015、L40 2478 个带候选目标的决策帧，保存在本机 `/tmp`，不进入 git。临时采集器已保存全合法目标，但没有在同一轮序列化原始候选边界，因此 `teacher_action_in_original` 的全量机械标签仍未形成。
- 训练器、branch1 原始语料和旧区混合数据依旧缺失，候选网重训、召回率对比及 32 局回归评估继续暂停；现有结果只作为候选遗漏复核证据。

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

## task86b 执行结果（2026-10-25）

- 用 `/tmp/task86b_manifest.json` 登记八个区域任务，学生 r11，`--cells 2 --random-cells 2`。白天、夜间、泳池、L40 使用稀疏补采（`position-rate 0.002`、2 samples），雾夜 L31、屋顶、L39 两族使用较密采集；全部分片和模型留在 `/tmp/task86b-corpus`、`/tmp/task86b-trained-full`，没有进入 git。
- 全量合并训练得到 13,581 行，其中 3,973 行为 kind 1 格点分叉，覆盖 179 个种子。留出集叶值成对一致率 `branch1 0.482 -> 新网 0.530`，格点成对一致率 `0.698 -> 0.508`。新网未满足静态格点评估目标。
- 核心 8 局方向性评估：L40 0.5 倍新网 `4/8`，旧 branch1 `7/8`；L39 三叶草新网 `3/8`。旧网 L39 对照因单局约 3 分钟仅完成 1 局（`0/1`），未形成完整配对。新网因此不进入生产，模型只保存在 `/tmp/task86b-trained-full/model.pt`。
- 结论：多区分叉训练链路已打通，叶值排序有所改善，但格点静态排序下降，且 L40 实战回退；当前数据配方和稀疏采样不足以支持替换 branch1。完整 32 局实战评估未继续，避免把不完整结果写成正式门禁。

## task86c 预登记（2026-10-26）

- 目标：只修复静态格点排序，不改变生产默认。task86b 新网不上生产，本轮从 branch1 权重重新开始。
- 语料：沿用八个区域任务（day、night、pool、fog L31、roof L41、L40 0.5 倍、L39 cactus、L39 blover），每区等量采样；稀疏区使用 `--position-rate 0.01 --samples 4`，全部使用 `--cells 2 --random-cells 2 --values none --dump-shards`。种子段另开并与既有语料机械核对，分片和审计结果写入 `artifacts/task86c/`，不进 git。
- 目标规模：合并后的 kind 1 行数至少 30,000，且八个区域均有数据；kind 0 行不进入本轮训练。
- 训练：扩展 `scripts/train_value_branches.py` 的 kind 过滤和学习率参数，使用 branch1 初始化，仅训练 kind 1，学习率为原值的十分之一。按种子划分训练集与留出集，每个 epoch 在留出种子上计算总体及分区格点成对一致率，保存留出总体一致率最高的模型，不使用叶值一致率作为早停指标。
- 晋级门禁（训练前固定）：留出格点一致率总体不低于 branch1，且每个区域相对 branch1 的下降不超过 0.02；否则停止，不做实战配对，不修改生产路径。通过后才跑 L40 0.5 倍、两族 L39 和 day/night/pool/fog L31/roof L41 各 32 局配对，沿用学生 r11、叶值 v5 及现行搜索参数，Mac 每个任务最多 6 个游戏进程。
