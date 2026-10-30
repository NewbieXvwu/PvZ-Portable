# task176：KL（策略惊讶）加权采样的离线臂

工作树 `klsample`（分支 `klsample`，基于 deploy19）。日期 2026-10-30。

## 0. 结论（TL;DR）

1. **可以用，但按现状不值得换。** 50% 均匀 + 50% 按 `KL(搜索目标‖r52 先验)` 加权的采样
   确实改变了学习结果，但离线动作准确率的提升很小（五地形宏平均 `type_acc`
   +0.5 / +0.75 个百分点，两个种子同号），且日间和 random 上是负的。
2. **变化的主要成分是“种得更凶”，不是“学得更好”。** KL 只在“搜索比较过 ≥2 个包络槽”
   的行上非零，而这些行几乎全是种植物行，所以加权一半把训练集里的种植物占比从
   49.4% 抬到 52.6%（+3.2 pp）。结果是 `plant_recall` +5 pp、`wait_false_plant`
   +3.4~3.8 pp，`cell_acc` 持平——这是操作点平移，不是新能力。
3. **真打小面板没有一致结论。** 五地形 × 4 种子 × 38 关 = 152 局/臂，同一批种子配对：
   种子 1 整体 +0.7 pp（p=1.0，无差异），种子 2 整体 +11.2 pp（p=0.0095）。
   分区看方向不稳定：日间 +19.4 / +11.1 pp，泳池 +16.7 / +22.2 pp，夜间 −33.3 / +5.6 pp
   （种子 1 夜间 p=0.004，种子 2 夜间无差异），雾天 0 / +8.3 pp，屋顶两臂都是 0/8。
   304 局合计 157 vs 175（51.6% → 57.6%），但 36 局/区/种子的噪声很大，夜间这项在两个
   种子间翻向，不能当成稳定收益。
4. **后续如果要继续这条线，先修混杂再谈收益**：把加权一半按植物/等待两类分别归一，
   让种植物占比与均匀臂一致，再比 2 个种子；真打面板至少 4 个种子。见 §7。

成本：KL 缓存一次前向 3.5 分钟（4.35M 行），每臂训练 8~9 分钟。便宜，结论可信。

## 1. 问题

训练从包里抽样时，每行对梯度的信息量并不相同：搜索目标和当前学生先验差得多的行，
正是“学生现在答错、且搜索有明确更好答案”的行。KataGo 的做法是把采样频率的一半
（`policySurpriseDataWeight=0.5`，g170 之后）按 `KL(搜索策略目标 ‖ 网络先验)` 加权，
另一半保持均匀。本任务验证这个思路在 PvZ 学生网络上是否带来可测量的收益：

- 指标：`day / night / pool / fog / roof` 五个地形的未见过对局（即未见卡组）上的留出准确率；
- 对照：同一数据、同一预算、同一份验证划分，只把采样换成均匀；
- 差异明显时再用 `bench_play` 做小面板真打。

## 2. 方法

### 2.1 目标与先验

- **搜索目标**（不重新采集，直接用包里已有的软标签）：
  - 包络层：`value_slots / value_values`（每个种子槽的最优搜索值 + wait 槽）→
    `softmax(values / T)`，`T = soft_temperature = 0.05`（与训练里的 cell 软目标同一温度）；
  - 格子层：`cand_entries` 中被标注卡片的候选格（最多 12 个）→ `softmax(cand value / T)`。
- **学生先验**（当前生产学生 r52，
  `/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/round_000052/student/model.pt`，
  width 64 / meanmax / row_pool / identity / v2timer，2026-10-30 09:03 晋升）：
  - 包络先验：`[P(plant)·softmax(packet), 1−P(plant)]`，在“搜索比较过的槽”上重新归一；
  - 格子先验：cell head 在该行被标注卡片的候选格上做 mask softmax。

### 2.2 每行 KL

```
KL_packet(r) = Σ_s  t_s · (log t_s − log p_s)          s ∈ 比较过的包络槽（≥2 个）
KL_cell(r)   = Σ_c  t_c · (log t_c − log p_c)          c ∈ 被标注卡片的候选格（≥2 个）
kl(r)        = KL_packet(r) + KL_cell(r)              （各自只在该层有定义的行非零）
```

包络层和格子层是同一个动作空间的两级分解：包络层衡量“等 / 种哪张卡”的分歧，
格子层衡量“种在哪”的分歧。两层都要求至少 2 个候选，单候选行的 KL 恒为 0，不参与加权。

缓存：一次前向（4,347,701 行，MPS，约 3.5 分钟），写一个行对齐的 npz
（`float16` KL + 两个 mask + meta）。不重放、不重采集、不改包。

### 2.3 采样混合

设训练集（去掉验证行后）为 `N`，每轮（epoch）的抽样索引：

```
perm = 均匀抽 N/2 行（不放回）
     ∪ 按 p = (1−share)/N + share·kl/Σkl 抽 N/2 行（有放回）
perm = shuffle(perm)
```

`share = kl_share = 0.5`，即 KataGo 的“一半均匀、一半按惊讶加权”。注意 `p` 自带均匀底
`(1−share)/N`，所以加权的一半也不是纯按 KL 比例抽；这与 KataGo 的频率权重一致。

实现位置：

- `python/pvz_recipe.py`：新增 `kl_weights`（路径）与 `kl_share`（[0,1]）两个字段；
- `scripts/train_rollout_policy.py`：在上述 `train_idx` 上构造 `kl_probs` 并做混合抽样；
- `scripts/kl_policy_surprise.py`：生成 KL 缓存（`--recipe --student --out --fingerprint`）。

## 3. 数据与预算（E3 口径）

数据：`~/pvz-bench-data/r42pack/` 的 r42 段（fingerprint `abdb7e217e0c5ea5`）。
沿用 E3 的划分（`~/pvz-bench-data/e3v2/split.json`，按对局/标签哈希分，固定种子）：

| 集合 | 标签数 | 行数 |
|---|---|---|
| train.joint（训练） | 575 | 4,347,701 |
| eval day | 28 | 356,396 |
| eval night | 40 | 570,562 |
| eval pool | 52 | 415,616 |
| eval fog | 30 | 137,133 |
| eval roof | 28 | 67,531 |
| eval random | 4 | 14,985 |

训练集是 r42 段按 E3 训练标签剪过的索引（`e3v2/train/joint` 的段全是
`r42pack/segments` 的软链），评估集的对局与训练集不相交。

预算（与 E3 joint64 完全一致）：4500 步、batch 2048、AdamW lr 0.002 cosine、
width 64 / meanmax / row_pool / identity / v2timer、`cell_soft_weight=3.0`、
`spread_weight=8.0`、`packet_balance=0.5`、`wait_keep=0.1`、验证 10%（`validation.seed=0`）。
种子 1 和 2，两个臂各两个种子。每臂约 8~9 分钟（MPS），四臂串行。

## 4. KL 缓存统计（r52 先验）

| 项 | 值 |
|---|---|
| 总行 | 4,347,701 |
| 包络层有定义（≥2 槽） | 3,108,724（71.5%） |
| 格子层有定义（≥2 候选） | 1,818,003（41.8%） |
| KL > 0 | 3,108,684 |
| 均值 | 1.02 |
| p50 / p90 / p99 / max | 0.80 / 2.14 / 4.11 / 13.14 |

混合后的集中度（`share=0.5`）：最高 1% 的行占 3.7% 概率质量，最高 10% 占 23.6%。
加权是温和的，不会把一半的抽样全压在少数行上。

训练集里植物行占 49.4%；加权一半里占 55.7%；混合后 52.6%（+3.2 pp）。
这就是 §5 里 `plant_recall` / `wait_false_plant` 一起变高的直接原因。

## 5. 结果

### 5.1 离线留出（E3 eval 包，按对局簇 bootstrap）

`type_acc`（五个地形 + random）：

| 臂 | day | night | pool | fog | roof | random |
|---|---|---|---|---|---|---|
| uniform_s1 | 0.7898 | 0.7947 | 0.8581 | 0.8868 | 0.8247 | 0.7493 |
| uniform_s2 | 0.7889 | 0.7957 | 0.8569 | 0.8840 | 0.8180 | 0.7300 |
| kl_s1 | 0.7878 | 0.8083 | 0.8661 | 0.8905 | 0.8249 | 0.7307 |
| kl_s2 | 0.7902 | 0.8103 | 0.8641 | 0.8914 | 0.8249 | 0.7060 |

五地形宏平均：

| 臂 | type_acc | plant_recall | wait_false_plant | cell_acc |
|---|---|---|---|---|
| uniform_s1 | 0.8308 | 0.7551 | 0.1077 | 0.4126 |
| uniform_s2 | 0.8287 | 0.7514 | 0.1079 | 0.4012 |
| kl_s1 | 0.8355 | 0.8070 | 0.1456 | 0.4033 |
| kl_s2 | 0.8362 | 0.8050 | 0.1419 | 0.4126 |

KL − uniform（同种子，百分点）：`type_acc` day −0.20 / +0.14、night +1.37 / +1.45、
pool +0.80 / +0.72、fog +0.38 / +0.74、roof +0.02 / +0.69、random −1.86 / −2.40。
两个种子同号，所以这不是噪声；但幅度只有 0.5~0.75 pp（宏平均），且方向在地形间不一致。

配对的 `recall` / `false_plant` 变化（两个种子一致、方向稳定）：

```
plant_recall      +0.093 / +0.096 (day)  +0.054 / +0.054 (night)  +0.052 / +0.050 (pool)
wait_false_plant  +0.065 / +0.061 (day)  +0.032 / +0.030 (night)  +0.031 / +0.031 (pool)
```

`cell_acc` 宏平均基本持平（0.413/0.401 → 0.403/0.413），格子层没有可辨识变化。

复现基准：`uniform_s1` 与 E3 `joint64`（同配方、种子 1）逐地形差 ≤0.20 pp，
说明本次实验口径与 E3 对齐。生产学生 r52 在同一批包上的 `type_acc` 是
0.813 / 0.833 / 0.879 / 0.913 / 0.838（day..roof），高于四个 4500 步臂，符合预期
（r52 训练得更久），也说明四个臂之间比的是“同一预算下的相对收益”。

### 5.2 真打小面板（bench_play，五地形，配对种子）

面板：`configs/gate_v5_panel.json` 的 day/night/pool/fog/roof 子集（9/9/9/9/2 关），
每关 4 个种子 = 152 局/臂；两臂用同一批种子（`--seed-start 900000000`），cpp 后端。
Δ = KL − uniform（百分点，Newcombe 95% CI，McNemar 精确 p）：

| 种子 | overall | day | night | pool | fog | roof |
|---|---|---|---|---|---|---|
| 1 | +0.7 [−7.6, +8.9] p=1.0 | +19.4 [+1.4, +36.6] p=0.065 | **−33.3** [−50.2, −12.5] p=0.004 | +16.7 [−2.2, +33.7] p=0.15 | 0 | 0（两臂均 0/8） |
| 2 | **+11.2** [+3.2, +18.9] p=0.0095 | +11.1 [−6.6, +28.3] p=0.34 | +5.6 [−13.4, +23.9] p=0.77 | **+22.2** [+3.5, +39.1] p=0.039 | +8.3 [−4.4, +20.5] p=0.38 | 0（两臂均 0/8） |

原始胜率：种子 1 85/152 vs 86/152；种子 2 72/152 vs 89/152。304 局合计 157 vs 175
（51.6% → 57.6%）。

读法：

- 池子在两个种子上都是 KL 更好（+16.7 / +22.2 pp），日间两个种子都为正（+19.4 / +11.1），
  与“种得更凶、白天阳光更宽裕”的机制一致；
- 夜间在两个种子间翻向（−33.3 / +5.6），其中一个还落在 p<0.01——36 局/区的噪声太大，
  不能据此说夜间变差，但也不能说有改善；
- 面板屋顶只有 2 关 8 局，四个臂全负，没有信息量。

## 6. 讨论：这个 KL 与 KataGo 的不是同一个东西

KataGo 的策略先验覆盖全部着点，包括 pass，所以“按 KL 加权”不会系统性地改变某一类
着点的先验频率。本仓库的包里，KL 只在“搜索比较过 ≥2 个包络槽”的行上定义，
而这些行几乎全是种植物行（等待行没有可比候选）。于是加权一半天然偏向植物行：
占比 49.4% → 52.6%，训练目标里的植物先验被抬高，这就是 `recall` 与 `false_plant`
同向上升、而格子层能力没有变化的来源。

换句话说：**离线指标的小幅提升，很大一部分只是把操作点往“多种”挪了**。
在决策序列上，“多种”在白天/泳池可能是有利的（更早建立火力），在夜间的阳光/墓碑
约束下是有代价的——这正是面板里 day/pool 为正、night 出现大负值的方向。

## 7. 结论与后续

- **不把 KL 加权并入生产配方**（现在）。离线宏平均 +0.5~0.75 pp、方向不一致，
  真打面板一个种子无差异、一个 +11.2 pp、分区翻向；证据强度不足以改动生产采样，
  而改动会牵动全部历史证据的可比性（AGENTS §8.4 的口径）。
- **P1：类别保形的 KL 臂。** 把加权一半按植物/等待两类分别归一（保持 49.4% 的植物占比），
  只让 KL 在类内分配质量。这样能分离“策略惊讶”和“植物先验平移”两种效应。
  2 个种子，同预算；若仍无可辨识收益，这条线可以关掉。
- **P1：面板最小规模。** 本次 152 局/臂在 10 pp 量级的效应上已经打满 CI；要分辨
  5 pp 需要 ≥4 个种子 × 全部五区（约 300 局/臂），并且夜间要单独看。
- **保留的实现。** `kl_weights` / `kl_share` 是配方字段（不影响未设置它们的旧配方），
  `scripts/kl_policy_surprise.py` 是可复用的“一次前向、不重采集”的惊讶缓存工具，
  后续别的加权方案（如 value 惊讶、能力分层）可以直接复用。

## 8. 复现

```bash
# 1) KL 缓存（一次前向，4.35M 行约 3.5 分钟）
python3 scripts/kl_policy_surprise.py \
  --recipe ~/pvz-bench-data/e3v2/recipes/joint64.json \
  --student ~/pvz-bench-data/task176/r52_student.pt \
  --out ~/pvz-bench-data/task176/kl_r52.npz --fingerprint abdb7e217e0c5ea5 --device mps

# 2) 四臂（kl 臂带 kl_weights + kl_share=0.5；uniform 臂两者为空/0）
python3 ~/pvz-bench-data/task176/make_recipes.py
python3 ~/pvz-bench-data/task176/run_train.py --recipe ~/pvz-bench-data/task176/uniform_s1.json
python3 ~/pvz-bench-data/task176/run_train.py --recipe ~/pvz-bench-data/task176/kl_s1.json

# 3) 离线评估（E3 固定五地形 + random）
python3 ~/pvz-bench-data/task176/eval_kl.py --run ~/pvz-bench-data/task176/runs/kl_s1

# 4) 真打小面板（两个臂用同一批种子，再配对比较）
cd ~/PvZAgent/.claude/worktrees/archbench
python3 scripts/bench_play.py --model ~/pvz-bench-data/task176/runs/kl_s1/model.pt \
  --out ~/pvz-bench-data/task176/bench/kl_s1 --zones day,night,pool,fog,roof \
  --per-task 4 --workers 3 --allow-partial
python3 scripts/bench_play.py --compare \
  ~/pvz-bench-data/task176/bench/uniform_s1 ~/pvz-bench-data/task176/bench/kl_s1
```
