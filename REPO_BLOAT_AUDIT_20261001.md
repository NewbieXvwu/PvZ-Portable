# 仓库囤积审计（2026-10-01）· 判据：现在有没有用

> 本文件是一次性审计报告。**按本文件执行完处置后请直接删除它**，否则它自己就成了第 22 份囤积。
> 本次审计**未删除、未移动、未修改任何文件**，只做了读取与统计。

## 0. 判据修正

上一版用"代码有没有引用"判定，**太窄**。这个仓库的读者是 Agent 和人，研究还在推进中，
不是成品仓库。改成：

> **推进下一步时，需不需要读它 / 需不需要加载它？**

用三条硬证据判定，不靠猜：

| 证据 | 回答的问题 |
|---|---|
| `resume.json` 指向哪个文件 | **要接着训练必须加载什么** |
| 结论层文件（学习曲线/评估/指纹/报告） | **Agent 读了才能决策的是什么** |
| 引用链是否完整 | **有没有东西谁都不指向** |

---

## 1. 政策裁决：哪份规定还在生效

你说不知道哪些规定被推翻了。这个可以**用机制本身测出来**，不用猜：

```bash
# 把当前所有已跟踪文件，逐个喂给现行的 .gitignore
git ls-files -z artifacts logs | git check-ignore -z --stdin --no-index
```

结果：

> **765 个已跟踪文件里，659 个（86%）会被现行 `.gitignore` 拦下 —— 合计 2.0 GB。**

这说明 **规则从未失效**。这 659 个文件是历次 `git add -f` 的强制覆盖，不是"规则之前就有的遗留"。

时间线（`git log` + `git show` 实测）：

| 时间 | 事件 |
|---|---|
| 09-30 19:40 | `artifacts/.gitignore` 写入"**检查点（.pt）和分片（runs/ 下的 .npz）始终不入库**" |
| 09-30 18:55 | `DESKTOP_HANDOFF.md` §83 写入"`.pt`/`.npz` ❌ 不入库" |
| **09-30 23:14** | 提交 `fc20609` **仍然推入** `replay_audit_v1/collection_policy.pt`（14.7 MB）+ 20 个 `.npz` |
| 10-01 09:26 | 新增 `CHECKPOINT_GIT_DELIVERY.md`，把"分片通过 Git 提交"写成常规交付做法 |

**裁决**：

- ✅ **仍然生效**：`DESKTOP_HANDOFF.md` §83 + `artifacts/.gitignore`
- ❌ **应删**：`CHECKPOINT_GIT_DELIVERY.md`。注意**不是因为它说谎** ——
  它描述的工具（`archive_research_evidence.py` + `research_checkpoint_chunks.py`）
  是真实且测试过的（验证记录 `artifacts/t5/perf/checkpoint_chunk_audit_v1.json`），
  100 MB 单文件闸门和 64 MiB 分片也确实存在于代码里。
  问题在于：闸门按**单文件 ≥ 100 MB** 判定，而检查点只有 44 MB，
  所以 `.gitparts/` 从未出现（实测：仓库里 0 个），
  43 MB 的检查点是**整个文件**被 `git add -f` 提交的。
  这份文档把一个**从未生效的兜底机制**写成了交付通道，等于给违规行为提供了一份
  看起来合规的说法。工具保留，文档删除，恢复步骤移到 `DESKTOP_HANDOFF.md` §2.3。

**副作用（顺带发现，是真问题）**：`.gitignore` 白名单只保护 `adventure2_level7/seeds/`，
`artifacts/multiterrain/seeds/*.json` **新增会被忽略**（实测），而代码要读这 5 个地形种子。
→ 新增地形种子另一台机器拿不到。白名单需要补。

---

## 2. 现行规则认可的交付集只有 9.4 MB

那 106 个"合规在库"的文件合计 **9,616 KB**，全部是 KB 级 json/txt：

```
artifacts/t5/throughput.json            ← 训练入口直接读
artifacts/t5/evaluation_parallel_equivalence*.json
artifacts/t5/curves/*.json
artifacts/t5/perf/*.json  *.txt         基准与诊断结论
artifacts/adventure2_level7/seeds/*.json
```

**实际交付了 2.0 GB，超标 218 倍。**

---

## 3. 四层分类（按"现在有没有用"）

| 层 | 体积 | 判据 | 处置 |
|---|---|---|---|
| **L1 结论层** | **19 MB** | Agent 读了才能决策 | **一个都不能删** |
| **L2 续跑层** | **429 MB** | 精确等于所有 `resume.json` 的指向，12 个检查点 | 只在还要接着训练时需要 |
| **L3 日志层** | **87 MB** | 滚动日志，摘要已进报告 | 可清 |
| **L4 死重量** | **1.43 GB** | **没有任何引用链指向它** | 可清 |

合计 ≈ 1.96 GB，与 `artifacts/` 实测 2.0 GB 吻合。

---

## 4. L1 结论层（19 MB）—— 这是"真正现在有用"的核心

每个证据目录里那 6%，逐目录实测：

| 目录 | 结论层 | 目录 | 结论层 |
|---|---|---|---|
| `reward_r0_seed0_v2` | 6.5 M | `trajectory_diagnostics` | 1.2 M |
| `reward_r1_seed0_v2` | 3.4 M | `t5a_smoke_v4` | 512 K |
| `reward_r0_seed0_v1` | 3.4 M | `aid_feasibility_v1` | 372 K |
| `reward_r1_seed0_v1` | 2.0 M | `aid_feasibility_v2` | 364 K |
| `observation_interrupt_v1_failed` | 360 K | 其余 11 个目录 | 各 < 300 K |

构成：`learning_curve.json`（学习曲线，这就是结论）、`evaluations/*.json.gz`（逐点评估）、
`provenance.json`（代码/模拟器/任务指纹）、`archive_manifest.json`、`resume.json`、
`experiment_config.json`、`training_state.json`。

**19 MB 承载了全部研究结论。剩下的 1.94 GB 是权重和日志。**

---

## 5. L2 续跑层（429 MB，12 个检查点）

这是"要接着训练就必须加载"的精确集合 —— 直接读 `resume.json` 得到，不是我猜的：

| 实验线 | 检查点 | 体积 |
|---|---|---|
| `reward_r0_seed0_v2` | `budget_complete` 43996K + `node375k` 43960K + `node250k` 43620K + `node125k` 43460K | 175 M |
| `reward_r1_seed0_v2` | `budget_complete` 43924K + `node250k` 43652K | 88 M |
| `reward_r0_seed0_v1` | `budget_complete` 43860K + `first_two_nodes` 43656K | 88 M |
| `reward_r1_seed0_v1` | `protocol_stop` 43636K | 44 M |
| `t5a_smoke_v4` | `update_000006_boundary` 43380K | 44 M |
| `observation_interrupt_v1_failed` | 2 个 1 MB 小检查点 | 2 M |

**判断建议**：v2 是当前线（10-01 10:34/10:58），保留；
v1 是历史对照线（10-01 00:53 / 08:47），如果确认不再回退，可降级为"结论层保留、检查点移出"。

---

## 6. L4 死重量（1.43 GB）—— 三块

### 6.1 归档不完整的冒烟目录：130 MB ← 本轮新证据

`t5a_smoke_v1/v2/v3` 各含一个 43 MB 的 `final_training_checkpoint.pt`。但它们的 `resume.json` 写的是：

```json
{ "checkpoint": "runs/run_1/update_000008_boundary.pt", "sha256": "df591c63..." }
```

而归档目录里**根本没有 `runs/` 目录**，只有一个 `final_training_checkpoint.pt`。

**结论：这三份归档的续跑链是断的。** 那个 43 MB 的文件：
- 不是 `resume.json` 指向的文件
- 不被任何门禁或报告按文件引用

→ **130 MB 完全孤立**，删掉不影响任何东西。

对照：`t5a_smoke_v4` 的 `resume.json` 指向 `runs/run_1/update_000006_boundary_...pt`，**该文件存在**（43380K）。
所以 v4 的归档是完整的，那 44 MB 有效。

### 6.2 逐字节重复的副本：496 MB

同一 update 被复制进多个进度快照目录，时间戳相同可判定同源：

```
reward_r0_seed0_v2/budget_complete/runs/run_1/update_000011_evaluated_1790816867278872855.pt
reward_r0_seed0_v2/node375k/runs/run_1/update_000011_evaluated_1790816867278872855.pt
reward_r0_seed0_v2/node125k/runs/run_1/update_000011_evaluated_1790816867278872855.pt
reward_r0_seed0_v2/node250k/runs/run_1/update_000011_evaluated_1790816867278872855.pt
```

15 个冗余副本。注意 `.git` 内部因内容寻址**已自动去重**（工作区 73 个 `.pt` → git 里 65 个 blob），
所以浪费的是**工作区磁盘**，不是 git。

### 6.3 其余中间检查点与评估分片

- 不在任何 `resume.json` 里的历史 update 检查点（如 `update_000000`、`update_000027` 等）
- `replay_audit_v1` 的 16.6 MB（`collection_policy.pt` 14.7 MB + 20 个 `.npz`）
- `observation_interrupt_v1_failed` 的 35 MB 评估分片（`.seed_jobs/**/*.npz`，`run_seed_jobs` 写的，可重跑再生）

---

## 7. L3 日志层（87 MB）

`artifacts/.gitignore` 原文写着"`.log` 不入库：滚动日志有 MB 级"。实际 74 个 `.log` 已入库。
最大几份：

```
13,408 KB  reward_r0_seed0_v2/budget_complete/logs/reward_r0_seed0_v2.log
13,324 KB  reward_r0_seed0_v2/node375k/logs/reward_r0_seed0_v2.log   ← 同一份日志的第二份
11,844 KB  reward_r1_seed0_v2/budget_complete/logs/reward_r1_seed0_v2.log
10,044 KB  reward_r0_seed0_v1/budget_complete/logs/reward_r0_seed0_v1.log
```

另有根目录 `logs/t5_research/` 46 个 `.log`（336 KB）已入库，而 `logs/` 现在也被 `.gitignore` 拦下。

---

## 8. 不要删的（看起来像囤积，其实有原则）

`TODO.md` §1.1/§1.3 明文要求：

> 3. 不能根据结果删掉失败种子或事后降低达标阈值……历史结果保持可追溯。
> 8. **失败要留下结果并诊断。**

所以下面这些**保留是对的**，动它们之前先想清楚：

| 项 | 体积 | 为什么留 |
|---|---|---|
| `observation_interrupt_v1_failed` | 36 M | §8 要求保留失败现场，且是当前正在处理的问题（11:30 提交） |
| `reward_*_v1` 的结论层 | 5.4 M | 历史对照，v2 的 baseline |
| `gates/T5-A-research.json` / `-v4.json` | — | 门禁证据，引用了 `t5a_smoke_v3` / `v4` 目录 |
| `late_policy_probes_v1`、`prefix_fix_v4`、`trajectory_diagnostics` | 2.6 M | 当前结论的依据 |

**要区分的是**：*结论层*（19 MB）不可替代；*检查点*（L2/L4）才是体积来源。

---

## 9. 执行方案

### 第一步：零风险（本地垃圾，22 MB，可逆）

```bash
rm -rf build/ .pytest_cache/ .DS_Store python/__pycache__/ scripts/__pycache__/
rm -f artifacts/t5/perf/worker_sweep_more.log artifacts/t5/perf/worker_sweep_critic.log \
      artifacts/t5/perf/worker_batch_sweep_2000.log artifacts/t5/perf/worker_batch_sweep_2000_mid.log
```

### 第二步：删孤立归档（130 MB，零引用）

`t5a_smoke_v1/v2/v3` 的 `final_training_checkpoint.pt` —— 续跑链已断，无引用。
保留各目录的 `evaluations/`、`learning_curve.json`、`provenance.json`（结论层）。

### 第三步：去重复副本（496 MB）

保留 `budget_complete` 下的那份，删 `node125k` / `node250k` / `node375k` 里的同名副本。

> **硬约束**：`archive_manifest.json` 记录了每个文件的 SHA256，`resume.json` 指向具体检查点。
> **不能零散 `rm`** —— 只能整目录处置，或重新生成 manifest。挑着删会让完整性校验链断掉。

### 第四步：移出 git（2.0 GB → 9.4 MB 合规集）

```bash
# 先确认台式机 artifacts/research/<candidate>/ 下仍有原始检查点
git rm -r --cached artifacts/research_evidence logs
echo 'artifacts/research_evidence/' >> .gitignore
echo 'logs/' >> .gitignore
```

> **前提**：本机 `artifacts/research/` 只有 reward_comparison 摘要（1.5 MB），**没有候选目录**，
> 所以本机这份很可能是当前唯一副本。移出前必须先确认台式机有原件。

### 第五步：清理文档

- 删 `CHECKPOINT_GIT_DELIVERY.md`（与生效规则冲突）
- 根目录 21 份 md 压回 8–10 份：0 引用的 `NIGHT_RUN_REPORT_20260930.md` 可删；
  `REWARD_LATE_UPDATE_PROBE_*` 一对可合并；`T5_*_PROGRESS_*` 并入 `TODO.md`；
  `TRAINING_DESIGN_PROPOSAL.md` 并入 `DESIGN.md`
- 删文档后必做：① 改掉 `python/`、`scripts/` 里对它的引用
  （`train_pvz_ppo_task_family.py` 的 `protected_paths` 会 SHA 校验 DESIGN/TODO）；
  ② 扫断链

### 第六步：修白名单（防复发）

把 `artifacts/multiterrain/seeds/` 加进 `artifacts/.gitignore`，否则新增地形种子同步不过去。

### 第七步（最后，可选）：重写 git 历史

`.git` 1.1 GB 里 `.pt` blob 原始 1465 MB。`git gc` **没用**（不可达对象仅 105 个）。
只能 `git filter-repo` + 强推，另一台机器需重新 clone，所有 commit hash 改变。
**等前六步稳定后单独安排。**

### 第八步

删掉本文件。

---

## 10. 台式机（WSL，`newbiexvwu@127.0.0.1:22222`）实测

### 10.1 不是"一个仓库"，是四条并行研究线

`git worktree list` 显示 4 个目录**共享同一个 `.git`**（object store 共用，所以三个从目录的 `.git` 只有 4 KB）：

| 目录 | 分支 | 体积 | 在做什么 |
|---|---|---|---|
| `~/PvZ-Portable` | `pvz-env` | **30 G** | 主线：奖励方案对比 |
| `~/PvZ-Portable-curriculum` | `research/curriculum-v1` | 916 M | 课程学习（先易后难的任务编排） |
| `~/PvZ-Portable-observation` | `research/observation-v1` | 1.2 G | 观测修复 |
| `~/PvZ-Portable-reset-disposal` | `research/reset-disposal-v1` | 355 M | 内存泄漏修复（每局泄漏 1.2 MiB） |

**它们是四条独立的实验线，不是副本，不能当冗余删。**

### 10.2 30 G 的构成

| 目录 | 体积 | 入库状态 |
|---|---|---|
| `artifacts/research` | **23 G** | **未跟踪**（`artifacts/.gitignore:2:*`，实测确认） |
| `artifacts/t5` | 2.5 G | 基准分片，部分未跟踪 |
| `artifacts/research_evidence` | 2.2 G | **已入库** |
| `.git` | 1.9 G | 比本机大 0.8 G（台式机提交更多） |
| 源码 / 构建 | ~0.4 G | |

`artifacts/research` 里 **`.pt` 检查点 16.6 GB / 417 个**。

### 10.3 根因在代码里，不在使用习惯

`python/pvz_research.py:358`：

```python
# Keep every snapshot immutable, including multiple resumes at the same
# update. The pointer alone changes; crash evidence is never overwritten.
path = output / "runs/run_1" / f"update_{state['updates']:06d}_{phase}_{time.time_ns()}.pt"
```

四个问题叠加：

1. **文件名带纳秒时间戳** → 每次保存都是新文件，永不覆盖。
2. **没有任何保留策略** —— 全仓库搜不到 `prune` / `keep` / `cleanup` / `max_checkpoints`。
3. `save("trained")` 在**每个 update** 都调用（`pvz_research.py:451` 与 `:465`）。
   一个 500k 决策的 run 存了 **73 个 `trained` 检查点**。
4. 注释里"崩溃证据永不覆盖"的理由，对**指针设计**成立，但**不能当作保留策略** ——
   要留崩溃现场只需要"最新一个 + 最近几个"，不需要 73 个。

单个 run 的检查点阶段分布（`reward_r0_seed0_v2`，共 80 个）：

| 阶段 | 数量 | 体积 | 是否必需 |
|---|---|---|---|
| `trained` | **73** | 3.1 G | 只有最后一个有用 |
| `evaluated` | 5 | 215 M | ✅ 分析脚本按这些节点取模型 |
| `boundary` | 1 | 43 M | ✅ 冻结边界 |
| `initial` | 1 | 43 M | ✅ 起点 |

**80 → 8，单 run 省 92%；全部 run 16.6 G → 约 1.3 G。**

### 10.4 完整因果链

```
每个 update 存一个 43 MB 检查点，永不删除
  → 单个 run 3.4 GB
  → 12 个 run 的计划（4 种奖励 × 3 次初始化）≈ 40 GB
  → archive_research_evidence.py 再复制一份到 research_evidence
     （脚本明说 "No source evidence is moved or removed"）→ 存储翻倍
  → Agent 再 git add -f 提交 → .git 再涨 1.9 GB
```

**归档脚本是复制不是移动，加上强制提交，同一个检查点最多同时存在三份。**

### 10.5 当前进度与正在跑的实验

`experiments/t5/reward_comparison_v2/queue.json` 定义了 **12 个 run**
（R0–R3 四种奖励 × seed0/1/2 三次初始化），每个 500k 决策，评估节点 125k/250k/375k/500k。

R0–R3 是一个 2×2 因子设计：

| | 折扣 γ=0.99 | 折扣 γ=1.0 |
|---|---|---|
| **带塑形奖励** | R0 | R3 |
| **不带塑形奖励** | R1 | R2 |

调度器 `run_reward_comparison.py`（pid 1177，已跑 4.2 小时）日志显示：

```
running 1/12 reward_r0_seed0_v2  ✅ 500,085 决策
running 2/12 reward_r1_seed0_v2  ✅
running 3/12 reward_r2_seed0_v2  ✅ 500,085 决策
running 4/12 reward_r3_seed0_v2  ✅
running 5/12 reward_r0_seed1_v2  🔥 正在跑（update 12）
```

**还有 7 个 run 要跑**，按当前策略会再产生约 24 GB 检查点。

另外：`run_reward_comparison.py` **不自动归档**（无 `archive` / `research_evidence` 引用），
归档是手动步骤。`reward_r3_seed0_v2` 至今**未归档**。

### 10.6 台式机处置方案

| 分组 | 体积 | 建议 | 可回收 |
|---|---|---|---|
| `reward_r0_seed1_v2`（正在跑） | 753 M | **不动** | — |
| 已完成的 seed0 v2（r0/r1/r2/r3） | 14.3 G | 每个 run 保留 `evaluated` 全部 + 最新 `trained`，删其余 72 个 | 9.8 G |
| v1 旧线（reward_*_seed0_v1） | 5.1 G | 已被 v2 取代 | 5.0 G |
| 冒烟 run ×4（t5a_smoke_r0_seed0_v1~v4） | 2.6 G | 冒烟阶段早已结束 | 2.5 G |
| 分片机制验证（checkpoint_chunk_*） | 445 M | 那套机制从未用于生产 | 445 M |

`artifacts/research` 全部**未跟踪**，清理不影响 git、不影响另外三个 worktree、不影响正在跑的队列。

---

## 11. 跨机同步检查点：换掉临时 SSH

需求：双机同步检查点、不依赖临时 SSH 隧道、最新成果要能随处下载。

**推荐：Hugging Face Hub**

- 免费，专为 ML 产物设计；台式机实测可访问（`huggingface.co -> 200`）。
- `hf upload` / `hf download`，**断点续传 + SHA 校验**，支持私有仓库。
- 从任何机器一行拉取，不需要隧道、不需要对方在线。
- 需要 `pip install huggingface_hub`（venv 里目前没有）+ 一个 access token。

**备选：GitHub Releases** —— 每文件 2 GB 上限（单个检查点 43 MB 远够），
下载走公开 URL，不需要额外账号，但需要写一个上传/下载的小脚本。

**不再推荐**：Git LFS（GitHub 免费额度 1 GB 存储 / 1 GB 月流量，装不下 43 MB × N）、
`git add -f` 提交检查点（就是当前问题的来源）。

---

## 12. 一句话总结

**19 MB 的结论层承载了全部研究结论；429 MB 的续跑层是"要接着训练才需要"的；剩下 1.5 GB 里，
130 MB 的续跑链已断、496 MB 是逐字节重复、87 MB 是自己规定不入库的日志。**

**台式机 30 GB 里，23 GB 是未跟踪的训练产物，其中 16.6 GB 是 417 个检查点 ——
而每个 run 的 80 个检查点里真正需要的只有 8 个。**

规则本来是对的，9.4 MB 就够跨机交付。真正要修的是
`pvz_research.py` 那个"每次保存都新建文件、永不清理"的存档策略 ——
不修它，清理完还会再长回来。
