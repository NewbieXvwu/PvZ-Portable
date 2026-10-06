# 行为克隆流水线（归档）

**这一批脚本是 2026-10-05 那次行为克隆实验的原始脚本，从 `/tmp` 抢救进仓库的。**
它们不是干净的库代码——**里面全是硬编码的绝对路径**（本机 macOS 路径、`/tmp` 目录）。
放在这里是为了**可复现**：没有它们，"BC 能到 80.86%" 这个结论就没有出处。

## 它产出了什么

| 项 | 值 |
|---|---|
| 教师 | `scripted_baseline_d1.py`（第 7 关、d1 卡组 `[0,1,2,3,4,5,7]`）——**256 种子 87.11%** |
| 示范 | 5,000 局 / 4205 胜 / 464 万决策 / **合法动作 100%** |
| 数据集 | 100 万条（98 万训练 / 2 万验证），wait:plant = 2.75:1 |
| 学生 | 与主 RL 完全同配置（6 层/256/8 头/ff 1024/gru 2×256/input_flags 7/events/progress_v1）|
| 验证准确率 | **plant 95.39%** / wait 99.76% / 卡包条件 98.99% |
| **结果** | 已见种子 215/256 = 83.98%；**未见种子 207/256 = 80.86%** |

对照：RL 从零训 25 万决策在同一任务上 **0%**。
**→ 同一架构、同一观察编码：BC 80.86%，RL 0%。架构和观察编码不是瓶颈。**

完整报告：`artifacts/t5/perf/bc_unblock_2b_v1.json`。

## 执行顺序

```bash
cd <仓库根>
# 1) 采集示范（用 d1 教师，5,000 局）→ 产出 sqlite
python scripts/bc_pipeline/collect_demonstrations.py
# 2) 训练 BC（默认 50 轮，按验证损失早停）→ 产出 best_model.pt
python scripts/bc_pipeline/train_bc.py
# 3) 评估（未见种子 30000-30255）
python scripts/bc_pipeline/evaluate_bc_cpu.py
```

## 本机路径

归档脚本最初使用 macOS 和 `/tmp` 硬编码路径。本机采集、前馈训练、CPU 评估和序列训练入口已改为从仓库位置和 `$HOME` 推导路径：

```python
ROOT       = Path(__file__).resolve().parents[2]
OUT        = Path.home() / 'PvZAgent-gru-bc-level7-v1'
TEACHER    = ROOT / 'scripts/bc_pipeline/scripted_baseline_d1.py'
RESOURCE   = Path.home() / '.cache/pvz-research-resources'
CHECKPOINT = Path.home() / 'PvZAgent-bc-handoff/best_model_pipeline.pt'
```

其余历史对照脚本仍可能含归档时的旧路径，运行前需检查。

## 各文件是什么

| 文件 | 原来在哪 | 说明 |
|---|---|---|
| `collect_demonstrations.py` | `/tmp/pvz_bc_collect_v2.py` | 采集示范 → 分片 sqlite → 合并 |
| `train_bc.py` | `/tmp/pvz_bc_train_eval.py` | **正式的那次训练**（读 v2 sqlite）|
| `train_bc_v1_superseded.py` | `/tmp/pvz_bc_train.py` | 第一版训练器，**已被上面那个取代**，留作沿革 |
| `evaluate_bc_sampled.py` | `/tmp/pvz_bc_eval_sampled.py` | 采样评估（第一版用的）|
| `evaluate_bc_cpu.py` | `/tmp/pvz_bc_2b/evaluate_cpu.py` | CPU 评估（正式那次用的）|
| `eligibility_check.py` | `/tmp/pvz_unblock_bc_2b.py` | E0/E1/E2 等待语义对照（`PvZEnv` vs `EventWaitEnv`）|
| `stop_after_training.py` | `/tmp/pvz_bc_2b/stop_after_training.py` | 训练早停钩子 |
| `scripted_baseline_d1.py` | `/tmp/pvz_deck_v2/scripted_baseline_d1.py` | **d1 教师**。与 `scripts/scripted_baseline.py` 的差异只有：射手上限 14→20、加 `DOUBLEPEA=7` 的规则 ⑤b、`POLICY_REVISION` 改名 |
| `train_bc_gru_sequence.py` | 本轮新增 | 按 episode/step 顺序训练，在 32 步分块间传递 GRU 隐状态 |

## 不要提交的东西

- **数据集 sqlite 有 10 GB**（分片还有 16 GB）。**不要入库，也不要传 HF**——
  **有脚本就能重建**（采集约 8 分钟，8 workers）。
- 训练出的 `.pt` 按项目约定走 Hugging Face（`realnewbiexvwu/pvz-agent-artifacts`，
  当前在 `bc-scripted-level7-v2/`）。

## GRU 序列对照

采集器保留每条样本的 `episode_id` 和 `step_id`，序列训练入口按它们排序，跨决策和 32 步分块传递隐状态。采集完成后运行：

```bash
python scripts/bc_pipeline/train_bc_gru_sequence.py
```

历史脚本没有单元测试。
