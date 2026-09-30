# T5 重跑指令（2026-09-30，用户裁定"优化立刻全做，合并回 env 分支再重跑"）

**执行者：台式机上的 Agent。逐条执行，不要跳步，不要自行放宽。**

本指令替代 `T5_OVERNIGHT_ORDER.md` 的 §5 起跑部分；那份文件的 §M1–§M5 硬阻断仍然有效，
其中 **M5 是本次新增的**（见 `T5_OVERNIGHT_ORDER.md` §M5）。

---

## 0. 为什么上次失败了（读一遍，避免重犯）

**已确认的事实：** 上一次阶段 0 跑到 **75 分钟仍没有落下第一个 optimizer 更新**，
`cumulative_episodes` 始终为 0。按预算第一个更新只要约 6 分钟。
执行者的日志里写着"当前环境已明确缺少 GPU 设备，无法恢复原 RTX 训练通路"，
但它选择继续等 3 小时止损线。

**主要假设（尚未证实，§3 会让你证实）：** WSL 长开机后 GPU 掉线，而
`resolve_device("auto")` 在 `torch.cuda.is_available()` 为假时**静默回退到 CPU**。
训练因此没有崩、只是退化：rollout 照旧在 18 个 CPU worker 上跑，而
`training_state.json` 只在更新落盘后才重写 —— 进程看起来还活着，实际寸步未行。
实测代价：同一批 2,000 局更新，RTX 5080 上 184.4 s，CPU 上 1,072.6 s，**慢 5.8 倍**。

**为什么这条只是假设**：`T5_OVERNIGHT_ORDER.md` §4.2 给的命令是 `--device cuda`，
而 `resolve_device("cuda")` 在 CUDA 不可用时是**直接 `raise ValueError`**，不会静默回退。
所以要么实际执行时用的是默认的 `--device auto`，要么 GPU 是在起跑**之后**才掉线的。
**§3 的 `hyperparameters.json` 里记录了当时的 argv，读出来就知道是哪一种 —— 必须写进报告。**

无论哪种，本次新增的 M5 硬阻断都覆盖了 `auto` 回退这条路；GPU 在跑动中掉线则会
以 CUDA 报错的形式崩掉，而不是空转。

**执行者当时的日志里已经写出"缺少 GPU 设备"，却选择继续等。**
这是把"按流程办事"凌驾于"报告异常"之上。**本次凡遇到与预算不符的静默，先报告，再等。**

---

## 1. 前置：恢复 GPU（用户手动做，Agent 不要尝试修 WSL）

1. 用户手动重启 WSL（已知 bug：长开机后 GPU 会掉线，只能重启恢复）。
2. Agent 确认 GPU 真的回来了，**不通过就不要继续**：

```bash
nvidia-smi
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
```

**必须看到 `cuda: True` 和设备名（应为 NVIDIA GeForce RTX 5080）。**
看不到就停下、写清楚、报告，不要试图用 CPU 顶。

---

## 2. 拉取代码

```bash
cd <仓库根目录>
git fetch origin
git checkout pvz-env
git pull --ff-only origin pvz-env
git log --oneline -3     # 应看到 5ab68ae 在顶部
```

合并进来的内容（`6095c24` → `5ab68ae`）：

| 提交 | 内容 |
|---|---|
| `eca9a4d` | 关系偏置索引跨层提升 + 可选融合 |
| `ec62593` | 全流程热点分析 |
| `8d5bdc8` | **M5 硬阻断**：CUDA 掉线时禁止静默 CPU 更新 |
| `5ab68ae` | **P0+P1 优化**：字节摘要、评估并行化、参考集后置、融合默认开启 |

**协议与观测版本未变**（`MODEL_ARCHITECTURE_VERSION = 5`、`OBSERVATION_VERSION = 3`），
不需要重新编译 PvZ 二进制。

**`episode_digest` 改变了 `trajectory_sha256` 的取值** —— 这正是必须在 T5 产出任何结果
之前定死的原因。合并后再跑，整轮证据才自洽。

---

## 3. 诊断当前状态（**必须先做，不要直接删**）

上次失败留下了一个**已消耗但毫无产出**的正式 run 名额。先看清楚再决定：

```bash
ls -la artifacts/t5/
python - <<'PY'
import json, pathlib
p = pathlib.Path("artifacts/t5/training_state.json")
if not p.exists():
    print("training_state.json 不存在 -> 这是一次干净起跑（run 1）")
else:
    s = json.loads(p.read_text())
    print("formal_runs     =", s.get("formal_runs"))
    print("cumulative_eps  =", s.get("cumulative_episodes"))
    print("stop_reason     =", s.get("stop_reason"))
    print("evaluations     =", len(s.get("evaluations", [])))
    for r in s.get("runs", []):
        print("  run", r.get("run_number"), "status =", r.get("status"),
              "updates =", r.get("updates"), "checkpoint =", r.get("checkpoint"))
PY
ls -la artifacts/t5/runs/ 2>/dev/null
find artifacts/t5/runs -name "gameplay_model_v1_ppo.pt" 2>/dev/null
```

**还要把当时真正跑的命令读出来**（`run_config["command"]["argv"]` 会原样记录 argv，
这能证实 §0 里的假设，必须写进报告）：

```bash
python - <<'PY'
import json, pathlib
for path in sorted(pathlib.Path("artifacts/t5/runs").glob("run_*/hyperparameters.json")):
    argv = json.loads(path.read_text()).get("command", {}).get("argv")
    print(path, "->", argv)
PY
```

重点看 `--device`：如果是 `auto` 或没写，就是静默回退 CPU；如果写的是 `cuda`，
说明 GPU 是起跑之后才掉线的，要按另一条线追查。

**判据：**

- 若 `formal_runs == 0`、没有 `runs/` 目录 → 直接跳到 §5，用 §5 的命令起跑。
- 若 `formal_runs >= 1`，且**没有任何 `gameplay_model_v1_ppo.pt`**、`cumulative_episodes == 0`
  → 那次运行零产出（这正是上次的情况）。走 §4 清理。
- 若**存在 checkpoint 且 `cumulative_episodes > 0`** → **停下报告**，不要清理，让用户裁定。

---

## 4. 清理零产出的失败 run（仅在 §3 判据命中时执行）

**只删这几项，不要碰别的**：

```bash
# 先备份，再删（备份放在仓库外，避免污染工作区）
mkdir -p ~/t5_aborted_backup
cp -a artifacts/t5/training_state.json ~/t5_aborted_backup/ 2>/dev/null
cp -a artifacts/t5/learning_curve.json  ~/t5_aborted_backup/ 2>/dev/null
cp -a artifacts/t5/runs                  ~/t5_aborted_backup/ 2>/dev/null

rm -f  artifacts/t5/training_state.json
rm -f  artifacts/t5/learning_curve.json
rm -rf artifacts/t5/runs
rm -rf artifacts/t5/evaluations
```

**必须保留**（它们是证据，删了训练起不来）：

- `artifacts/t5/throughput.json` —— 训练脚本会读 `single_core_threshold_met`，缺了直接报错
- `artifacts/t5/perf/` —— **M5 要读 `perf/ppo_update_2000_flex_saved.json` 判断更新设备**
- `gates/` 下的全部文件（尤其 `gates/T4.json` 与 `gates/T5_stage0_criteria.json`）
- `artifacts/task_family/` 下的冻结任务清单

**理由**：那次 run 一个更新都没落盘，没有任何可保留的训练证据；而它把
`formal_runs` 抬到了 2，会让本次重跑被当成 run 3 —— 而 run > 1 时脚本会去加载
`runs/run_2/gameplay_model_v1_ppo.pt` 续训，那个文件根本不存在，必然崩。
清掉之后重跑就是干净的 run 1，不需要 `--motivation`。

---

## 5. 起跑阶段 0

```bash
cd <仓库根目录>
python python/train_pvz_ppo_task_family.py \
  --device cuda --curriculum cap1 \
  --max-episodes-per-run 10000 \
  --minibatch-chunks 16 --learning-rate 1e-4 --attention-backend auto \
  --initialization-note "lane-token architecture (v5) supersedes the T4 seed-0 baseline; T4 hash recorded in gates/T4.json" \
  --output-dir artifacts/t5
```

要点：

- `--curriculum cap1` —— 阶段 0 只训 5 个 cap1 任务。
- **`--max-episodes-per-run 10000` 是阶段 0 的预算**（`gates/T5_stage0_criteria.json`
  的 `episode_budget`）。脚本默认是 20000，不显式覆盖就会跑成 20000 局。
- `--device cuda` —— **显式写死**。这样 CUDA 不在时 `resolve_device` 会立刻
  `raise ValueError`，比 `auto` 更早暴露问题。
- `--initialization-note` 是必需的：网络结构版本 5 > T4 的 4，脚本会拒绝无说明的起跑。
  上一次跑动时必然也传了（否则起不来），可以复用当时 `hyperparameters.json` 里的原文。
- `--workers` 默认取 `throughput.json` 的 `selected_parallel_workers`（18）。
- **不要**传 `--allow-cpu-update`。它只是逃生口，不是常规选项。

### 前 10 分钟必须盯住的三件事

| 时间 | 应看到 | 看不到就是异常 |
|---|---|---|
| 起跑后 < 1 min | 打印 `T5 run 1 update 1 ...` 之类的 rollout 进度 | 立刻停下，看 stderr |
| 约 3 min | rollout 完成 2,000 局（`... 2000/2000`） | 停下报告 |
| **约 6 min** | **第一个 update 落盘**：`run=1 update=1 episodes=2000 ...`，且 `training_state.json` 里 `cumulative_episodes` 变成 2000 | **停下报告**（上次就是卡在这里空转 75 分钟） |

一次更新（2,000 局）的预算：rollout 约 162 s + 更新约 184 s + 固定开销约 18 s ≈ **6 分钟**。

---

## 6. 卡住判定与止损

沿用 `T5_OVERNIGHT_ORDER.md` §8：

- 阶段 0 预算 **约 1 小时**（10,000 局 = 5 次更新 + 2 次评估）。
- **超过预算 3 倍（约 3 小时）仍未结束 → 视为卡住**：停止、写清楚卡在哪、push 证据。
- 若 **第一个更新在起跑 30 分钟后仍未落盘** → 提前触发卡住判定，不要再等。
  （M5 挡掉了 `--device auto` 静默回退 CPU 这一条；若仍然发生，说明是别的原因，
  需要报告而不是继续等。）

阶段 0 的早期零信号硬停止（M1）仍然有效：
每 5,000 局评估时若 `stage0_set.pass_rate == 0.0` **且** 5 个 cap1 任务的滚动 64 局胜率全为 0，
立刻 `stop_reason = "stage0_no_signal"`、落盘、退出，**不跑满 10,000 局、不进阶段 1**。

---

## 7. 本次优化了什么（执行者不必改代码，但要知道数字从哪来）

详见 `TRAINING_HOTSPOT_ANALYSIS.md` §9。要点：

| 项 | 效果 | 验证方式 |
|---|---|---|
| `episode_digest` 字节摘要 | 固定开销 18 s/update → 约 0.6 s | `python/test_episode_digest.py`（13 项） |
| 评估 18 worker 并行 | 评估从串行变并行 | `scripts/evaluation_parallel_equivalence.py` **逐条逐位相同** |
| 中间评估跳过参考集 | 每次评估省 25% | `test_t5_overnight.EvaluationTests` |
| 关系偏置融合默认开启 | rollout 1.14–1.31× | `scripts/relation_bias_benchmark.py` |

**注意**：融合**不作用于 PPO 更新**（更新走 CUDA FlexAttention 的 `score_mod`，不走融合路径）。
所以更新那部分的时间与上次相同，这是预期，不是回归。

**评估每次有约 8.8 s 的进程池固定开销**（进程启动 + 3.68M 参数分发给 18 个进程），
这是预期的，不是 bug。

---

## 8. 报告要求

结束时写 `T5_RERUN_REPORT.md`，**必须包含**：

1. §3 诊断命令的**原始输出**（证明清理是有依据的），**包括当时 argv 里的 `--device`**；
2. 起跑命令原文 + `nvidia-smi` / `torch.cuda.is_available()` 的输出；
3. 第一个 update 落盘的实际耗时（与 6 分钟预算对比）；
4. 每 5,000 局的 `stage0_set.pass_rate` 序列；
5. 终止原因、实际局数、`formal_runs` 的最终值与剩余名额；
6. **未采用的方案 + 实测数据**（沿用本仓库"诚实记录"的做法）；
7. **明确写出仍未验证的假设**。

---

## 9. 红线（违反即视为失败）

- R1 不修改 `gates/` 下任何文件、`artifacts/task_family/` 下任何冻结清单。
- R2 不传 `--ignore-stage0-gate` 进阶段 1。
- R3 不传 `--allow-cpu-update` 来"绕过" M5。
- R4 阶段 0 零信号时不得跑满预算、不得进阶段 1。
- R5 不得为了"让门禁过"而修改判据（`gates/T5_stage0_criteria.json` 是预注册的）。
- R6 遇到与预算不符的静默，**先报告再等**，不得空转到止损线。
