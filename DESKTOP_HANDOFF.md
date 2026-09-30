# 台式机执行交接手册

本机（macOS）没有足够的算力跑真实实验，**主要执行机是台式机**：Windows 11 + WSL2（Ubuntu），
RTX 5080 16 GB，通过临时虚拟局域网 SSH 访问。这份文件是给在台式机上接着干的执行者（人或
Agent）看的，目标是**读完就能跑，不需要回问本机**。

任务清单在 [TODO.md](TODO.md)（唯一执行任务书），性能账目在
[PPO_UPDATE_ANATOMY.md](PPO_UPDATE_ANATOMY.md)，历史失败在 [DESIGN.md](DESIGN.md) §12。
本文件只讲**怎么在那台机器上干活**，不讲做什么实验。

---

## 1. 机器与环境

| 项 | 值 |
|---|---|
| SSH | `ssh -p 22222 newbiexvwu@127.0.0.1`（临时虚拟局域网，端口/地址可能变） |
| 登录 shell | Windows PowerShell 5.1 → `wsl -d Ubuntu -e bash` |
| WSL 内核 | 6.18.33.2-microsoft-standard-WSL2，x86_64 |
| GPU | RTX 5080，16303 MiB；驱动 617.14；`cap (12, 0)` = sm_120；`bf16_supported True` |
| Python 环境 | **`/home/newbiexvwu/.venvs/ml`**（python 3.14.4、torch 2.14.0+cu132、numpy 2.5.2） |
| 仓库 | `~/PvZ-Portable`（分支 `pvz-env`） |
| 模拟器资源 | `~/.cache/pvz-research-resources`（`main.pak` + `properties`），不入库 |
| 可用内存 | 16 GB（WSL 可见量按实测确定，见 TODO §7.2） |

**torch 只在那个 venv 里。** 直接 `python3` 会找不到 torch，必须用
`/home/newbiexvwu/.venvs/ml/bin/python`，或先 `source /home/newbiexvwu/.venvs/ml/bin/activate`。

仓库里 `scripts/win_ssh.py` 封装了整条链路（payload 走 base64，绕开三层引号重解析）：

```bash
export PVZ_DESKTOP_HOST=newbiexvwu@127.0.0.1
export PVZ_DESKTOP_PORT=22222
python3 scripts/win_ssh.py wsl --venv ~/.venvs/ml 'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
python3 scripts/win_ssh.py wsl --file probe.sh          # 长脚本写文件再发
python3 scripts/win_ssh.py ps  'Get-ChildItem C:\'      # 直接跑 PowerShell
```

### PowerShell 5.1 的三个坑

1. `$()` 会被 PowerShell 当变量展开 —— bash 脚本里的 `$(...)` 必须走 `--file`。
2. 控制台代码页是 GBK，非 ASCII 输出会变乱码（`win_ssh.py` 已强制 UTF-8）。
3. `&&` 不是语句分隔符；管道给 `wsl.exe` 会前置 UTF-8 BOM（`win_ssh.py` 已绕开）。

### 偶发故障

`System.AccessViolationException ... AmsiScanBuffer` —— PowerShell 的 AMSI 扫描偶发崩溃，整个
调用 1 秒就失败。**这不是脚本问题，直接重试即可**（第二次通常成功）。

---

## 2. 同步：全程走 git，不用 tar / scp

两边都是同一个仓库（`origin` = `NewbieXvwu/PvZ-Portable`），分支 `pvz-env`。
**不要再打 tar 包、不要再往 `/mnt/c` 拷文件**：手工拷贝会丢掉"这段结果对应哪个代码版本"
这件事，而它恰恰是证据的一部分。

```bash
# 台式机：开工前
cd ~/PvZ-Portable
git status                        # 先看有没有本地未提交改动
git pull --ff-only origin pvz-env

# 台式机：跑完后，把证据提交推送
git add artifacts/t5/perf artifacts/t5/curves artifacts/t5/throughput.json
git status                        # 确认没有把分片/检查点带进来
git commit -m "..."
git push origin pvz-env

# 本机：取回
git pull --ff-only origin pvz-env
```

### 什么入库、什么不入库（`artifacts/.gitignore`）

| 内容 | 入库？ | 理由 |
|---|---|---|
| `artifacts/t5/throughput.json` | ✅ | **训练入口直接读它**，不入库另一台机器拉完就跑不起来 |
| `artifacts/t5/stage0_gate.json`、`evaluation_parallel_equivalence*.json` | ✅ | 门禁与等价性证据，KB 级 |
| `artifacts/t5/perf/*.json` / `*.txt` | ✅ | 基准证据，KB 级（全部合计约 148 KB） |
| `artifacts/t5/perf/*.log` | ❌ | 滚动日志 MB 级，通宵跑会更大。留在台式机，报告里写摘要 |
| `artifacts/t5/curves/*.json` | ✅ | 学习曲线 |
| `runs/` 下的分片 `.npz`、`*.pt` 检查点 | ❌ | 太大（分片 272 MB/批），而且两边都会写，pull 必然冲突 |
| `training_state.json`、`learning_curve.json`（在 output-dir 里） | ❌ | 同上，运行产物 |

**学习曲线要复制一份到 `artifacts/t5/curves/` 才会被提交**（output-dir 里的那份不入库）。
检查点留在台式机上，需要时单独取。

### 第一次拉取会失败一次（这是预期的）

`artifacts/t5/throughput.json` 和 `artifacts/t5/perf/*.json` 原本是被 `.gitignore` 忽略的，
2026-09-30 改成入库。台式机上如果已经有同名文件（未跟踪），`git pull` 会拒绝：

```
error: The following untracked working tree files would be overwritten by merge:
    artifacts/t5/throughput.json
    artifacts/t5/perf/xxx.json
```

**本机那些 json 是从台式机拷回来的副本，台式机才是产生者**，所以以台式机为准：

```bash
mkdir -p ~/evidence-backup
cp -r artifacts/t5/perf artifacts/t5/throughput.json ~/evidence-backup/   # 先备份
# 按上面报错里列出的路径逐个删掉，然后重拉；不要 git clean -f 一锅端
git pull --ff-only origin pvz-env
# 拉完后比对：不一致时用备份里的（台式机实测）覆盖回去，并提交
diff ~/evidence-backup/throughput.json artifacts/t5/throughput.json
```

### 冲突怎么办

证据由**执行机（台式机）产生并提交**，本机只拉取。同名文件冲突时以执行机为准——它是
产生者。`git pull` 报冲突不要用 `--force`，先看 `git diff` 确认丢的是哪一边。

**未同步回本机的远端结果不能记作已完成**（TODO §1.2）。

---

## 3. 真实数据在哪

性能基准要用真实分片，不是合成数据。

* **台式机**：`~/PvZ-Portable/artifacts/t5/runs/run_2/.seed_jobs/update_0001/fcd5098ca3d43a7574e5f4dedb77b158d4ccc5caf666d2a74ae67d817df002d2/`
  （2000 个 `seed_*.npz`，`seed_0.npz` 41 transitions、tokens `(78,5)`）
* **本机没有这些分片**：`artifacts/t5/runs` 在 `artifacts/.gitignore` 里被排除。
  `artifacts/t5/throughput_shards_*` 全部为空目录，不要用它们跑基准。
* 分片有两种格式：新的是 `{"metadata":..., "result":...}`（`run_seed_jobs` 写的），
  旧的分片直接存 episode。**一律用 `pvz_seed_jobs.read_episode()` 读**，不要在调用点自己
  开码判断——`attention_benchmark` 就是这么悄悄失效的。

---

## 4. 常用命令

```bash
PY=/home/newbiexvwu/.venvs/ml/bin/python
cd ~/PvZ-Portable
SHARDS=artifacts/t5/runs/run_2/.seed_jobs/update_0001/fcd5098ca3d43a7574e5f4dedb77b158d4ccc5caf666d2a74ae67d817df002d2

# 回归：135 项 unittest（仓库没有 pytest）
cd python && $PY -m unittest discover -s . -p "test_*.py"

# 注意力层的前向/反向分解 + 融合 A/B + 逐项反向（PPO_UPDATE_ANATOMY.md §10.1–10.3）
$PY scripts/relation_bias_cuda_bench.py --data-dir $SHARDS \
    --episodes 16 --frames-per-episode 16 --repeats 20 --term-repeats 3 \
    --output artifacts/t5/perf/relation_bias_cuda.json

# 512 局规模的 PPO 更新，dense vs flex（§10.4 / §10.6）
$PY scripts/ppo_update_benchmark.py --data-dir $SHARDS --episodes 512 \
    --sequence-length 16 --minibatch-chunks 16 --learning-rate 1e-4 \
    --attention-backend dense --output artifacts/t5/perf/ppo_update_512_dense.json

# fp32/bf16 等价性论证（§10.5）
$PY scripts/precision_equivalence.py --data-dir $SHARDS --attention-backend dense \
    --output artifacts/t5/perf/precision_equivalence_dense.json
```

注意：**仓库根是 `~/PvZ-Portable`，但 unittest 要在 `python/` 子目录里跑**，脚本在仓库根跑。

---

## 5. 测量纪律（这几条都是踩出来的）

1. **多档比较必须每档各自预热。** `torch.compile` 只覆盖被预热的那一档，而且按 dtype
   各编译一次。只预热第一档会把后来的档位测慢：曾经把 2.0x 的胜利报成 2.6x 的失败，
   也曾把 bf16 报成比 fp32 慢 3.85x（真实是快 1.03–1.06x）。
2. **前向和反向分开计时。** 关系偏置 97% 的成本在反向（前向 0.47 ms、反向 17.78 ms），
   只看合计会把原因归错。
3. **复现旧记录时逐变体对。** 旧扫描 6 个变体里 5 个在 ±15% 内复现，只有 1 个差 8.8 倍——
   不逐个对就会以为"全都变了"。先看 JSON 顶层字段（`warmup`、`attention_backend`）判断
   那个数字是哪个版本的脚本产出的。
4. **CPU 上的开关收益不能外推 CUDA。** 关系偏置融合在 CPU 上是 1.30x，在 CUDA 上是 177x，
   差 125 倍。
5. **改完要交替重跑 A/B，不要各跑一次。** 单次 `auto` 测出 90.2 ms/步看着像少赚了，
   交替 A/B 各两轮才是 82.5/82.8（对 flex 的 126.8/131.1，1.53–1.59x）。90.2 是异常值。
   报结果时要带**路径指纹**（例如显存峰值与显式 dense 相同 = 2470.0 MB）证明走的确实是
   那条路径。
6. **反悔路径要留着。** 改 `auto` 的解析而不是删掉 `flex` 选项，就是为了能重测。

---

## 6. 当前待办（按优先级）

性能侧，都在生产机器上做（TODO §7.2）：

1. **确认 worker pool 的 5.5 s 启动截距。** 本机两点拟合：180 局 10.24 s、720 局 24.46 s →
   斜率 26.3 ms/局、截距 5.5 s。按 2,000 局/批算是 rollout 墙钟的 9.5%（记录机器上可能到
   31%）。不需要 CUDA、不涉及模型语义；确认后这是**目前最大的单项杠杆**。
   未确认前不要动手——复用的 worker 每批要重新 `load_state_dict`，还要处理崩溃与长跑内存增长。
2. **重做并行口径画像。** 本机 18 worker 的争抢放大是 4.88x，记录机器是 2.03x，差得比代码
   改动还多。所以本机的单核 3.7x **不要外推到生产的 18 worker**；只有 `--workers 1` 的
   口径可跨机器比较。

研究侧（TODO §9）：**T5-A 开始** —— 实验配置、完整恢复（模型/优化器/随机状态/累计更新数/
交互量/课程进度）、新配置评估闭环，然后 T5-E 的 3 初始化种子学习验证与预算 B。
当前没有 T5 学习达标证据，没有架构优胜结论，也没有完整关卡的熟练验收结果。

---

## 7. 已经做完、不要再重做的

* 关系偏置的 CUDA 实测与 `auto` → dense 的实施（§10）。结论与旧记录相反：dense 更快。
* fp32/bf16 的等价性论证（§10.5）。真实收益 1.03–1.06x，不值得开，正式训练保持 fp32。
* `attention_benchmark` / `ppo_update_benchmark` / `precision_equivalence` 的分片读取
  （统一走 `read_episode`）与每档预热。
* 9 处未使用 import 已清理；pyflakes 剩余 4 处是未使用局部变量与占位 f-string，需要理解
  意图后再动。

## 8. 提交纪律

* 开始远端任务前记录实际代码版本（`git rev-parse --short HEAD`）与待运行配置。
* 在台式机上提交时，commit message 说清改了什么、实测数字是什么、证据落在哪个文件。
* 证据走 git 提交推送（§2），**不要用 tar / scp / `/mnt/c` 拷**。没推送的远端结果不算完成。
* 提交前 `git status` 看一眼：分片（`.npz`）和检查点（`.pt`）不该出现在暂存区，
  出现了说明 `.gitignore` 被绕过，先查清楚。
* 旧文档里的固定提交号和吞吐命令只作历史记录，不要拿来对当前代码下结论。

---

## 9. 无人值守工作的硬约束

通宵跑之前先读这一节。**无人值守的失败模式不是"跑得慢"，是"跑了一夜什么也没留下"。**

1. **先冒烟再放大。** 用 30 分钟以内的小预算（`--rollout-episodes 200
   --max-episodes-per-run 1000`）跑通一次完整闭环，确认检查点、学习曲线、评估都出来了，
   再按冒烟实测的吞吐推算整夜能跑多少。**不要一上来就按 8 小时设预算。**
2. **长跑必须分段，段间用 `--init-checkpoint` 串联。** 完整恢复（T5-A）还没实现，
   进程崩了权重就没了。分成 4–6 段，每段结束会写一个检查点，崩了最多丢一段。
   段数不要超过 8——`MAX_FORMAL_RUNS = 8`，到顶就不让再跑了。
3. **每个实验用独立 `--output-dir`。** 两个原因：一是 `training_state.json` 里的
   `formal_runs` 是 per-directory 的，新目录从 1 开始；二是 **run_number > 1 会自动继承
   上一个 run 的权重**（`train_pvz_ppo_task_family.py` 第 811 行），同一个目录里跑第二个
   候选会静默继承，污染"从随机初始化开始"这件事。
4. **门禁拦住就停下报告，不要硬闯。** `--curriculum all`（20 任务）需要
   `artifacts/t5/stage0_gate.json` 存在且 `result == "pass"`，否则直接报错退出。
   这不是 bug：短任务课程用 `--curriculum cap1` 就不受它约束。真要用
   `--ignore-stage0-gate` 必须带 `--motivation` 写清理由，并在报告里原样记下来。
5. **日志落盘，不要管道。** `>> logs/xxx.log 2>&1`，每 30 分钟一行进度（当前局数、
   已用时间、最近一次 loss）。`... | tail -N` 会吞掉流式输出，卡住时完全看不到进度。
6. **盯内存。** WSL 只有 16 GB，18 worker 的进程树 RSS 峰值约 19 GB（有共享页重复计入，
   但 OOM 是真的）。跑之前先按 §7.2 实测的可用内存定 worker 数，不要用旧配置硬套。
7. **失败要留现场。** 崩了不要清理：日志、分片、`training_state.json`、最后那个检查点
   全部保留，把错误信息原样写进报告。"跑挂了但不知道为什么"比"没跑"更糟。
8. **不许为了跑通而放宽标准。** 不许改任务集、不许删失败种子、不许事后降低达标阈值、
   不许为了让门禁过而改评估器。预算不够就报告预算不够。
9. **收尾要把曲线复制出来并推送。** output-dir 里的 `learning_curve.json` 和
   `training_state.json` **不入库**（两边都会写，pull 必冲突），所以收尾时必须
   `cp <output-dir>/learning_curve.json artifacts/t5/curves/<实验名>.json`，
   连同 `artifacts/t5/perf/` 的证据一起 `git add && commit && push`（§2）。
   忘了这一步 = 一夜白跑——本机 pull 下来什么也看不到。
