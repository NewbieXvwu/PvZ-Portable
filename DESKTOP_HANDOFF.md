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
| 登录 shell | **`cmd.exe`**（不是 PowerShell）→ 需要时再 `wsl -d Ubuntu -e bash` |
| WSL 内核 | 6.18.33.2-microsoft-standard-WSL2，x86_64 |
| GPU | RTX 5080，16303 MiB；驱动 617.14；`cap (12, 0)` = sm_120；`bf16_supported True` |
| Python 环境 | **`/home/newbiexvwu/.venvs/ml`**（python 3.14.4、torch 2.14.0+cu132、numpy 2.5.2） |
| 仓库 | `~/PvZ-Portable`（分支 `pvz-env`） |
| 模拟器资源 | `~/.cache/pvz-research-resources`（`main.pak` + `properties`），不入库 |
| 可用内存 | 16 GB（WSL 可见量按实测确定，见 TODO §7.2） |

**torch 只在那个 venv 里。** 直接 `python3` 会找不到 torch，必须用
`/home/newbiexvwu/.venvs/ml/bin/python`，或先 `source /home/newbiexvwu/.venvs/ml/bin/activate`。

仓库里 `scripts/win_ssh.py` 封装了整条链路（脚本经 **stdin** 送进去，绕开 cmd.exe 的二次解析）：

```bash
export PVZ_DESKTOP_HOST=newbiexvwu@127.0.0.1
export PVZ_DESKTOP_PORT=22222
python3 scripts/win_ssh.py wsl --venv ~/.venvs/ml 'python -c "import torch; print(torch.__version__, torch.cuda.is_available())"'
python3 scripts/win_ssh.py wsl --file probe.sh          # 长脚本写文件再发
python3 scripts/win_ssh.py ps  'Get-ChildItem C:\'      # 直接跑 PowerShell
```

### 为什么是 stdin，不是 `-EncodedCommand`

**登录 shell 是 `cmd.exe`，不是 PowerShell。** 早期版本用
`powershell -EncodedCommand <base64>`，cmd 会重新解析这个参数字符串，base64 尾部的 `=`
填充和 `+/` 字符活不下来 —— 表现是**静默失败**：退出码 1，零输出，看起来像脚本自己出错。
现在改成 `powershell -NoProfile -Command -` / `bash -s`，脚本内容走 stdin，
argv 里只剩纯 ASCII 选项。改这段之前先读这一节。

### PowerShell 5.1 的三个坑

1. `$()` 会被 PowerShell 当变量展开 —— bash 脚本里的 `$(...)` 必须走 `--file`。
2. 控制台代码页是 GBK，非 ASCII 输出会变乱码（`win_ssh.py` 已强制 UTF-8）。
3. `&&` 不是语句分隔符；管道给 `wsl.exe` 会前置 UTF-8 BOM（`win_ssh.py` 已绕开）。

### 远端脚本的两个坑

1. **`set -euo pipefail` 会让整段脚本提前死。** 一个不匹配的 `grep`（退出码 1）就会
   终止后面所有命令，看起来像"远端什么都不干"。不确定的 `grep` 后面加 `|| true`。
2. **`~` 会在本机被展开。** `--venv ~/.venvs/ml` 这类路径必须加引号
   （`--venv '~/.venvs/ml'`），否则传过去的是本机 home 路径，WSL 里找不到。

### 偶发故障

`System.AccessViolationException ... AmsiScanBuffer` —— PowerShell 的 AMSI 扫描偶发崩溃，整个
调用 1 秒就失败。**这不是脚本问题，直接重试即可**（第二次通常成功）。

---

## 2. 同步：代码与结论走 git，大文件走 Hugging Face Hub

两边都是同一个仓库（`origin` = `NewbieXvwu/PvZ-Portable`），分支 `pvz-env`。
**不要再打 tar 包、不要再往 `/mnt/c` 拷文件**：手工拷贝会丢掉"这段结果对应哪个代码版本"
这件事，而它恰恰是证据的一部分。

但 **git 不是唯一的通道**。2026-10-01 的审计发现，把检查点提交进 git 会把仓库撑到 GB 级
（实测 `artifacts/` 被推了 2.0 GB，而规则认可的只有 9.4 MB）。分工如下：

| 通道 | 传什么 | 怎么传 |
|---|---|---|
| **git** | 代码、实验配置、门禁、结论层（KB 级 json）、文档 | 下面的 §2.1 |
| **HF Hub** | 检查点 `.pt`、评估分片 `.npz`、已归档证据 | §2.2 |

**不许用 `git add -f` 绕过 `artifacts/.gitignore`。** 细则见 [AGENTS.md](AGENTS.md) §1。

### 2.1 走 git 的部分

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

### 2.2 走 Hugging Face Hub 的部分

检查点 43 MB 一个，一个 run 几十个。**它们永远不进 git。** 走 HF Hub：
支持断点续传、SHA 校验、私有仓库，任何机器一行拉下来，不依赖临时 SSH 隧道。

一次性配置（两台机器各做一次）：

```bash
pip install 'huggingface_hub>=0.23'          # 台式机用 ~/.venvs/ml/bin/pip
export HF_TOKEN=hf_xxx                        # 见 §2.4 取 token
export PVZ_HF_REPO=<你的用户名>/pvz-agent-artifacts
```

日常用法：

```bash
python3 scripts/hf_sync.py ls                    # 远端有哪些 run
python3 scripts/hf_sync.py push <run>            # 上传最新检查点 + 结论层
python3 scripts/hf_sync.py push <run> --dry-run  # 先看会上传什么，不实际传
python3 scripts/hf_sync.py pull <run>            # 在任意机器下载
python3 scripts/hf_sync.py pull-all              # 全部拉下来
```

脚本同步两样东西：

* **检查点**（每个 run 约 286 MB）—— `evaluated` / `initial` / `boundary` / `resumed`
  全部，加**最新一个** `trained`，**再加 `resume.json` 指向的那个文件**。
  最后一项是硬要求：实测四个 `reward_r*_v2` 的 `resume.json` 都指向 `boundary`
  而不是最新的 `evaluated`，所以「只传最新 `evaluated`」的版本下载回来**接不上**。
  历史 `trained` 中间快照没有引用链指向，不传。
* **结论层**（KB 级）—— `learning_curve.json` / `training_state.json` /
  `experiment_config.json` / `provenance.json` / `resume.json` /
  `evaluations/*.json.gz`。

`push` 会打印每个检查点的体积和 `resume.json` 的解析结果；如果 `resume.json`
指向的文件不存在，它会明确警告「这份归档接不上」，不会假装成功。

仓库会在第一次 `push` 时**自动创建为私有**（`private=True`），不需要手动建。

### HF完整证据传输补充（2026-10-01）

本机 Python 固定 `/home/newbiexvwu/.venvs/ml/bin/python`。默认上传范围继续包含
resume.json指向的模型；原始NPZ、末段诊断窗口和日志需要显式包含：

```bash
/home/newbiexvwu/.venvs/ml/bin/python scripts/hf_sync.py push reward_r0_seed1_v2 \
  --include-rollouts --include-trained-window \
  --log logs/t5_research/reward_r0_seed1_v2.log --dry-run
```

去掉dry-run才上传，源run有正在持有的执行锁时会拒绝，不中断训练。先把完成的run上传，
当前活跃run等结束再传。原始工作目录的裁剪仍由既有8快照策略处理；HF默认仍只传最新
trained，include-trained-window额外携带现存最后8个供诊断，不执行任何删除。

整棵归档或真实中断现场（包括全部失败、日志、分片及中间状态）使用：

```bash
/home/newbiexvwu/.venvs/ml/bin/python scripts/hf_sync.py push-evidence \
  artifacts/research_evidence/reward_r2_seed0_v2/budget_complete \
  --name archives/reward_r2_seed0_v2/budget_complete --dry-run
```

已存在archive_manifest.json时逐文件核验引用链；缺失或损坏的resume指针会拒绝。
实际上传生成包含每文件SHA256/体积的MANIFEST.json（schema2），可用原pull/pull-all下载，
再对下载目录执行`hf_sync.py verify <下载的run或归档目录>`。这验证传输完整性，不替代
模型的真实恢复/学习门禁。18项离线回归及两个实际现场dry-run通过，见
artifacts/t5/perf/hf_full_evidence_offline_audit_v1.json。当前未配置HF凭据，未发生实际上传。

### 2.3 恢复已归档的证据

`scripts/archive_research_evidence.py` 把一份实验快照按原相对路径导出到
`artifacts/research_evidence/<实验>/<快照>/`，训练本身不受影响（不移动、不删除源证据）。
单文件 **≥ 100 MB**（`--single-file-limit-bytes`，默认 `100_000_000`）会切成
64 MiB 的 `.gitparts/` 分片，清单记录每片与整文件的字节数 / SHA256。

> 实测检查点只有 44 MB，**这个闸门从来没有触发过**。它是兜底，不是常规通道 ——
> 别把它当成"检查点可以进 git"的许可证。

**分片不要提交进 git**（这正是 2026-10-01 被删掉的 `CHECKPOINT_GIT_DELIVERY.md` 写错的地方）。
归档留在本地，或按 §2.2 走 HF。

```bash
# 导出快照
/home/newbiexvwu/.venvs/ml/bin/python scripts/archive_research_evidence.py \
  --experiment-dir artifacts/research/<candidate> \
  --archive-dir artifacts/research_evidence/<candidate>/<new-snapshot> \
  --log logs/t5_research/<candidate>.log

# 从快照恢复到新目录（含分片的情况）
/home/newbiexvwu/.venvs/ml/bin/python scripts/research_checkpoint_chunks.py restore-archive \
  --archive-dir artifacts/research_evidence/<candidate>/<snapshot> \
  --destination-dir artifacts/research/<candidate-restored-new-directory>
```

恢复会核对全部归档文件、片序、片 SHA 及整文件 SHA，并核对 `resume.json` 指针；
失败留下新目录和 `partial`，**不覆盖已有证据**。
验证记录：`artifacts/t5/perf/checkpoint_chunk_audit_v1.json`（真实 44.5 MB 检查点经
4 MiB 强制分片后完整复原，SHA 与原件一致）。

### 2.4 取 Hugging Face token

1. 注册 / 登录 <https://huggingface.co>。
2. 打开 <https://huggingface.co/settings/tokens> → **New token**。
3. 类型选 **Write**（要能建私有仓库和上传），名字随便填，例如 `pvz-agent-sync`。
4. 复制 `hf_...` —— **只显示这一次**，关掉就再也看不到，只能重建。
5. 两台机器各设一次环境变量：

```bash
# 本机 macOS：追加到 ~/.zshrc
echo 'export HF_TOKEN=hf_你的token' >> ~/.zshrc
echo 'export PVZ_HF_REPO=<你的用户名>/pvz-agent-artifacts' >> ~/.zshrc
source ~/.zshrc

# 台式机 WSL：追加到 ~/.bashrc
echo 'export HF_TOKEN=hf_你的token' >> ~/.bashrc
echo 'export PVZ_HF_REPO=<你的用户名>/pvz-agent-artifacts' >> ~/.bashrc
source ~/.bashrc
```

自检：`python3 scripts/hf_sync.py ls` —— 能打印出（空的）仓库名就说明 token 通了。

> token 等于账号写权限。**不要提交进 git**，不要贴进任何会入库的文件。

---

## 3. 真实数据在哪

性能基准要用真实分片，不是合成数据。

* **台式机**：`~/PvZ-Portable/artifacts/t5/runs/run_2/.seed_jobs/update_0001/fcd5098ca3d43a7574e5f4dedb77b158d4ccc5caf666d2a74ae67d817df002d2/`
  （2000 个 `seed_*.npz`，`seed_0.npz` 41 transitions、tokens `(78,5)`）
* **本机没有这些分片**：`artifacts/t5/runs` 与 `artifacts/t5/throughput_shards_*`
  都在 `artifacts/.gitignore` 里被排除，本机（2026-10-01 清理后）连目录都不存在。
  台式机上它们**有内容**（`throughput_shards_20260929T074802Z` 有 2011 个文件、
  `night_measurement_shards` 4900 个），但都是 2026-09-29 那批吞吐基准的旧输入，
  与 §4 里指向的 `artifacts/t5/runs/run_2/.seed_jobs/...` 不是同一批 ——
  **跑基准只用 §4 给的那个路径**。
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

2026-09-30 持续 T5–T7 目标已授权必要代码修改。新的显式配置研究入口与恢复协议见
[RESEARCH_EXECUTION.md](RESEARCH_EXECUTION.md)；下面的旧 `--init-checkpoint`、八次 run
及 stage0 细节仍描述 legacy 入口。新入口使用 `--experiment-config` 和完整 `--resume`，
正式实验需先取得新闭环通过证据。冒烟、日志、内存、失败现场、冻结标准及 git 同步纪律继续适用。

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
