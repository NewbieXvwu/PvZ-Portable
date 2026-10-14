# 通宵任务 v5：**教师诊断上线 + 把干预实验做对**

**时间不设上限。唯一原则：结论要能经受"统计上显不显著"的追问。**

---

## 0. 前情

### 0.1 v4 已完成（不要重做）

- **A0 前置校验通过**：`whatif` 前缀能逐位复现（seed_30496，393/393 帧一致）；`train_full_level7_v1_d1.preplanted=[]`。
- **A3（收紧判据的"有钱却不种"）**：输局错失率 **41.5%** vs 赢局 **26.1%**（**+15.4pp，p=0.00006 显著**）。
- **A1 干预实验（15 个配对失败局，v3_1m，reason=僵尸进屋，final_wave≤20）**：

| arm | 通关 | 终局波均值 | vs I0（Holm 逐步下降）|
|---|---|---|---|
| **I0 原策略** | **0/15** | 13.67 | — |
| I1 首破行 c4/c5 种 1 豌豆 | 1/15 | 14.60 | 1.000 不显著 |
| I2 连种 10 步豌豆 | 6/15 | 19.87 | 0.0625 不显著 |
| **I3 连种 10 步向日葵** | **8/15** | 21.87 | **0.0234 显著** |

**→ "补火力能救局"没通过；"补向日葵救不了"被推翻。I2 vs I3 不显著（p=0.69）。**

### 0.2 ★ v4 留下的**致命缺口**：干预剂量太低

- **I1 只有 2/15 局真的种下豌豆**（10 局当时阳光 <100、3 局卡冷却）。
- **I2 的 150 个强制决策只有 9 次真种下**；**I3 只有 14/150**。
- 其余全部因为"没有合法落点"退化成等待。

**→ 三个后果**：
1. 结果只是"**意向处理**"效应，不是"补种策略"的效应。**必须重做，保证剂量。**
2. **I3 的"显著"极可能是"扰动轨迹"的伪影**——把轨迹岔开就有一定概率活更久，**跟种什么无关**。
3. **"第一次被突破时它常常没钱"本身是重要事实** → 那个干预点**太晚**。

**→ 本次的核心就是补上这两件事：① 加扰动对照；② 把干预点提前 + 保证剂量。**

### 0.3 现场事实（本机 Agent 已测，直接采信）

- **348 局逐步回放已分析完**（结论在 `MEMORY/docs/failure-mode-analysis.md`，脚本 `/tmp/an*.py`）。
- 前 5 波赢/输完全一样；分水岭第 5~8 波；**被吃植物里向日葵占比：赢 15~18% vs 输 42~47%**；
  **被吃位置：赢局前排 c4-c5（46~50%），输局最后排 c0-c1（43~49%）= 向日葵区**；
  同一波输局少 0.5~1.1 火力/行；阳光余额（w20）赢 950 vs 输 325；输局从不升级（寒冰/双发 1.5~3% vs 赢 11~14%）。
- **★ 教师基础设施已经在台式机上装好并验证通过**（见 §2）。**这是本次最大的新能力。**
- **WSL keeper 已修好**：`PvZWSLKeeper` 现在每 5 分钟跑 `~/.pvz-keeper.sh`，
  它**同时**保证 ① CLIProxyAPI 在跑、② WSL 不被空闲关机。**别动它。**

---

## 1. 环境

```bash
source ~/.pvz-teacher-env.sh     # 教师流水线全部环境变量（dsh/DSH_HOME/GEMINI_API_KEY/PVZ_*）
```
该文件内容（已就位）：
```
PATH += ~/.local/opt/node-v24.21.0-linux-x64/bin, ~/.local/bin
DSH_HOME=~/.local/share/pvz-agent/dsh-home
GEMINI_API_KEY                       （值放在环境变量里，不写入仓库；CLIProxyAPI 的 key，不是 Google 官方 key）
PVZ_ROOT=~/PvZ-Portable
PVZ_PYTHON=~/.venvs/ml/bin/python3      （有 torch 2.14.0+cu132）
PVZ_RESOURCE_DIR=~/.cache/pvz-research-resources
```

- **非交互式 shell 读不到 `.bashrc`**：取 HF 凭据用
  `eval "$(grep -E '^export (HF_TOKEN|PVZ_HF_REPO)=' ~/.bashrc)"`
- **python 用 `~/.venvs/ml/bin/python3`**（系统 python3 没 torch/numpy）。
- 命令可能带 `set -euo pipefail`，**无匹配的 grep 一律加 `|| true`**。
- **⚠️ GPU 独占**：起 CUDA 前 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`；`pkill` 后等 5 秒。
- **⚠️ 并行规则**：纯读 JSON 的分析可并行；**任何跑模拟器/模型的东西（训练/评估/capture/whatif/教师会话）必须串行**。
- **改 PPO 参数只能改配置 JSON**；`--workers`/`--rollout-threads`/`--rollout-device` **不许传**。
- 先查磁盘 `df -h ~`；可用 < 100 GB 就停。

---

## 2. 教师流水线（**已装好，直接用**）

### 2.1 起一次真实会话（照抄）

```bash
source ~/.pvz-teacher-env.sh
S=/tmp/pvz-v5/<会话名>; mkdir -p "$S/.dsh/skills" "$S/run"
# 先验 skill 按需带（见 §2.3）
cd "$S"
nohup dsh --profile pvz-teacher "$(cat prompt.md)" > run/session.log 2>&1 &
```
- **cwd 决定会话 id**：`$DSH_HOME/sessions/--private-tmp-pvz-v5-<名>--`。
- **长任务书放 `prompt.md`**，用 `"$(cat prompt.md)"` 传；也可传 `-` 从 stdin 读。
- **续跑**：`dsh --profile pvz-teacher --session-id session-xxxx "继续"`。
- **工具调用轨迹**：`tools/trace_session.py` 从会话记录还原（调了什么、参数、结果、token）。

### 2.2 起之前自检三件事（**每次都要**）
```bash
source ~/.pvz-teacher-env.sh
which dsh                                                    # ① 应在 ~/.local/opt/node-v24.21.0-linux-x64/bin
python3 -c "import json;print(json.load(open('$DSH_HOME/profiles/pvz-teacher/package.json'))['dsh']['profile']['bundles'])"
   # ② 必须是 ['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless','@local/pvz-teacher']
   #    ★ 少 dsh-headless 会启 web 应用并永久卡住（已踩过）
$PVZ_PYTHON -c "import torch;print(torch.__version__)"        # ③ 必须打得出来
curl -sS -o /dev/null -w '%{http_code}\n' -H "x-goog-api-key: $GEMINI_API_KEY" http://localhost:8317/v1beta/models   # 必须 200
```
**网关挂了就跑一次 `~/.pvz-keeper.sh`（它会自动拉起），别自己去 kill。**

### 2.3 已实测通过的基线
- `dsh --profile pvz-teacher "调用一下 mcp__pvz__ping，把返回的 JSON 原样告诉我"` → **14.5 秒**、
  返回 13 个工具、`rl_replay: true`。
- `verify_s1.py`（capture → index → whatif，真跑模拟器）→ **全部通过**，
  whatif 12 个候选、约 1~3 秒/个、找到能通关的替代方案。
- **成本**（README 实测）：一次完整诊断会话 **9 分钟 ~ 2.5 小时**、20~40 轮工具调用。
  **所以只挑少量局，且必须把已有发现当"假设"交给它验证。**

### 2.4 会话任务书模板（`prompts/diagnose_and_teach.md`）
**教师自己的任务模板在仓库里，读它**：`$PVZ_ROOT/dsh/pvz-teacher/prompts/diagnose_and_teach.md`。
**★ 但我们这次要额外加一段"已知假设"**（否则它会从零猜，浪费轮次）：

> 已知（由本机逐步回放 348 局得出，三个检查点全部复现）：
> - 这局属于"早死/中死"。同一波时它的火力比赢局少 0.5~1.1 个/行。
> - 被吃掉的植物里**向日葵占 42~47%**（赢局只有 15~18%）；植物死在**最后排 c0-c1**（赢局死在前排 c4-c5）。
> - 它在"至少两行火力 <2"时**有 41.5% 的机会没补火力**（赢局 26.1%，p=0.00006）。
> - 干预实验显示：**在第一次被突破那一刻强制改种，I2（火力）校正后不显著、I3（向日葵）显著**，
>   但**剂量极低**（I2 只有 9/150 次真的种下去）。
> **请判断：它是在哪个决策点上开始走错的？是"该种火力却种了向日葵/等待"，还是"钱不够被迫"？
> 两者如何区分？** 用 `frame` / `lane` / `actions` / `whatif` 给证据。

---

## 3. 轨道 A（★ 最高优先级）：**把干预实验做对**

### A1 ★ 补"扰动对照" I4 —— 回答"I3 的显著是真效应还是伪影"
**加一个 arm：I4 = 在同样的干预点，强制一个"随机合法动作"**（随机选一个当时合法可执行的
`plant` 或 `wait`，与 I2/I3 的时机完全相同）。
- **若 I4 也救回 ~8/15 → I3 的显著是轨迹扰动伪影，不是"向日葵有用"。**
- **若 I4 明显低于 I3 → I3 可能有真效应（但要说清机制）。**
**统计口径与 I0/I2/I3 完全一致**（精确配对 McNemar + Holm）。

### A2 ★ 重做 I2/I3：**保证剂量**
上次失败的原因是"目标落点不合法就退化成等待"。这次：
- 干预前先枚举**当时所有合法落点**（阳光够、卡可用、格子空、行内射手 <4），
  **从合法集合里挑**（I2 挑火力最薄的行、I3 挑最需要补经济的行）。
- **记录每个 arm 的实际送达率**，**送达率 <50% 的 arm 必须在报告里标注"意向处理，剂量不足"**。
- 如果某局在干预点**根本没有合法落点**（钱不够），**单独记一类 `I-noop`**，
  并把"被突破时有多少比例局是没钱状态"作为**独立结论**报出来（这本身是重要发现）。

### A3 ★ 提前干预（上次因为"关键点干预能救回"而没做，但现在理由变了）
上次触发条件不成立是因为 I3 显著。**但 I3 的剂量只有 14/150，且可能是伪影。**
**所以 A3 照做**：干预点改到**第 5 波开始时**（分水岭之前），内容：
- **I5**：把火力最薄的 2 行各补到 2 个射手（同样要挑合法落点、记送达率）。
- **I6**：同 I5 的时机，但补向日葵（反向对照）。
**判据**：I5 显著优于 I0 → "提前加固能救"；I5 与 I6 无差别 → 问题不在"补什么"，在别处。

### A4 便宜的补充（纯读数据）
- **A3 的"末波花费"上次没匹配到那局**（我引用的 575→250→175 不在 348 局里）。
  **别再找那一局**；改成**按"末波死"这一簇整体统计**后段花费构成（每局的最后 3 波花了什么）。
- **"有钱却不种"的时间分布**：把错失按**波数**分箱（w5-10 / w11-15 / w16-20 / w21+），
  看错失集中在哪一段。

---

## 4. 轨道 B：**教师诊断（新增能力，只挑 8~10 局）**

**挑局**（都用 `--task-manifest experiments/t7/bc_finetune_level7_v1/train.json --task-id train_full_level7_v1_d1`）：
- A1 里**I0 复现了原结局**的局中，挑 **3 局早死/中死**
- **2 局末波死**
- **2 局稳赢**（对照）
- **A1/A2 里"干预救回来了"的 2 局** ← 最重要，看它当时**本该做什么**
- **1 局 B9（截断 32）崩掉的**

**每局一个独立会话**（会话很慢，别串在一个会话里）。
**产出按"它错在哪"分类归纳，不要逐局流水账。**
**明确要求它：每条判断必须过"三问"闸门（反例问 / 证据问 / 迁移问），过不了就写"本局无新增 skill"。**

---

## 5. 轨道 C：**补跑训练实验**（GPU 串行，优先级低于 A/B）

| 序 | run | 唯一变量 | 预算 | 为什么 |
|---|---|---|---|---|
| C1 | `bc_lr10_v1` | `learning_rate` 1e-4 → 1e-5 | 300k | "更保守"方向，从没测到 |
| C2 | `bc_seed1_v1` | 只换 `initialization_seed`+`sampling_seed` | 100k | **噪声地板** |
| C3 | `bc_gae95_v1` | `gae_lambda` 1.0 → 0.95 | 100k | 从没测到 |
| C4 | `bc_mb8_v1` | `minibatch_chunks` 2 → 8 | 100k | B2 的 OOM 缩小版 |

**评估节点**：300k 的设 `[100000,200000,300000]`；100k 的设 `[50000,100000]`。
**基线 = v3 的对应节点**；**不同预算的对比要注明预算不等**。
**任何"变化小于噪声"→ 必须写"不显著"。**

---

## 6. 健壮性

1. **每 30 分钟把结论层推到 HF**（不要等收工）：
   ```bash
   eval "$(grep -E '^export (HF_TOKEN|PVZ_HF_REPO)=' ~/.bashrc)"
   cd ~/PvZ-Portable && ~/.venvs/ml/bin/python3 scripts/hf_sync.py ls
   # 推：codex-report-v5.md、各 run 的 learning_curve.json / evaluations/*.json.gz、教师会话日志
   # 不推：.pt、episode_samples
   ```
2. **每完成一个阶段就往 `~/codex-report-v5.md` 追加**（不要憋到最后写）。
3. 严格串行部分写进队列驱动器 `/tmp/n5_queue.sh`，每步**先确认 GPU 空闲**。
4. 每实验独立日志 `/tmp/n5_<run>.log`；进度 `/tmp/n5_progress.txt`。
5. **不许用 `tmux kill-server`**，只用 `tmux kill-session -t <名>`。
   **⚠️ 特别注意：`tmux kill-server` 会杀掉 keeper 之外的会话；而 `cpa`（网关）已不依赖 tmux，放心。**
6. 超时：教师会话 **3 小时**、干预/capture **3 小时**、100k 训练 **5 小时**、300k 训练 **8 小时**；
   超时记 `timeout`、杀掉、继续。

---

## 7. 交付

写进 **`~/codex-report-v5.md`**：

- **§A1 扰动对照 I4**：I0/I1/I2/I3/I4 的对照表 + **"I3 的显著是真效应还是伪影"的判定**
- **§A2 保证剂量的 I2/I3 重做**：各 arm 送达率 + 结果 + **"被突破时有多少局没钱"的独立结论**
- **§A3 提前干预 I5/I6**：结果 + 判定
- **§A4** 末波死后段花费构成 + 错失的时间分布
- **§B 教师诊断**：按"它错在哪"分类归纳 + 产出的 skill（过闸门的）
- **§C 训练实验**：每个跑了没、结果、显不显著
- **§D 一句话总结**：**根因是什么（按证据强度排）+ 下一步（按优先级）**

**每个数字都要能复现**（哪个文件哪个字段）。
**每个 git 提交号都要能被 `git cat-file -t` 验证**（`c4de1ee` 那次不存在）。
**不许把"不显著"写成"更好/更差"。**
**不许把"没跑成"（基础设施故障）当成"假设被否"。**
**报告里的 Holm 校正数字必须和同一份 JSON 里的数字一致**（v4 报告正文与 JSON 对不上过一次）。

---

## 8. 明确不要做的

- **不要重做**失败模式分析（§0.3）、A0 校验、A1 原始实验、A3 的"有钱却不种"（已完成）
- **不要调参调参调参**（用户原话："我们别再各种微调了"）
- **不要动 `~/.pvz-keeper.sh` 和 `PvZWSLKeeper` 计划任务**（它们同时保活 WSL 和网关）
- 不要碰 `artifacts/adventure2_level7/training_summary.json`（封存证据）
- 不要删 `artifacts/` 里的 `.pt`（`archive_manifest.json` 记了 SHA256）
- 不要 `rm -rf`；删文件用 `python3 -c 'import os;os.unlink(...)'`
- **不要 push `.pt` 和 `episode_samples` 到 HF**（只推 KB~MB 级的结论层）
- **不要改 `python/` 和 `scripts/` 下的源码**；确实需要改：**先备份、写清改了什么和怎么回滚、报告里单列一节**
  （**`dsh/pvz-teacher/` 下的工具/提示词可以改**，改完要同步部署副本：
  `$DSH_HOME/profiles/pvz-teacher/node_modules/@local/pvz-teacher/`）
- 不要动 `~/PvZAgent-bc-handoff/`
