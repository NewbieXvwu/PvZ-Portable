# PvZAgent 项目长期笔记（curated）

> 整理于 2026-10-14（去平台化 + 蒸馏）。**只留"跨会话仍有效"的结论与硬规则**；
> 数字与推导一律在各日 `YYYY-MM-DD.md` 或 `MEMORY/docs/`，本文只给结论与出处。
>
> **平台说明**：本文件与同目录每日日志位于仓库 `MEMORY/`（随 git 同步），任何 Agent 平台都可直接读写；
> WorkBuddy 侧另有软链 `.workbuddy-ai/memory → ../MEMORY` 兼容旧路径。文中的 `~/.workbuddy-ai/...` 是
> **WorkBuddy 托管的用户级路径**（托管解释器/venv、用户级配置），**其它平台没有这些目录**，照抄无效。

## 运行环境 / 测试陷阱

- **必须用 mise 的 Python**：`/Users/newbiexvwu/.local/share/mise/installs/python/3.14/bin/python3`（有 numpy + torch）。
  WorkBuddy 托管的 `~/.workbuddy-ai/binaries/python/...` 下**没有** numpy/torch。
- 测试：`cd python && <mise python> -m unittest discover -s . -p "test_*.py"`（**无 pytest**）。
- 真实模拟器资源：`/Users/newbiexvwu/Downloads/Plants_Vs_Zombies_V1.2.0.1073_EN`。
- `test_env_protocol.py` 的 `_resource_dir()` 是**假资源目录**（1 字节 main.pak），拿它调真实 `env.reset()`
  → 子进程卡死、父进程永久等。
- `multiprocessing.Pool` initializer 抛异常时 `imap_unordered` **不报错**，父进程永久挂起；
  `run_seed_jobs` 的 `stall_timeout`（默认 900 s）把它变成 `RuntimeError`。
- 长跑探针**不要** `| tail -N`（丢流式输出）→ 写日志文件。

## 文档结构（唯一任务书 = `TODO.md`）

核心 7 份：`README.md` / `TODO.md` / `DESKTOP_HANDOFF.md` / `DESIGN.md` / `PPO_UPDATE_ANATOMY.md` /
`OPTIMIZATION_TASKS.md` / `PROTOCOL_OBSERVABILITY_AUDIT.md`。**新增根目录文档前先想能不能并入这 7 份之一**
（历史反弹到 21 份）。删文档后必做：① 改掉 `python/`、`scripts/` 里的引用（`train_pvz_ppo_task_family.py`
的 `protected_paths` 会 SHA 校验 `DESIGN.md`/`TODO.md`，缺失直接报错）；② 扫断链。

## 仓库约定（体积/交付/检查点规则见 `AGENTS.md` §1–§4，这里只记它没写的）

- 封存证据不可改：`artifacts/adventure2_level7/training_summary.json` 的 `episode_hash` 只能保留、
  不能改语义；新算法走新函数名 + provenance 记算法名。
- 两机同步两条通道（分支 `pvz-env`，origin `NewbieXvwu/PvZ-Portable`，**不用 tar/scp/mnt/c**）：
  git 走代码/配置/门禁/结论层，`.pt`/`.npz` 走 Hugging Face Hub（`scripts/hf_sync.py`，仓库
  `realnewbiexvwu/pvz-agent-artifacts`），**永远不进 git**。**HF 仓库文件数上限 20,000**（10-09 曾到 19,985）。
- `artifacts/` 默认全忽略，白名单只有 `t5/throughput.json`（**训练入口直接读它**）、`t5/stage0_gate.json`、
  `t5/perf/*`、`t5/curves/*`；output-dir 的 `learning_curve.json`/`training_state.json` **不入库**。

## 台式机（WSL）连接与运维

连接与日常操作见 `DESKTOP_HANDOFF.md`。速记：`export PVZ_DESKTOP_HOST=newbiexvwu@127.0.0.1
PVZ_DESKTOP_PORT=22222`，用 `scripts/win_ssh.py ps|wsl`；或 `ssh -p 22222 ... "wsl -d Ubuntu -e bash -s" < 脚本`
（**取大文件用 base64 走 stdin**）。WSL `Ubuntu`，venv `~/.venvs/ml`（torch 2.14.0+cu132，RTX 5080 16 GB，
`nproc=20`）；**venv 参数要加引号**。4 个 worktree（`~/PvZ-Portable` 主 + `-curriculum`/`-observation`/
`-reset-disposal`），**队列从主 worktree 跑**。远端 wrapper 带 `set -euo pipefail` → **无匹配的 grep 一律加
`|| true`**，否则吞掉后面所有输出。

**★ 踩过的坑（这些不在 `DESKTOP_HANDOFF.md` 里）：**

- **WSL 空闲就会关机，杀掉 tmux 里的长跑**（PID 1 换新 = 重启过；tmux server 一起死，**不留 EXIT 行**）
  → **没有 keeper 的长跑一定活不过一个 SSH 空档**。`PvZWSLKeeper`（Windows 计划任务，每 5 分钟）动作 =
  `wsl.exe -d Ubuntu -e /home/newbiexvwu/.pvz-keeper.sh`。**别动 `~/.pvz-keeper.sh`、`~/.pvz-chain.sh`、`PvZWSLKeeper`。**
- **`~/.pvz-chain.sh` 的三个关键设计**：① 用「进程存活」而非「日志新鲜度」判断 codex 死活（否则**误杀健康的运行**）；
  ② **「忙守卫」**：`n5_queue.sh` 在跑 / 有 `train_pvz_ppo` 进程 / GPU 上有进程 → 推迟接续，否则两个训练抢 GPU →
  `CUDA error: busy`；③ case A 要检查 `NEXT` 存在。改完必做 `bash -n` + 沙箱 8 分支测试。
- **GPU 独占**：起 CUDA 前查 `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`；**`pkill` 后等 5 秒**。
- **隧道 = UURemote（UU远程）的端口转发，不是 ssh 隧道**。**Mac 侧代理抖动会让它拆掉端口映射且不自动重建**
  → 所有远端操作**重试 ≥4 次（间隔 6~8 秒）**。**用户指示：不要再开定时任务**（会跑空）。
- **PowerShell 经 SSH 调用**：多层引号必炸 → `powershell.exe -NoProfile -EncodedCommand <base64(UTF-16LE)>`；
  **反引号会被远端 bash 吃掉**；**`Register-ScheduledTask` 曾静默失败，`schtasks` 才可靠**；
  **`AllowStartIfOnBatteries` 不是真实属性名**（真的是 `DisallowStartIfOnBatteries`）。

## 分工与 codex

- **训练队列由另一个 Agent 负责**；本 Agent：仓库体积治理、制品交付通道、跨机同步、文档与约束。
- codex 在台式机跑：任务书 `~/codex-task.md`、日志 `~/codex-run.log`、报告 `~/codex-report*.md`。启动：
  `tmux new-session -d -s codex "cd ~/PvZ-Portable && cat ~/codex-task.md | codex exec -C ~/PvZ-Portable --dangerously-bypass-approvals-and-sandbox - > ~/codex-run.log 2>&1; echo EXIT=\$? >> ~/codex-run.log"`。
- **报告数字可信度高**（多次独立复核吻合），但**会报不存在的 git 提交号** → 任务书必须要求**"每个提交号都能被
  `git cat-file -t` 验证"**；**Holm 数字曾与同一份 JSON 对不上**（v4）→ 也写进任务书。
- **它会 commit 但不 push** → 收工时替它推；**它的护栏拒 `rm -f`** → 用 `python3 -c 'os.unlink(...)'`。
- **接管别人的活之前先查它为什么这么做**（10-08 教训）；**任务书必须自包含**（接续轮是全新上下文）+
  内嵌决策规则（不让它停下来问）；**每 30 分钟推结论层到 HF** 这条规则救过场。

## 回放 RL 检查点的两条硬规则

1. **卡组只能来自训练任务清单，不能用 `deck_for_level()`。** 训练是 `train_pvz_ppo.py:64` =
   `env.reset(deck=task["deck"], ...)`，清单在 `experiments/t7/*/train.json`。卡槽错位时**这一局会照常跑完
   并给出结果**（静默）。`policy=ppo` 不带 deck 现在直接报错；用 `capture --task-manifest <清单> --task-id <任务>`
   一次拿全 level/deck/cap/mult/**预种植物**。
2. **`provenance.fingerprints.simulator` 是训练时的 `build/pvz-portable`**（`evaluate_full_acceptance.py` 拿它做
   硬门禁）。本机 `2056e13e` / 主线 `dc8747f4` / 老 v4 `371a75d7` 都不同。**跨平台编译同源码字节就不同 →
   "不一致"≠行为变了**；`python/*.py`+`scripts/*.py` 的**源码指纹**才是语义判据。

## 任务名陷阱：`heldout_*` ≠ `train_*`

`heldout_roof_1` = 第 45 关 / cap3 / deck `[0,1,3,4,7,33]`；`train_roof_1` = 第 42 关 / cap1 / deck
`[0,1,2,3,4,33]`。早先 `lookup_task()` 剥掉 `heldout_` 前缀回退到 `train_*`，静默拼错。**已修**：先收
`experiments/t7/*/progress_evaluation.json`（精确键），**只允许 `validation_X → train_X` 一种别名**，
`heldout_X` 一律不回退。**推广规则：任何"按名字回退匹配"的查表都必须显式白名单别名。**
另：**屋顶卡组含花盆（33）**，草地 vs 屋顶不是同一动作空间。

## 奖励与训练机制（结论；推导见 `2026-10-04.md`）

- 奖励 = 势能塑形 + 终局 ±1；`Φ` = (min(1,阳光/1000) + min(1,波数/总波数) + 植物**平均**血量)/3，
  `Φ(terminal)=0`，`γ^(本步tick/300)`。**终局打折已于 10-09 移除**（`3936add`），化简式 `G_t = −Φ(s_t) + Y`；
  `VALUE_SEMANTICS` 改名 `undiscounted_terminal_v1`（价值目标变了，**必须换语义版本号**，否则新旧检查点静默混淆）。
  打折写法会让唯一梯度只剩"缩短局长" → **v2 从 80.47% 崩到 0.78%**；v3 修好后 1M 到 **92.58%**。
- **塑形项不提供学习方向**：望远镜相消为 `−Φ(s_t)`，期望梯度为零（Ng 1999）→ **"reward hacking（攒阳光）"
  不成立**（动作空间里**没有"收阳光"**）。**λ=1 是防御性选择**（当时 critic 不可靠），不是"λ=1 更好"。
- **成本结构**：Σrollout 21% vs **ΣPPO 79%**；PPO ≈ **64 ms/transition** vs 基准 **0.75 ms/transition** → 差 85 倍。
- **PPO 参数只能改配置 JSON**：CLI 的 `--ppo-epochs`/`--minibatch-chunks`/`--learning-rate` 对研究配置**完全无效**；
  `--workers`/`--rollout-threads`/`--rollout-device` **不许传**（会 raise）。**续跑身份**（`3c5117b`）：
  `canonical_digest` **忽略 `workers`/`worker_threads`，但 `max_actions`/`worker_device` 仍纳入**。
- **已知限制**：`minibatch_chunks=8`（及 16）**在 16 GB 显存上 OOM**；`--rollout-episodes` 的标定值是 2000
  （注释写明），而所有实验配置用 16；`bc_gae95_v1` 遇到过 CUDA/Dynamo worker 错误，**重试即可**。

## ★★ 主问题：训练会"熵爆炸"崩溃，而**是否崩溃由随机种子决定**

（10-13/10-14 查实；完整数据见 `2026-10-14.md`、`docs/codex-report-v6.md`。）

- **同一配方、同一起点、只换随机种子**：v3（seed 0）1M → **92.58%**；`bc_seed1_v1`（seed 1）100k →
  **50.78%（贪心）/ 39.06%（采样）**。**→ 种子方差（≥34pp）远大于 256 局的评估噪声（±4.07pp）。
  单种子结论一律不可靠。**
- **崩溃不是"不玩了"**：C2 峰值火力与 v3 相同、花阳光更多，但**被吃 2.4 倍、割草机 2 倍** → **"照打但防不住"**。
- **★★ 熵（`update_history[*].losses.entropy`）末值是"会不会崩"的强预测器**：100k 末次熵 **< 0.012 的全部是
  基准/不显著；> 0.015 的全部显著变差**（样本内 12/12，后扩到 15/15）。基准 v3 稳定 0.0047→0.0065；
  C2 涨到 **0.113**、criticwarm **0.333**。**⚠️ 阈值在同一批历史上选的，不是独立验证，不能外推成崩溃概率。**
- **★★ 因果已由干预钉死（10-14 v6）**：给已知会崩的 seed1 加"**熵守卫**"（`optimizer.step()` 后在**同一
  minibatch** 重算熵，超 `ppo.entropy_guard.max_entropy` 就**恢复本次 train_update 开始时的模型参数 +
  AdamW 状态**并停止该次更新）。三档阈值全部极显著救回：无守卫 50.78%/39.06% → **0.01**：84.38%/85.16%；
  **0.02（推荐）**：84.38%/84.38%；**0.05**：86.33%/81.25%；**vs 无守卫 +33~+46pp（Holm p ≤ 1e-15）**，
  三档两两不显著（p≥0.729）。**→ 熵爆炸是崩溃的「因」，不只是「果」。** 守卫对好种子**无副作用**
  （`seed0+守卫0.02` vs v3 @100k：贪心 −0.78pp p=0.856、采样 +3.52pp p=0.422）。
- **⚠️ 反例**：v3 的熵**最大**也到过 0.0851（单次尖峰）却没事 → **看末值/持续性，不看单次尖峰**。
  **⚠️ 时序钉不死因果**（C2/B10 熵先于成绩下滑，criticwarm/B5 成绩先降，B8 同节点后恢复）→ **只有干预能定因果**。
- **★ 这改写了整个项目的解读**：**6 个"变差"的 run 熵末值全部 > 0.015**（seq32 / val2 / gae95 /
  criticwarm / seed1 / 旧 v2）→ **它们可能是同一个失败模式，不是各自参数的问题**；**必须给它们都加守卫重跑**，
  才能分开"参数效应"与"崩溃效应"（v7 轨道 B）。无守卫多种子 100k：seed0 84.77%、seed1 50.78%、seed2 63.67%
  → **崩溃率 ≥ 2/3**。
- **★ 假设 H2（正反馈环）**：一批 16 局大多输 → 优势几乎全负、方向单一 → 梯度把策略推开 → 熵升高 →
  表现变差 → 下一批输得更多。**好种子能被"KL 刹车 + 后续批次转好"打断，坏种子锁死。**
  **`won=0` 是点火源但不是充分条件**；**KL 刹车挡不住熵爆炸**（它管"改了多少"，不管"改歪了没有"）。
- **★★ 代价**：守卫改了 `python/train_pvz_ppo.py` 与 `python/pvz_research.py`，而
  `experiment_identity_digest()` 纳入源文件指纹 → **源码 hash 变了 → 所有旧 run 用当前源码都不能续跑**。

## 失败模式：落点错配

（10-12 起，348 局逐步回放 + 教师独立印证；完整分析见 `docs/failure-mode-analysis.md`、
`docs/lane-mismatch-analysis.md`、`2026-10-13.md`。）

- **判据（复用仓库自己的函数）**：`scripts/render_episode.py::_reachable_dps(frame, row, front_x)` 返回
  `(dps, n_reach, n_all)`；**射手所在列 >= 该路最前僵尸所在列 → 打不到**（同列也算）。`_col_of(x)=(x-40)//80`。
- **★ 必须按波数控制**（否则"死得早→局短→累计量小"会误导）；**必须在同一检查点内部对比**。
- **布局**：向日葵 **98% 在 c0-c1**；攻击植物 **100% 在 c2-c5**；**c6-c8 从不种**（c0=房子侧，c8=入口）。
- **三指标（局级置换检验，p 全部 <1e-5）**：A 无条件"种在僵尸身后"占比 赢 0.537 vs 非赢 0.719；
  **B P(种在身后 | 别的路正在裸奔) 赢 0.485 vs 非赢 0.789**；**C 每帧"承压裸路数"（w5-15）赢 0.691 vs 非赢 1.437**。
  **★ 非平凡细节：赢局在有路裸奔时比平时更少种在身后（0.485<0.537），输局更多（0.789>0.719）
  → 赢局压力下会改变行为，输局不会。** ⚠️ 该分析混进了 `bc_seq32_100k`（崩掉的检查点）→ "三检查点复现"要打折。
- **★ 干预实验的教训**：触发点选得不对就会**剂量不足**（v5b 的 I7 只有 1/150 送达，因为"首次裸路"总在
  wave 1 / 僵尸 c9，那时 ahead/behind 都没有合法空格、且 149/150 阳光不足）→ **触发条件必须加"两种落点都有
  合法空格 + 阳光够"**。另外**只把轨迹岔开就能"救回"一批局**（I4 5/15、I7/I7b 各 9/15、I7c 12/15，I0 0/15）
  → **任何干预实验都必须配"随机合法动作"对照**，否则会把扰动伪影读成机制。

## 教师流水线（dsh/pvz-teacher）——**两台机器都能跑**

- **本机（Mac）**：`dsh` 在 `~/.local/share/mise/installs/node/24/bin`；`DSH_HOME=~/.local/share/pvz-agent/dsh-home`；
  mcp-venv `~/.local/share/pvz-agent/mcp-venv`；CLIProxyAPI `/opt/homebrew/bin/cliproxyapi`（v8.0.10）+
  `/opt/homebrew/etc/cliproxyapi.conf` + auth `~/.cli-proxy-api/`。**`GEMINI_API_KEY`** 是 CLIProxyAPI 的
  api-keys（**不是 Google 官方 key**），**值只放环境变量、不写入仓库**；Gemini 凭据是 **`antigravity` 类型**
  的 OAuth（带 refresh_token）；**`gemini-3.8-flash-high` 是网关唯一认的模型 id**。
- **台式机（WSL）**：全部**不用 Docker**；Node 24 `~/.local/opt/node-v24.21.0-linux-x64`；CLIProxyAPI
  `~/.local/opt/cliproxyapi/`；**`~/.pvz-teacher-env.sh`**（任务书 source 它，`GEMINI_API_KEY` 也放这里）。
- **★ 装机 5 坑（换机器必看；详见 `2026-10-12.md`、`docs/codex-night-plan-v5.md`）**：① `~/.venvs/ml` 的 pip
  配了 `bfsu` 镜像 → 装 mcp 报 **403**，要 `--index-url https://pypi.org/simple`；② **npm 11 默认跳过 install
  script** → `npm config set allow-scripts=... --location=user` 再重装；③ **`dsh plugin add` 建的 profile 会漏掉
  `@deepseek-ai/dsh-headless`** → **dsh 启 web 应用并永久卡住**（0 字节输出、`do_epoll_wait`），只需手工写进
  `package.json` 的 `dsh.profile.bundles`；④ **`$DSH_HOME/.credentials.yaml` 必须 600**；⑤ **bundle 里是绝对
  路径**，换机器改**部署副本**（`$DSH_HOME/profiles/<profile>/node_modules/@local/pvz-teacher/cordis.patch.yml`），
  **不要改仓库里的 `bundle/cordis.patch.yml`**（git 跟踪，会弄脏仓库）。
- **用法**：**会话必须带 `--task-manifest`**（否则诊断的是另一副卡槽）；**一局诊断 9 分钟 ~ 2.5 小时 /
  20~40 轮**（只挑少量局，且**把已有发现当"假设"交给它验证**）。**起会话前自检**：① `which dsh`
  ② bundles 必须是 `['@deepseek-ai/dsh-base','@deepseek-ai/dsh-headless','@local/pvz-teacher']`
  ③ `curl -H "x-goog-api-key: $GEMINI_API_KEY" http://localhost:8317/v1beta/models` → 200。

## 用户立场（长期有效）

- 整个奖励系统/训练范式需要从头重新设计；**此前所有项目规矩都是他在特定时点定的，他有权推翻**。
- **一切从事实出发，不从书本出发。** 不许把项目里自己定的规则当物理定律 —— 不许说"做不到"，
  要说"**需要改哪一行**"。
- **不要再各种微调参数**（原话）。要问"它到底在干什么"。
- 派活原则：**长跑交给 codex（它在台式机不会关），本 Agent 负责把关**；但**"跑出来的东西对不对"不能交出去**。
