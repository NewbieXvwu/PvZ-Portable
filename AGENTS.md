# AGENTS.md — 在本仓库工作的 Agent 必须遵守的约束

本文件对本仓库的**所有** AI Agent 生效（Claude Code、Codex、Gemini CLI、WorkBuddy 等）。
它与 `DESIGN.md`（设计与当前计划，§13–§14）、`MEMORY/docs/student-bottleneck-plan.md`（当前进度）并列，但优先级最高的是**本文件里的体积与交付规则** ——
因为这些规则被违反过，代价是可测量的。

> `CLAUDE.md` 是本文件的**软链**（Claude Code 读它即读本文件），不要单独维护一份内容。

规则不是建议。`artifacts/.gitignore` 就是规则本身；**`git add -f` 绕过它等于违反本文件**。

---

## 1. 体积预算（最重要，先读这条）

2026-10-01 的审计实测：

| 指标 | 实际 | 规则认可 |
|---|---|---|
| `artifacts/` 推入 git 的体积 | **2.0 GB** | 9.4 MB |
| 超标倍数 | | **218 倍** |
| 被跟踪但现行 `.gitignore` 会拦下的文件 | **659 / 765（86%）** | 0 |

**硬性约束：**

1. **禁止 `git add -f`**（以及 `-f` 的任何等价形式）。`.gitignore` 说忽略就是忽略。
2. **单个 commit 新增的二进制文件合计必须 < 10 MB。** 提交前自查：
   ```bash
   git diff --cached --numstat | awk '{s+=$1} END {print s" 行"}'
   git diff --cached --stat | tail -1
   ```
   发现 `.pt` / `.npz` / `.log` 出现在暂存区就撤销。
3. **永远不要提交**：检查点 `.pt`、评估分片 `.npz`、滚动日志 `.log`、`artifacts/research_evidence/`、
   `artifacts/research/`、`logs/`。
4. 提交前跑一次自查：
   ```bash
   git ls-files -z artifacts logs | git check-ignore -z --stdin --no-index
   ```
   输出应为空。非空说明仓库里已经有违规文件（不是本次提交造成的），不要用 `-f` 继续扩大。

---

## 2. 大文件走 Hugging Face Hub，不走 git

分工是明确的：

| 通道 | 传什么 |
|---|---|
| **git** | 代码、实验配置、门禁、结论层（KB 级 json）、文档 |
| **Hugging Face Hub** | 检查点 `.pt`、评估分片 `.npz` 等大文件 |

```bash
export HF_TOKEN=hf_xxx
export PVZ_HF_REPO=<用户名>/pvz-agent-artifacts

python3 scripts/hf_sync.py push <run>          # 上传检查点集 + 结论层
python3 scripts/hf_sync.py push <run> --dry-run # 先看会上传什么
python3 scripts/hf_sync.py pull <run>          # 在任意机器下载
python3 scripts/hf_sync.py ls                  # 看远端有什么
```

**上传范围（每个 run 约 286 MB）**：`evaluated` / `initial` / `boundary` / `resumed`
全部，加最新一个 `trained`，**再加 `resume.json` 指向的那个文件**，以及结论层。

`resume.json` 那一项是硬要求，不是可选项。实测四个 `reward_r*_v2` 的 `resume.json`
都指向 `boundary` 检查点、而不是最新的 `evaluated`；只传最新 `evaluated` 的版本
下载回来是**接不上**的（2026-10-01 实测）。改动 `_collect()` 时别把这条去掉。

**不要**用 tar / scp / 临时 SSH 隧道做常规同步，也不要用 Git LFS
（GitHub 免费额度 1 GB 存储 / 1 GB 月流量，装不下 43 MB × N 的检查点）。

**HF 仓库自 2026-10-24 起公开**（与 GitHub 同为 LGPL-3.0），只当两台机器之间的中转站：

- 只放模型权重和数值化的对局记录（状态、动作、示范轨迹）。
- **禁止上传任何游戏资源或衍生画面**：图片、音效、`main.pak`、可执行文件、截图、校准图。游戏版权归 PopCap / EA。
- 不放密钥与账号文件。
- 文件被对方取用后就删（用户 10-24："不要囤积"），不在 HF 归档实验数据。

---

## 3. 检查点

PPO 训练线（`python/pvz_research.py`、`train_pvz_ppo*.py`）连同它的检查点保留 / 裁剪 / 分片归档工具
已于 2026-10-19 删除（tag `pre-cleanup-20261019` 可取回）。现行的学生网络与价值网络检查点都是单个
`model.pt`（`artifacts/student/<名>/`、`artifacts/hard*/value_*/`），不入 git，跨机器走 HF `transfer/`。

---

## 4. 删除前必须确认引用链

- 删代码 / 文档前，先查谁还引用它（`git grep`），改掉引用再删；整条线退役先打 tag。
- 本机若还留着旧的 `artifacts/research*/` 目录，要删就**整目录**删；其中 `archive_manifest.json`
  记录了每个文件的 SHA256，零散删单个检查点会让校验链断掉。
- **`git rm` 会同时删工作区文件**，删前确认文件都被 git 跟踪（`git ls-files --error-unmatch <path>`）。

---

## 5. 文档纪律

- **新增根目录 `.md` 之前，先想能不能并入现有文件。** 现有核心文档见 `README.md` 末尾的导航表。
- 一次性进度报告用完即删，或并入 `DESIGN.md` / `MEMORY/docs/`。历史上根目录 md 曾从 7 份反弹到 21 份。
- 删文档后必做两件事：
  1. 改掉 `python/`、`scripts/`、`dsh/` 里对它的引用；
  2. 扫断链（`](...)` 指向的本地文件是否还存在）。

---

## 6. 不要复活已废止的做法

- **检查点不进 git，分片也不行。** 2026-10-01 曾有 43 MB 的检查点被整个 `git add -f` 提交，
  `artifacts/` 涨到 2.0 GB；当时的分片归档工具闸门是单文件 ≥ 100 MB，从未触发。工具已随 PPO 线删除。
- **Git LFS**：已评估，免费额度不够（1 GB 存储 / 1 GB 月流量），不要引入。
- **手写规则策略**：规则脚本基线（`scripted_baseline.py`）已删除。用户明确不要手写规则或公式，
  诊断与改进都走学生 + 搜索老师 + 数据。

---

## 6.5 禁止挡在关键路径上的形式主义验证（2026-10-26 用户定，严格执行）

看上去“正常、合理、安全”的验证，如果同步卡在关键路径上，严重拖慢真正的工作，就是违规。

事故经过：llmplay 包的 `resume()` 在 GPT Pro 弱沙箱里把 191 份旧存档逐份串行重放复验，几十分钟过去模型一直在等，没有继续干活。这些存档主控导入时还会在台式机上全部重放，沙箱里这一遍完全多余。

**规则：**

1. 设计或加入任何验证、检查、预检之前，先估算它在**实际运行环境**里要多久：慢机器、全量数据、最坏情况。再估算它挡住了谁。
2. 能用便宜的手段就用便宜的：摘要、大小、抽样都可以。全量重放、全量重算只放在下游，而且下游本来就要做的地方，不要在上游重复做。
3. 慢的验证必须**异步、并行、或者推迟**，不能让 Agent、采集、训练或外部模型干等；也不能每次续跑、每次启动都从头做一遍。
4. 只有一种情况允许阻塞：不做就会造成真实损失（坏数据进训练、生产状态被破坏），而且没有更便宜的替代。即便如此，也要报告进度和预计耗时。
5. 评审别人的代码时，把“这个检查要多久、挡住了谁”当成必查项。
6. **Agent 自身的工作方式同样适用（主要针对 Codex / GPT 系：干什么都想再校验一遍）。**
   - 验收要和改动规模相称：小改动只跑相关测试。
   - 不要每改一处就重跑整套测试、冷解压、断网容器、跨机器逐位比对、全机种子审计。这类重型验收在任务最后做一次，并且只在任务书要求时做。
   - 任务书没要求的审计不要自己加；报告里也不要大段罗列做了哪些校验。

7. **台式机是生产机，测试不能抢生产内存。** 2026-10-27 12:09，一个 agent 在台式机跑 `pytest python/test_qc_resilience.py`，涨到 17 GB，把生产 qc 挤到 OOM 崩溃，看门狗两次重启后停机，停采约 35 分钟。
   在台式机上：只跑与改动直接相关的单个测试（`-k`），一律加内存上限（如 `ulimit -v 8000000`），并先看 `free -g`；需要大内存的测试放到 Mac 上跑。

8. **Mac 同样不能被测试拖垮。** 2026-10-28，一个 agent 在 Mac 上同时两次整套运行 `test_curriculum_queue.py`、`test_curriculum_pipeline.py`、`test_qc_resilience.py`。其中 `test_persistent_state_error_*` 无限写 `curriculum.log`，三次共写出约 20 GB，把磁盘写满；同时内存暴涨，macOS 冻结了所有应用，经 Mac 中转的台式机连接也断了约 2.5 小时。
   - **禁止整套运行这三个文件**，只能用 `-k` 跑与改动直接相关的用例。
   - 每次运行都要加 `timeout 300` 和 `ulimit -f 2000000`（单个文件不超过约 1 GB）。
   - 同一时间只能开一个 pytest。

## 6.6 同一逻辑只实现一次；原型不能持久，长期必须迁移到 C++（2026-10-31 用户定）

事故经过：GPT Pro 包（`scripts/llmplay/`）自己实现了一套对局存档写入，用明文 `frames.jsonl`。生产早在 task32/33 就把存档改成了 npz，体积只有原来的 7.5%，这条路却没跟上。结果上传包膨胀到几 GB，最后一版传到一半被截断；台式机导入库 6.9 GB 全是明文。双入口必然分叉，这只是其中一次。

**规则：**

1. **同一件事只有一个实现。** 适用于游戏规则、观测特征编码、老师搜索、整局对局、存档 / 帧读写、任务与卡组定义、导入、门禁。
   - 新代码必须调用现有实现，不要另写一份“简化版”或“兼容版”。
   - 发现已有两份时，合并成一份，删掉另一份。
2. **生产路径以 C++ 为准。** 游戏规则、特征、搜索、推理的权威实现在 C++。Python 只保留调用 C++ 的薄接口：
   - 训练（PyTorch）；
   - 外部模型使用的逐步对局接口（reset / step / observe）；
   - 示范重放、调试与分析工具。
   薄接口不能自己重算一遍 C++ 已有的逻辑。
3. **Python 原型可以有，但不能持久。** 新特征、新网络、新搜索思路可以先在 Python 里做小规模实验，例如叠加在 C++ 特征之上，或在 PyTorch 里逐步对局。
   - 原型被生产采纳时，同一批改动里移进 C++。
   - Python 版删掉，或改成调用 C++。
   - 移植只做一次小的一致性对比（例：雾关整局原生 A/B，90 局胜负翻转 0 次），不常驻重复校验（§6.5）。
   - 没被采纳的原型，实验结束就删。
4. **部署脚本不复制粘贴。** 不要每版复制一份几乎相同的 `qc_deployN.sh`。启动命令只有一处来源，看门狗与部署共用。

## 7. 现有约定的权威顺序

当文档之间冲突时，按这个顺序裁决：

1. 本文件（AGENTS.md）—— 体积与交付规则
2. `artifacts/.gitignore` —— 机器可执行的规则，**可以用 `git check-ignore --no-index` 验证**
3. `DESIGN.md`（§13–§14 是现行方案）与 `MEMORY/docs/student-bottleneck-plan.md`（现行进度）
4. 其他文档

判断某条规定是否还在生效，**不要靠猜**，用机制验证：

```bash
git ls-files -z artifacts logs | git check-ignore -z --stdin --no-index
```

`--no-index` 必须加，否则 git 会跳过已跟踪文件、永远返回空。

**两个坑（都踩过）：**

1. **不要用 `-v` 做这个检查。** `git check-ignore -v` 会把**否定规则**（`!...`）也算作命中并
   打印出来、退出码 0，于是白名单里的文件会被误报成违规。要 `-v` 就必须看行首是不是 `!`。
   上面的写法不带 `-v`，只列出真正会被忽略的路径。
2. **要判断"某个文件到底能不能提交"，别问 `check-ignore`，直接问 git：**
   ```bash
   git add -n <path>     # 能加就打印 add '...'，被忽略会报 pathspec 错
   ```
   这是唯一不会骗人的判据。

**另：`git rm` 会同时删工作区文件。** 删归档目录前先确认文件都被 git 跟踪
（`git ls-files --error-unmatch <path>`），否则删掉就真没了。

---

## 8. 任务与卡组（2026-10-03 定）

**卡组必须覆盖关卡地形。这是任务设计的硬约束，不是偏好。**

本仓库的全部实验（脚本基线、教师诊断、RL 训练）都跑在冒险模式 50 关上，
而关卡地形由 `Board::PickBackground`（每 10 关一个区）决定：

| 区 | 场景 | 地形要求 |
|---|---|---|
| 1–10 | 白天草地 | 无 |
| 11–20（含 35） | 夜间草地（天不掉阳光；有墓碑格） | 无（墓碑格种不了） |
| 21–30 | 白天泳池 | 水路必须有睡莲(16)或其升级香蒲(43) |
| 31–40（35 除外） | 夜间泳池有雾 | 同上 |
| 41–49 / 50 | 屋顶 / Boss | 屋顶格要有花盆(33)——本环境屋顶关开局预置 c0–c4 |

因此：

1. **禁止全关卡共用一套固定卡组。** 之前 `(0,1,2,3,4,5)` 跑所有关，
   泳池关两条水路零火力 → 结构性输局。诊断这种局只会得出"卡组×地形不匹配"
   这种一次性的结论，产不出可教的政策缺陷数据。新关卡入选任务集前，
   先用 `scripts/pvz_tasks.py` 的 `deck_for_level`（场景判定从
   `pvz_constants.background_for_level` 现解析，含第 35 关 ScaryPotter
   特例）核对覆盖。**反过来同理：打不过的时候，必须先考虑是不是该换卡组，
   再考虑改策略** —— 卡组覆盖不了地形的局是结构性输局，在动作层怎么改都救不回来；
   把"该换卡组"误诊成"策略失误"是最坏的一种错误结论。
2. **升级植物必须声明所有权。** `Plant::IsUpgrade`（Plant.cpp）列出的
   升级植物（40–47，含香蒲 43）进卡组时，`PlayerProfileContext` 的
   `owned_upgrade_plants` 必须包含它，且槽位数 = 卡组长度（`LawnApp.cpp:1329`
   校验，违反直接拒绝 reset）。升级植物**只能种在各自底座上**
   （香蒲→睡莲上，Board.cpp:2849），空格直接种被 `Board.cpp:2866` 拦下。
3. **卡组长度上限 10**（`SEEDBANK_MAX`，GameConstants.h），下限 6。
4. **改任务卡组 = 换任务族。** 卡组变化会改变任务语义与全部历史证据的
   可比性。改之前必须：新开任务族名（不覆盖旧名）、在 DESIGN.md 的任务
   清单里登记、并且旧存档的 `whatif` 重放不受影响（重放走存档里记录的
   deck，不读现行规则）。

5. **戴夫推荐卡（`CrazyDavePickSeeds`）是随机的，不能当“这张卡在这关有用”的依据。**
   生成卡组（扰动族、留出族、挑选 / 门禁卡组）时，用手写的**按地形黑名单 / 白名单**
   排除废卡（例：白天无气球无雾时的三叶草、只产钱的金盏花），并在 DESIGN 登记名单。
   用户 2026-10-27 明确允许：这属于任务设计，不属于被禁止的“手写策略规则”。
   曾因此出错两次：d7q 用戴夫候选当卡池；门禁 heldout1（L7 带三叶草）连续否决 9 轮晋升。

配套实现（改卡组规则只改这里，别处引用）：
`scripts/pvz_tasks.py` 的 `deck_for_level` / `profile_for_deck`；
场景文字渲染在 `scripts/episode_query.py` 的 `_fmt_scene`。

---

## 9. 项目记忆（跨平台）

记忆文件在仓库的 **`MEMORY/`**（随 git 同步），**任何 Agent 平台都可直接读写**。
WorkBuddy 侧另有软链 `.workbuddy-ai/memory → ../MEMORY`，所以两条路径指向同一批文件。

下面这段是 **WorkBuddy 记忆系统「工作区记忆层」的原文**，只做了去平台化微调
（路径改成仓库相对的 `MEMORY/`、`Edit tool` 改成"你的文件编辑工具"、删掉平台专有的跨会话检索那条）。
**不要再改写成自己的话** —— 保持与各平台记忆协议可对齐。

### Workspace Memory (read/write)

Directory: `MEMORY/` | Scope: current project only

Files:
- `MEMORY/YYYY-MM-DD.md` — daily work log. **Append-only**, never overwrite.
- `MEMORY/MEMORY.md` — index of curated long-term project notes (**links only, no summaries** — open the linked file for content). Limit: 3,000 chars/session.
- If today's log does not exist, create the directory and dated file first.

Retrieving historical context: choose the right source as needed — no need to read everything.
- This project's past work → read local daily logs (most recent first) or `MEMORY/MEMORY.md`.
- Both sources can be used together if local logs are incomplete.

No historical dependency → skip reading memory files.

Role boundary: Workspace memory is supplemental only. It does NOT replace the assistant's normal reply, final answer, or any user-requested deliverable.

**When to write (MUST follow):** Immediately after completing substantive work, append a brief note to `MEMORY/YYYY-MM-DD.md` using your file-editing tool. Substantive work includes:
- Built or modified a website/application
- Fixed a bug
- Wrote or generated a report or document
- Completed code refactoring or architecture changes
- Chose a technical approach (framework, design pattern, etc.)
- User shared project conventions or preferences → also update `MEMORY/MEMORY.md` in place

Daily logs are append-only. Do NOT record transient information (search results, temporary paths, tool errors). Only persist what has lasting value across sessions.

Maintenance: Distill daily logs older than 30 days into `MEMORY/MEMORY.md` by topic, then delete the old files. Do not store secrets unless the user explicitly asks.

**本项目补充（不属于原提示词）：**

- `MEMORY/MEMORY.md` 是**纯索引**：只放目录与链接，**不放任何摘要**（放了摘要，模型就不肯去读正文了）。
  长期笔记的正文在 `MEMORY/notes/*.md`，按主题拆分。
- `MEMORY/docs/` 放历史报告与一次性分析（codex 任务书/报告、失败模式分析等），**不属于"记忆"**，
  不受 append-only 约束。
- **密钥一律不写字面值**：文档里用环境变量引用（如 `$GEMINI_API_KEY`），值只放在
  本机 `~/.zshrc` 与台式机 WSL 的 `~/.profile` / `~/.bashrc` / `~/.pvz-teacher-env.sh`。

