# AGENTS.md — 在本仓库工作的 Agent 必须遵守的约束

本文件对本仓库的**所有** AI Agent 生效（Claude Code、Codex、Gemini CLI、WorkBuddy 等）。
它与 `TODO.md`（执行任务书）、`DESKTOP_HANDOFF.md`（两台机器交接）并列，但优先级最高的是**本文件里的体积与交付规则** ——
因为这些规则被违反过，代价是可测量的。

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

python3 scripts/hf_sync.py push <run>   # 上传最新检查点 + 结论层
python3 scripts/hf_sync.py pull <run>   # 在任意机器下载
python3 scripts/hf_sync.py ls           # 看远端有什么
```

**不要**用 tar / scp / 临时 SSH 隧道做常规同步，也不要用 Git LFS
（GitHub 免费额度 1 GB 存储 / 1 GB 月流量，装不下 43 MB × N 的检查点）。

---

## 3. 检查点保留策略

代码层面已加限制（`python/pvz_research.py` 的 `save()`）：`trained` 阶段只保留最新
`PVZ_TRAINED_CHECKPOINT_KEEP`（**默认 8**）个，`evaluated` / `initial` / `boundary` 永久保留，
另外任何被 `training_state.json` 指针引用的检查点永远不删。

**为什么是 8 而不是 1**：`scripts/research_late_policy_probe.py` 会 glob 最后 8 个 update
做末段策略探针。改成 1 会让那个探针静默退化。**不要调小它**，除非同时改掉那个探针。
`python/test_checkpoint_pruning.py` 里有一条 `assertGreaterEqual(TRAINED_CHECKPOINT_KEEP, 8)`
守住这个下界。

**不要绕过它。** 单个 500k 决策的 run 曾堆积 **80 个检查点（3.4 GB）**，其中真正必需的是少数：

| 阶段 | 数量 | 是否必需 |
|---|---|---|
| `trained` | 73 | 只有最新 8 个（续跑 + 末段探针） |
| `evaluated` | 5 | ✅ 分析脚本按这些节点取模型 |
| `boundary` | 1 | ✅ 冻结边界 |
| `initial` | 1 | ✅ 起点 |

手工清理用 `scripts/cleanup_research_artifacts.sh`（先空跑确认，`DRY_RUN=0` 才执行）。
**归档目录不能零散删单个检查点**，见 §4。

---

## 4. 删除前必须确认引用链

归档目录里的 `archive_manifest.json` 记录了**每个文件的 SHA256 与字节数**，
`resume.json` 还指向具体的检查点。

- **不能零散 `rm` 掉单个检查点** —— 会让完整性校验链断掉。
- 只能**整目录处置**，或重新生成 `archive_manifest.json`。
- 整条删除一条实验线前，先确认它的结论层（`learning_curve.json`、`evaluations/*.json.gz`、
  `provenance.json`）已经存在于 `artifacts/research_evidence/` 且已入库。

---

## 5. 文档纪律

- **新增根目录 `.md` 之前，先想能不能并入现有文件。** 现有核心文档见 `README.md` 末尾的导航表。
- 一次性进度报告用完即删，或并入 `TODO.md` / `DESIGN.md`。历史上根目录 md 曾从 7 份反弹到 21 份。
- 删文档后必做两件事：
  1. 改掉 `python/`、`scripts/` 里对它的引用
     （`train_pvz_ppo_task_family.py` 的 `protected_paths` 会对 `DESIGN.md` / `TODO.md` 做 SHA 校验，
     文件不存在会直接报错）；
  2. 扫断链（`](...)` 指向的本地文件是否还存在）。

---

## 6. 不要复活已废止的做法

- **分片归档不能当 git 交付通道用。** `scripts/archive_research_evidence.py` +
  `scripts/research_checkpoint_chunks.py` 是**真实且已测试**的工具（验证记录
  `artifacts/t5/perf/checkpoint_chunk_audit_v1.json`，真实 44.5 MB 检查点强制分片后 SHA 完整复原），
  但它**不是**允许把检查点塞进 git 的理由。2026-10-01 的实际情况是：
  - 分片闸门是**单文件 ≥ 100 MB**（`--single-file-limit-bytes` 默认 `100_000_000`），
    而检查点只有 44 MB → **闸门从未触发**（仓库里没有任何 `.gitparts/`）。
    于是 43 MB 的检查点被**整个文件** `git add -f` 提交，`artifacts/` 涨到 2.0 GB。
  - 真正坏掉的是**保留策略无上限**（每个 update 存一个，从不清理），已由 §3 修掉。
  - `CHECKPOINT_GIT_DELIVERY.md`（10-01 09:26 新增）把"分片走 git"写成了推荐做法。
    它的技术描述没错，但把一个**休眠的兜底机制**写成了常规交付通道，与现行规则冲突，
    **已删除**。恢复已归档证据的步骤搬到了 `DESKTOP_HANDOFF.md` §2.3，工具本身保留。
- **Git LFS**：已评估，免费额度不够（1 GB 存储 / 1 GB 月流量），不要引入。

---

## 7. 现有约定的权威顺序

当文档之间冲突时，按这个顺序裁决：

1. 本文件（AGENTS.md）—— 体积与交付规则
2. `artifacts/.gitignore` —— 机器可执行的规则，**可以用 `git check-ignore --no-index` 验证**
3. `TODO.md` —— 实验顺序与证据规则
4. `DESKTOP_HANDOFF.md` —— 两台机器的操作细节
5. 其他文档

判断某条规定是否还在生效，**不要靠猜**，用机制验证：

```bash
git ls-files -z artifacts logs | git check-ignore -z --stdin --no-index
```

`--no-index` 必须加，否则 git 会跳过已跟踪文件、永远返回空。
