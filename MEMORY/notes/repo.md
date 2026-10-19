# 文档结构 / 仓库约定

## 文档结构

根目录只留 `README.md` / `AGENTS.md`（= `CLAUDE.md`）/ `DESIGN.md`；现行计划与进度在
`MEMORY/docs/student-bottleneck-plan.md`。2026-10-19 删掉了 PPO 线的全部代码、报告、门禁、实验清单，
以及规则脚本策略（tag `pre-cleanup-20261019` 可取回）。**新增根目录文档前先想能不能并入现有文件**
（历史反弹到 21 份）。删文档后必做：① 改掉 `python/`、`scripts/`、`dsh/` 里的引用；② 扫断链。

## 仓库约定（体积/交付规则见 `AGENTS.md` §1–§4，这里只记它没写的）

- 封存证据不可改：`artifacts/adventure2_level7/training_summary.json` 的 `episode_hash` 只能保留、
  不能改语义；新算法走新函数名 + provenance 记算法名。
- 两机同步两条通道（分支 `pvz-env`，origin `NewbieXvwu/PvZ-Portable`，**不用 tar/scp/mnt/c**）：
  git 走代码/配置/门禁/结论层，`.pt`/`.npz` 走 Hugging Face Hub（`scripts/hf_sync.py`，仓库
  `realnewbiexvwu/pvz-agent-artifacts`），**永远不进 git**。**HF 仓库文件数上限 20,000**（10-09 曾到 19,985）。
- `artifacts/` 默认全忽略，白名单只有冻结的种子集（`adventure2_level7/seeds/`、`multiterrain/seeds/`）。
- **密钥一律不写字面值**：文档里用环境变量引用（如 `$GEMINI_API_KEY`），值只放在本机 `~/.zshrc` 与
  台式机 WSL 的 `~/.profile` / `~/.bashrc` / `~/.pvz-teacher-env.sh`。
