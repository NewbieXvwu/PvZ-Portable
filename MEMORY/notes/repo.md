# 文档结构 / 仓库约定

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
- **密钥一律不写字面值**：文档里用环境变量引用（如 `$GEMINI_API_KEY`），值只放在本机 `~/.zshrc` 与
  台式机 WSL 的 `~/.profile` / `~/.bashrc` / `~/.pvz-teacher-env.sh`。
