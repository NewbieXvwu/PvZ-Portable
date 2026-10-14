# 奖励与训练机制（结论；推导见 `2026-10-04.md`）

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
