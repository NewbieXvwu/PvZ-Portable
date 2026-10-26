# task113：deploy14e 定版（看门狗 + 门禁 v4 Holm 校正 + 两个根因修复）

主控决定（用户 10-26 授权主控对整个项目、包括门禁有完全裁量权）。

## 背景

- 第 20 轮门禁：总体 +0.2pp [−1.6, +2.0]，pool +5.5pp；day −3.6pp，单侧 McNemar p = 0.027，被判“显著区退步”而拒绝。
- 门禁对 6 个区逐个检验，每个都用 α = 0.05，没有做多重比较校正。在 Holm 校正下，最小 p 值要 < 0.05 / 6 ≈ 0.0083 才算显著，0.027 不算。已连续 9 轮被拒。

## 要做的（分支 deploy14e）

1. 把 deploy14e（task109 的 b7120205 / 60d8b233，基于 cefb9989）变基或合并到 **origin/deploy14d 最新 cbd96daf**。cbd96daf 是去掉 7 个 llmplay 层任务继承的 `demo_start` 的热补。
2. **门禁 v4**（`python/pvz_promotion.py`）：
   - 新增 `GATE_VERSION = 4`。面板、种子池、配对数、总体非劣 3pp 规则都与 v3 相同；
   - 区否决改为对各区单侧 McNemar p 做 **Holm–Bonferroni 逐步校正**，族错误率 0.05；
   - `policy(4)` 的 `zone_veto` 文字如实写明校正方法；
   - 不改 v1–v3 的 policy 文本，因为契约里有它们的哈希；
   - 测试：用第 20 轮的实际数字（6 区，day p = 0.027，其余见 `round_000020/evaluations/*/decision.json`）验证 v4 不否决；构造一个 p < 0.0083 的区验证 v4 照样否决；
   - DESIGN 登记 v4 及理由。
3. **单格配置错误不再让 qc 停**：
   - 规划起点格时，如果 `demo_start` 存档的种子不在该任务的 `seeds` 里，或者存档不存在，就把这格记为“配置无效”并退休、写日志，不派组；
   - 加测试。task109 的隔离若已覆盖 search_play rc=2，就补一个针对这种情况的测试。
4. **根因**：查 `scripts/llmplay/make_llmplay_tasks.py`（pvz-env 与 llmplay 分支），它生成 C 层等派生任务时复制了源族的 `demo_start`。改成不复制，并修到 pvz-env。
5. 如果分支 `demo2545`（task112：L25 / L45 示范起点格）已推送完成，一并合入；没有就不等。
6. 更新 `scripts/qc_deploy14e.sh`（台式机复制到 `~/pvz-logs/qc_deploy14e.sh`）：
   - REV、CODE_FILES、基线 SHA 按 cq 现状重算；
   - 不加回填；
   - 门禁版本切换写清楚，切换要在轮边界。
   - **不要执行**，主控执行。

## 约束

- 只跑相关测试。
- 最多 2 个游戏进程。
- 遵守 AGENTS.md §6.5。
- 推送 deploy14e，结果写进本文件末尾（合并时同步到 pvz-env 的同名文件）。
