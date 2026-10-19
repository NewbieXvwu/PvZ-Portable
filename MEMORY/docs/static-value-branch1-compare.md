# 两份静态价值网 branch1 的同集对照

可快进合入的提交号：`23ee7f91aba2d5ef37be1b0ebae49cb3ae4493d8`（已验代码合并点，包含 `origin/pvz-env` 的 `1ffb38a`；后续文档/对照提交在其后，整支仍可快进）。第一项回归全部通过，第二项尚未开始实测，不改正式老师。

同步以主线删除为准，未恢复 PPO、规则脚本或 BC；自有审计导入均走 `pvz_tasks`。CPU/CUDA 在 `build-task12/` 独立重建，128 条 C/C++ 编译命令已确认 `-ffp-contract=off`；历史冻结程序保持原样。

对照模型：Mac `~/pvz-logs/mac_branch1/model.pt`（585243 字节，MD5 `cda25bfcf1b6dd58a226787fcccb9375`），台式机 `~/PvZ-search/artifacts/hard/value_branch1/model.pt`（584603 字节，MD5 `b16a75242bd98bd1f0bc2f791f2c8e3d`）。两者各自旧留出集指标不可直接横比，本任务将在同一评估轨迹、同一候选池和共同未来上对比。

同步验收：Python 208+9=217项全部通过，无skip，含原生CUDA合批/数组扩容/内核切换；[构建与完整回归证据](static-value-branch1-merge-validation.json)。
