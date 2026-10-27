# task126：挑选面板加“没见过的卡组”，每个配方训两个种子（并入 deploy14h）

主控 2026-10-27 09:45 派给台式机 codex-t125（plus）。分支 `selheldout`，基于 `origin/rebalance2`（deploy14h 内容：p=0.5 再平衡 + 雾夜区 save-candidates + task123 共享映射 / 准入）。

## 背景（task125 结论）

- 第 21 轮 heldout −33pp 不是打包 bug：同数据同配方，第 20 轮两个种子在 L7 换卡组同 32 种子上 **4 vs 28 胜**；第 21 轮 A 5 胜、B 0 胜。没见过的卡组上的泛化是**训练种子的抽签**。
- 现行挑选面板（`selection_manifest.json`）没有换卡组任务，K=2 又是 A、B 各一个，所以挑选筛不掉泛化差的种子。

## 主控决定

1. **挑选面板加入换卡组任务**：用**与门禁 heldout 不同**的卡组（门禁用 `day_l7_eval_d7p_holdout1`、`day_l9_eval_d7p_holdout2`；挑选新建 holdout3 / holdout4 等，按 AGENTS §8 合法覆盖地形，新任务族名，DESIGN 登记），种子从挑选池取、不与门禁 / 训练 / 记分板重叠。份额与现有挑选任务相当（不要让它主导总分）。
2. **K=4：A（dw4i2）× 2 个种子 + B（dw4i2_rebalanced_p05）× 2 个种子**，串行训练（task123 准入已会自动串行）。轮报告记每个候选的配方、种子、挑选分（含换卡组分项）。若你测算单学生训练（含装载）> 70 分钟，改为 K=3（A×2 + B×1）并说明。
3. 合进 `scripts/qc_deploy14h.sh`（REV 换成本分支提交；CODE_FILES 补新文件；未激活 / 回滚分支仍是 deploy14g 在役参数），更新台式机 `~/pvz-logs/qc_deploy14h.sh`，不执行。

## 约束

- AGENTS §6.5：只跑相关测试、一次 dry-run；不加额外审计。
- 不动 tmux qc；结论写在本文件末尾，推送分支。
