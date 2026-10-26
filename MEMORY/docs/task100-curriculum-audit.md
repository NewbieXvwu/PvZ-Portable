# task100：deploy14 课程审计——每个新格都真的在采吗？

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-llmplay`）。只读任务：读 `~/pvz-logs/curriculum_state.json`、`curriculum.log`、`qc.out`、`curriculum_state.pipeline/`、cq 的清单与 configs。**不碰 tmux `qc`、cq 的文件、state**；如需复现，用拷贝。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/qcwatch` 的只读副本即可（代码改动归 task98，不要改）。

## 为什么
deploy14 上线后，主控已经发现一个“格在却不采”的 bug：原生 L25 的 `pool_l25_orig_train` 带了倍率阶梯 `ladder`，被起点格逻辑当成起点格退休，每次重启都“added → retired”，从来没有采过。修复 43458aa8 已在 cq 里就位，下次重启 qc 时生效。很可能还有同类问题。

## 要做的
1. **对账**：列出 deploy14 计划进课程的全部格，逐格给出状态与证据：
   - fog_user_v2 L31；
   - fog79b L36 / L37 / L38；
   - 原生 L25；
   - 13 个示范起点格（雾夜 `_hv2` 加屋顶）；
   - 门禁 v3 的 fog / roof 区。
   每格要列：是否在 state 的 cells 里、是否被退休（原因）、08:50 以来被排了几组、当前档位与阶梯位置、窗口胜率、teacher_rates 底数是否命中。
2. **异常清单**：
   - 从没被排过的格；
   - 反复 added / retired 的格；
   - 档位不动或越界的格；
   - 种子块剩余不足 3 轮的格（用于 task99 的优先级）；
   - teacher_rates 缺底数的格；
   - 门禁 v3 首轮是否正确领了种子池第二段之前的块。
   逐条给出代码位置与根因猜测。
3. **round 18 的训练失败**（plane_of 72/84）已由主控热补。请核对热补之后第 18 轮重新打包、训练能否完成，以及门禁 v3 首次运行的结果。
4. 结论写 `MEMORY/docs/curriculum-audit.md`，提交到分支 `qcwatch`（与 task98 同一分支，只加文档），`git push origin qcwatch`。修复建议交主控分派，不要自己改生产。

## 规则
- 只读生产；只用绝对路径；`PVZ_NO_POPUP=1`；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行。

## 结果

## 问题
