# task50：雾区与屋顶区的代码合进主线，构建新部署引擎，让 qc 能采屋顶 L41

> 主控 2026-10-23 22:30（台式机时间）写。执行者：台式机 Codex（Business，codex6）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/zones`（分支 `zones`，从 pvz-env 拉出，主控已建好）。最多 2 个游戏进程；**不碰 tmux `qc`**、`deploy` 与 `deploy2` 工作树。

## 背景
- **task46 屋顶**（分支 `roof`，codex4 已完成）：老师在 L41 原版 15/16，可直接进课程；L43 在 0.5 倍为 14/16。详见 `roof` 分支的 `MEMORY/docs/task46-roof-zone.md`。
- **task45 雾区**（分支 `fog`，codex3 仍在做诊断）：已有雾观测（human / xray 族属性）、雾量编码、三种开雾植物、推演输入屏蔽，**改了引擎观测**；非雾存档 16.6 万帧 0 差异。目前雾区没有可进课程的关。
- qc 现在用 `deploy2` 的 PGO 二进制（f2ae755a）与 `cq` 工作树的 Python。要让 qc 采屋顶，就得把两边代码合进主线，并在新目录构建部署二进制。

## 要做的
1. 在 zones 里依次 `git merge --no-ff origin/roof`、`git merge --no-ff origin/fog`。冲突主要在 `make_multiplier_tasks.py` 的 `FAMILIES`、DESIGN §14.x 编号、`curriculum_queue.py` / `unified_panel.py`、日志。规则：
   - 两边的族都保留，既有族顺序、档距与种子起点不变；
   - DESIGN 编号顺延，修正文中所有引用；
   - 日志两侧全保留，按时间排。
2. 跑 `python -m pytest python -q` 要求全过。用本工作树 build，`PVZ_RESOURCE_DIR=/home/newbiexvwu/.cache/pvz-research-resources`。
3. **新部署工作树**：
   - `git -C /home/newbiexvwu/PvZ-search worktree add --detach /home/newbiexvwu/PvZ-search/.claude/worktrees/deploy3 <zones HEAD>`；
   - 拷 `artifacts/perf/engine_roots.json`，跑 `scripts/build_pgo.sh bench`；
   - `grep -c fprofile-generate build/build.ninja` 必须是 0。
4. **门禁**，deploy3 对 deploy2：
   - 4 个非雾存档整局逐帧重放 0 差异；
   - 64 root 叶摘要逐位相同；
   - 配对吞吐在噪声内。
5. **课程**：
   - 屋顶 L41 原版按 task46 的建议进 `FAMILIES` 或任务清单；雾区不进；
   - zones 工作树里 `curriculum_queue.py` dry run 能列出 L41 的 cell 与完整命令，命令里的引擎是 deploy3；
   - 写清 qc 重启命令。
6. 推 `zones`，在"结果"写：合并提交号、测试数字、deploy3 SHA、门禁数字、dry run 输出摘要。主控审后快进 pvz-env 并切 qc。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树。
- fog 分支之后还会有新提交（诊断），你合的是当前状态；之后的增量由主控再合。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥。
- 每完成一小步就提交、push，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- 步骤1a：合入 origin/roof 当前快照 2dcaef33；自动合并无冲突，保留屋顶全部族与种子映射，默认课程只准入 L41 r7 原版。

- 步骤1b：roof 合并 a12ca704；合入 origin/fog 快照 f1dce346。雾族保留原索引8/9，roof置末且沿用独立48M段；DESIGN 雾18/19、屋顶顺延20；双方日志按提交时间交错保留。课程显式禁入全部fog族，仅准入屋顶L41。

- 步骤1c：fog 合并 e9ff587a 已推送；课程/倍率定向28 passed + 4 subtests。修复种子宽度测试将独立roof误算进700k段的假设，新增所有旧族固定种子起点与roof/fog准入回归。

- 步骤2：本工作树Release构建完成；PVZ_RESOURCE_DIR指定缓存资源，全套Python **382 passed + 133 subtests passed**（66.31秒、零跳过）。分支任务逐字段审计roof318/fog366项全同，合并414项，见task50-zones-registry.json。

- 步骤3a：新建detached deploy3于7a67066a，复制perf的64-root engine_roots.json；未改deploy/deploy2。

- 步骤3b：deploy3 PGO两阶段成功，源7a67066a5e62cec47432eaef87c542f6782270c3；二进制SHA256 `e231cdb9f872e50a7eec55180c8ca2b156076e2f0ffaba41a4100ab10d39a12d`，140份profile，最终ninja generate=0/use=161。构建证据task50-zones-build.json。

- 步骤4：门禁证据提交31697972；deploy3/deploy2四存档整局逐tick完整观测 **166138帧/引擎、0差异**；64 root×3续走=192叶摘要及逻辑统计在五轮/十次运行全部逐位相同。配对CPU吞吐比中位1.04437×，范围0.98294–1.08156×，log比值t近似95%区间0.99496–1.08425含1，噪声内无退化；见task50-zones-gates.json。新增zones_replay_gate.py可复跑当前部署对照。

- 步骤5：本树清单414任务/847872种子逐卡组合法；扫描330494份JSON、835020个既有种子，重叠0。隔离状态dry run列出38组/1824局，含 `roof_l41_train_r7 mult=1 games=48` 与deploy3完整collect命令，fog cell=0；摘要task50-zones-course.json。

- 步骤6：交付审计通过：两次no-ff父链、双方日志全保留、DESIGN编号唯一、部署二进制/清单哈希、测试/门禁/课程证据均已复核。所有步骤新增二进制0字节、受跟踪忽略文件0；本任务游戏进程已归零，qc/deploy/deploy2未修改。审计task50-zones-completion-audit.json；提交并推送zones交主控审核。

- [A.1] 合入pvz-env快照9b8ae8b13a7389104e474b7d19534f229c601c96，保留Mac clang PGO与原GCC路径；日志双方按提交时间保留。下一步合vcpp并重跑完整老师门禁。

- [A.2] pvz-env合并8df7fa90；合入vcpp实际快照8e1f91f76b56c690b4b3a5c3134594a8b5f16d5b。保留CPU专用路径、雾human/xray、指标与worker复用，价值叶子/静态候选改用引擎实现，旧CUDA文件保持删除。直接Board价值编码补接雾可见性，新增双编码/双观测/静态/推演真实引擎回归。

- [A.3] vcpp合并6b840b1e已推送；本树Release重建完成，全套 **395 passed + 133 subtests passed**（85.71秒，零跳过），含双编码×human/xray直接Board逐位特征、静态与叶值回归。诊断/重标注调用已迁移到导出价值权重接口；老师门禁冻结原task37配置、旧源码/旧二进制，三任务各32局两臂全部新跑，不复用旧局。见task50-a-validation.json。

主控审核并快进pvz-env、让旧qc在组间正常结束后，重启命令如下（本文仅记录，未执行；上限2游戏进程）：

```bash
/usr/bin/tmux new-session -d -s qc '/usr/bin/env PVZ_NO_POPUP=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/newbiexvwu/.venvs/ml/bin/python3 /home/newbiexvwu/PvZ-search/.claude/worktrees/zones/scripts/curriculum_queue.py --rounds 100 --workers 2 --round-games 768 --games-per-group 48 --manifest /home/newbiexvwu/PvZ-search/.claude/worktrees/zones/artifacts/stage4/multiplier_panel.json --state /home/newbiexvwu/pvz-logs/curriculum_state.json --log /home/newbiexvwu/pvz-logs/curriculum.log --out /home/newbiexvwu/PvZ-search/artifacts/hard --frames /home/newbiexvwu/PvZ-search/artifacts/student/frames_v2 --recipe /home/newbiexvwu/PvZ-search/.claude/worktrees/zones/recipes/dw4i.json --executable /home/newbiexvwu/PvZ-search/.claude/worktrees/deploy3/build/pvz-portable --resource-dir /home/newbiexvwu/.cache/pvz-research-resources 2>&1 | /usr/bin/tee -a /home/newbiexvwu/pvz-logs/qc.out'
```

沿用生产state/偏移和dw4i配方、v4/branch1，控制器自动加入L41、暂缓所有fog；主控须先确认旧控制器已退出、现有组间停止标志已清理，再执行上述命令。

## 问题

- 历史fog四存档的摘要不能直接充当deploy2参考：deploy2重放L7通过，但L12第0帧已与历史摘要不同；保留原始动作/摘要，使用当前deploy2重生成完整逐tick参考，再由deploy3严格比对全字段。

- 最终父链核验更正：初读origin/fog为15e947b3，merge执行时共享ref已更新为f1dce346；实际e9ff587a第二父节点为f1dce346，额外含追加D报告及诊断脚本。测试、PGO、门禁均基于该实际合并内容；任务字段审计也已对实际父节点重算，366项全同。

## 主控追加 A（2026-10-23 23:05）：一并合 task37（价值网进引擎）与 pvz-env 新提交，重建 deploy3
task37（分支 `vcpp`，codex2-b 已完成）：老师一致率 100% / 99.99% / 100%，**整局 CPU 降到 0.78–0.82**；报告见 `vcpp` 分支的 `MEMORY/docs/value-in-engine.md`。qc 只切一次，所以合在一起：
1. 在 zones 依次 `git merge --no-ff origin/pvz-env`（含 task47 Mac clang PGO，GCC 路径不变）和 `git merge --no-ff origin/vcpp`。冲突两边功能都留。`search_play.py` 的叶值路径以 vcpp 的引擎内实现为准；要与 v4（编码 v2）、雾观测（human / xray）都对上。
2. 全套测试全过。task37 的老师一致率门禁在 zones 上重跑一遍（三任务各 32 局，用 vcpp 的同一脚本）。
3. **重建 deploy3**：它还没被 qc 用，可以重建。把 deploy3 快进 / 检出到新的 zones HEAD，`scripts/build_pgo.sh bench`，`fprofile-generate` 计数为 0。门禁同前：4 个非雾存档逐帧 0 差异、叶摘要逐位相同（同一价值网口径）、吞吐在噪声内或更快。
4. 重新 dry run（L41 进课程），在"结果"写 `[A]` 一行：合并提交号、测试与门禁数字、deploy3 SHA、qc 重启命令。
