# task41：把已收尾的 dw5（task27）与 pool（task22）两个分支合进主线

> 主控 2026-10-23 09:55（台式机时间）写。执行者：Mac 上的 Qoder（DeepSeek-Flash）。
> 工作树 `/Users/newbiexvwu/PvZAgent/.claude/worktrees/merge41`（分支 `merge41`，从 `origin/pvz-env` c378084f 拉出，主控已建好）。
> 只做合并、解冲突和测试，**不推 pvz-env**；推 `merge41`，主控审完再快进。

## 为什么
两个分支都已收尾，却一直没合进主线（`python3 scripts/loose_ends.py` 列出来的线头）：
- `origin/dw5`（47 个提交，task27：dw5i 训练、`planted_types` / `planted_slots`、面板、加载提速等）；
- `origin/pool`（10 个提交，task22：按区面板生成器 `scripts/zone_panel.py`、`scripts/make_pool_panel.py`、泳池族 p9k 的登记与报告）。
主线在这期间合了 audit（整条 CUDA 线退役，tag `retire-cuda-20261023`；判死开关删除，tag `retire-flags-20261023`）、keep（task31）、oom（task34）。

## 步骤
1. `git -C /Users/newbiexvwu/PvZAgent/.claude/worktrees/merge41 merge --no-ff origin/pool`，解冲突，跑测试（见第 3 步），提交。
2. 再 `merge --no-ff origin/dw5`，解冲突，跑测试，提交。
3. 测试：`cd /Users/newbiexvwu/PvZAgent/.claude/worktrees/merge41 && python3 -m pytest python -q -p no:cacheprovider`。主线现状是 253 过 / 1 个环境性失败（`test_dagger_experiment_runner.py::...only_own_parent_and_child`，Mac 进程树问题，与合并无关）。合并后不得多出失败；需要引擎的测试会因工作树没有 `build/` 而跳过——**用主工作区的引擎跑一遍需要引擎的测试**：先 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j4`（在 merge41 工作树里，单独编，不碰主工作区的 build），再跑全套。
4. `git push origin merge41`，在本文“结果”写：每个冲突怎么解的（一行一个）、测试数字。

## 解冲突的规则（按这个判，不要猜）
- **CUDA 相关**（`scripts/cuda_*.py`、`cuda/`、`--rollout-backend`、`--cuda-*`、`_cuda_continuations`、`env.cuda_dump`）：主线已删，**一律以删除为准**；dw5 对这些文件的修改丢弃（modify/delete 冲突选删除）。
- **判死开关**（`--ema`、`--soft-weight`、`--candidate-generator`、`--candidate-coverage-*`、`--shovel-candidates`、`--cpu-sparse-tracks`、`--measure-cpu`、`--cpp-rollouts`、`--frames-format` 写入分支、`--margin`/`--max-actions`）：主线已删，以删除为准；如果 dw5 的新代码引用了它们，改成主线的常量/唯一实现，不要把开关加回来。
- **dw5 的功能改动保留**：`planted_types` / `planted_slots`（games.jsonl 按植物种类记种植）、面板与加载提速等 task27 的成果，要在主线的新结构上重新落好。`scripts/search_play.py`、`scripts/unified_panel.py`、`scripts/sim_skip_agreement.py` 的冲突按“主线结构 + dw5 新功能”合。
- **`scripts/make_night_panel.py`**（pool 分支与主线冲突）：主线的版本含 task28/31 的扰动族与 keep 逻辑，保留主线；pool 分支对它的改动若是独立功能就移植进来，若只是旧版重构就丢弃，在“结果”里写明是哪种。
- **DESIGN.md**：两边都加了 §14.x 小节。主线现有编号到 §14.14。pool 分支的泳池族登记、dw5 的条目若与主线已有小节重复（主线 §14.10 是统一面板），合成一节；新增的接在 §14.14 之后顺延编号，并修正文中所有“§14.x”引用。
- **`MEMORY/2026-10-22.md`、`MEMORY/2026-10-23.md`**（只追加的日志）：两边的条目都保留，按时间排序。
- **`MEMORY/docs/task27-dw5i.md`**：主线有主控追加的段落，分支有执行结果，两边都保留。

## 规则
- 只用绝对路径，不 `cd` 进别的工作树（merge41 自己除外）。`PVZ_NO_POPUP=1`。
- 不跑采集、不跑训练；Mac 上另有采集在跑（`mac_dagger_p5a.sh`），编译用 `-j4`。
- 禁止 `git add -f`；不提交 `.pt` `.npz` `.log`；不写字面密钥。

## 结果

- 步骤1（pool）：合并 1a99cff9。冲突三处：`make_night_panel.py` 保留主线（含 task25 n10 族与 L20 原版开局），pool 的 zone_panel 薄包装重构整份丢弃（属旧版重构，会丢后续功能）；`DESIGN.md` 主线 §14.10–14.14 不动，pool 泳池族登记改为 §14.15，`MEMORY/docs/pool-panel.md`、`task22-pool-panel.md` 里的 §14.10 引用一并改 §14.15；`MEMORY/2026-10-22.md` 两侧条目全保留，task22 两条按时间插在 task25 之后。测试（无 build/）：245 passed / 6 failed（5 个 test_cpp_rollout 因缺引擎、1 个 test_dagger_experiment_runner 环境性）/ 26 skipped，与合并前基线逐项相同。

## 问题
