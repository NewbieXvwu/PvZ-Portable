# task18（主控 2026-10-21 下发给 Mac CodeBuddy / DeepSeek）：采集 worker 内存瘦身

先读仓库根目录 `AGENTS.md`（优先级最高），再读本文。

## 背景
台式机 WSL 只有 24 GB（宿主 32 GB），训练（约 11 GB）与采集同时跑时可用内存掉到 2 GB 以下、swap 满，已经两次差点 OOM。
实测（2026-10-21，台式机，`scripts/search_play.py` 16 进程采集）：**每个 worker 的 Python 进程约 650 MB**（`python3 -c from ...`），
外加每个 `pvz-portable` 约 210 MB；16 个 worker 合计约 14 GB——采集才是最大的内存用户，比训练还多。
目标：在**结果逐位不变**的前提下，把每个 worker 的 Python 常驻内存降到 250 MB 以下（16 个 worker 省出约 6 GB）。

## 要做的
工作区：`git -C /Users/newbiexvwu/PvZAgent worktree add .claude/worktrees/slimworker -b slimworker origin/pvz-env`，只在这里改代码、单独编译
（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`）。
主仓库工作区与 `build/` 不许碰（Mac 正式采集在用）。**同时最多 4 个游戏进程**。不要 `cd` 进别的工作树（用绝对路径 / `git -C`）。

1. **先测量**：用正式采集的参数（`artifacts/hard_mac/round15_flags` + `--student-candidates 2 --dagger-beta 0`，学生 v33（HF `transfer/v33_model.pt`，下到工作树的 `artifacts/student/v33/model.pt`；`$HF_TOKEN` / `$PVZ_HF_REPO` 在 `~/.zshrc`），
   清单 `artifacts/stage4/day_panel.json`，任务 `day_l9_train_d7`，`--seed-offset 60000 --seeds 8 --workers 2`，`--frames` 指向工作树里的临时目录）跑一次，
   记录每个 worker 进程的 RSS 随时间的峰值，并弄清 650 MB 由什么组成（torch 导入、学生 / 价值网络 / 静态价值网络各自、numpy 缓冲、每局档案缓存……）。
   可以用 `tracemalloc`、`psutil`、逐步注释导入的方式，结论写进报告。
2. **再瘦身**：常见手段（按实测挑，不要猜）：worker 里不需要的模块不导入；模型只在用到的进程里加载；用 `fork` 让只读权重在父子进程间共享页；
   局内累积的大数组及时释放；`torch.set_num_threads(1)`（如果还没设）。**不改搜索 / 标签 / 采集的任何逻辑与默认值**。
3. **逐位验证**：改前改后同一组种子（上面 8 局）的 `games.jsonl` 与每局 `frames.npz` / `meta.json` 内容必须完全一致（写个比对脚本，忽略纯计时字段）。
4. 跑现有测试 `python3 -m pytest python -q`，全部通过（Mac 上 `test_dagger_experiment_runner` 的内存暂停测试原本就因平台失败，注明即可）。

## 交付
- 分支 `slimworker` 本地提交，可以 `git push origin slimworker`；不推 pvz-env、不合并。
- 报告追加到本文末尾“结果”一节：650 MB 的组成、改了什么、改前改后每 worker RSS 峰值、逐位比对结果、测试数。
- 有疑问写在本文末尾“问题”一节，按最佳判断继续，不要停下来等。

## 不要做的
- 不改默认值、不改搜索与标签逻辑、不训模型、不写手工规则。
- 不写字面密钥；不 `git add -f`；`.pt/.npz/.log` 不进 git。
