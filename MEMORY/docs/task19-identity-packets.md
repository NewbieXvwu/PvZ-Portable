# task19（主控 2026-10-21 下发给台式机 Codex）：按植物身份出卡 + 训练卡组扰动

先读仓库根目录 `AGENTS.md`（优先级最高，尤其 §8 卡组与地形），再读 `MEMORY/docs/student-bottleneck-plan.md` 末尾 10-21 的几条（加宽阶段、6 卡面板、选卡设计）。

## 背景
学生 `RolloutNet`（`python/pvz_rollout_policy.py`）的观测已经按植物种类编码（棋盘每种植物一个通道，标量里每种植物“在手 / 冷却”两格），
但**输出按卡槽**：`self.packet = nn.Linear(g, PACKETS)` 给 10 个卡槽各一个 logit，`self.packet_embedding = nn.Embedding(PACKETS, width)` 也按卡槽号。
所以“第 3 槽 = 樱桃”是学生隐式记住的，卡组一变就错位；实测同一个 v32 在 7 卡 [0,1,2,3,4,5,7] 下 L9 65%，6 卡 [0..5] 下只剩 2%
（`MEMORY/docs/day-panel.md`）。加宽阶段（更多关卡、按地形配卡组、以后学选卡、传送带关同一种卡可能有多张）都要求**按植物身份出卡**，
并且训练时就见过多种卡组。两件事都不是手写规则：一个是网络结构，一个是训练数据分布。

## A. 按植物身份出卡（网络 + 导出 + C++）
工作区（台式机）：`git -C ~/PvZ-search worktree add ~/PvZ-search/.claude/worktrees/identity -b identity origin/pvz-env`，只在这里改代码、单独编译
（`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_ROLLOUT_BLAS=ON && cmake --build build -j4`）。

1. `RolloutNet(..., packet_head="slot" | "identity")`，默认 `slot`（结果必须与现在逐位一致）。`identity` 时：
   - 一张可学的植物向量表 `nn.Embedding(PLANT_TYPES, E)`（`PLANT_TYPES` 见 `pvz_value_net.py`），按每个卡槽里的**植物种类**查表；
   - 每个卡槽的 logit = 小 MLP([g, 植物向量, 该槽动态状态（是否可用 / 冷却进度 / 买不买得起，已在观测里的就从观测取）])，空槽 / 不可用槽照现在的方式屏蔽；
   - 选格头的“所选卡”向量改用同一张植物向量表（不再按槽号）。
   - 可选（做得到就做，写清楚）：把植物向量也接到棋盘输入（第一层卷积对植物通道的权重本身就是一张表，看能否共享），不强求。
2. 训练脚本 `scripts/train_rollout_policy.py` 加 `--packet-head identity`，检查点记录它，`load_student` 读回。
3. 导出新版本 `PVZRN006`（头里加 packet_head 字段与植物向量维度），旧版本照常能读。
4. C++ CPU 前向（`src/EnvironmentRollout.cpp`）实现 identity 头；与 Python 对拍（仿 `python/test_cpp_rollout.py`，随机权重 + 真实局面，
   误差 < 1e-5、出卡与选格 argmax 全同）。CUDA（`cuda/`）只加明确报错 `Fail("identity packet head not supported on CUDA yet")`。
5. 单测：同一局面把卡组的卡槽顺序打乱（同一组植物、不同槽位），identity 头选中的**植物**不变（slot 头一般会变）。

## B. 训练卡组扰动（采集侧）
1. 任务清单支持“卡组扰动”（字段名自定，如 `deck_perturb: {"swaps": [1, 2], "pool": [...]}`）：每个种子按种子确定性地从基础卡组换掉 1–2 张，
   候选池 = 该关已解锁且该地形可用的植物（复用 `scripts/pvz_tasks.py` 的地形约束；升级植物按 AGENTS §8 声明所有权或直接排除）。
   卡槽顺序也按种子打乱。**实际用的卡组写进每局 `meta.json`**，重放（`episode_query.py whatif`）读存档里的卡组，不读清单。
2. 可行性：先只用 AGENTS §8 的地形覆盖规则；引擎选卡界面的警告（`SeedChooserScreen::OnStartButton` / `SeedNotRecommendedToPick`）
   能在无头模式下调用就接上（软约束：默认排除被警告的卡组），做不到写进报告。
3. 生成脚本：给 `artifacts/stage4/day_panel.json` 的 `day_l{7,8,9,10}_train_d7` 生成带扰动的新任务族 `day_l*_train_d7p`（DESIGN §14.7 登记，种子另起一段，
   如 1840000+ 起），以及评估用的“留出卡组”面板（每关几套训练里不会出现的卡组，种子 30000–30255）。
4. 冒烟：`search_play.py` 用扰动任务各跑 2 局（学生 v33：`~/PvZ-search/artifacts/student/v33/model.pt`），档案可被训练加载器读出，meta 里卡组正确。

## C. 验证（小规模，正式训练由主控排）
- 用现有数据（`--labels` 照 `~/pvz-logs/queue99.sh` 里的 dw3 那套）训一个 `--packet-head identity --row-pool` 的模型（1 个种子，GPU 空闲时再跑，
  先 `pgrep -f train_rollout_policy` 确认没有别人在训），评 7 卡面板 L7–L10 与 6 卡面板（任务 `day_l*_eval` 与 `day_l*_eval_d7`，每关 256 局）：
  目标是 7 卡不比 dw3s1（L7–L10 248 251 229 237）差太多。6 卡指望不大（还没有扰动数据），如实报告。

## 规则
- 台式机正在跑正式队列（tmux `q98` / `q99` 等）：**不要动任何 tmux 队列、不要改 `~/PvZ-search/.claude/worktrees/rowpool` 与 `~/PvZ-search` 主工作区**。
  同时最多 4 个游戏进程；GPU 被别人训练占用时不要训。
- 分支 `identity` 提交，可以 `git push origin identity`；不推 pvz-env、不合并。
- 不改默认值；不写手工规则；不写字面密钥（`$HF_TOKEN` / `$PVZ_HF_REPO`）；不 `git add -f`；`.pt/.npz/.log` 不进 git。
- 每完成一步提交一次；结果与问题追加到本文末尾“结果 / 问题”两节，按最佳判断继续，不要停下来等。额度用完就停在一个能续的提交上，并在“结果”里写清做到哪一步。

## 结果

- 2026-10-21，A1：在独立identity工作树实现可选identity头：共享植物embedding供逐槽MLP与选格头使用，逐槽输入植物ID/active/冷却进度/买得起，空槽及不可用槽屏蔽；重复植物的全局标量按max聚合、平分时按植物身份决策，避免顺序泄漏。checkpoint加载及PVZRN006导出已接入；6项Python检查通过（重排、重复卡、可用性、共享embedding梯度、往返及缺失输入），与基线7384b8d对24个slot配置的初始化、全部logit和导出逐字节相同。训练数据接线及C++尚待下一步，不将此视为A完成。

## 问题

- 可选的棋盘卷积与植物embedding共享暂不做：空间卷积是每植物×3×3核，直接共用将改变干线结构；当前只共享出卡/选格两头，不改变旧slot网络。没有改任何其他工作树或tmux队列。
