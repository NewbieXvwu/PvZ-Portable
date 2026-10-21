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

- 2026-10-21，A2完成：训练器支持--packet-head identity、记录头类型/E维度，实际逐槽输入使用独立float32文件映射与按原训练行对齐的侧缓存；--read-cache-dir只读复用既有缓存，缺失项写本工作树，旧v1/v2lite/v2编码签名均与7384b8d相同。C++ CPU接PVZRN006（width/pool/context/obs/row_pool/packet_head/E），旧PVZRN002–005保持；两个CUDA入口均明确拒绝identity。11项新检查通过（含2epoch CPU实际训练及只读缓存）、24种identity/192个真实局面最大误差1.19e−7、全部argmax/动作相同；新C++顺序/平分检查通过。旧C++ 48配置回归通过，旧/新默认slot两epoch训练张量与验证loss逐位相同。A完成，B扰动及C完整dw3验证仍待完成。

- 2026-10-21，C前瞻登记：验证使用queue99 dw3的82标签、38,271份现有档案，与dw3s1检查点标签及各损失权重完全匹配；仅新增identity头（按行汇总、seed1、12epoch、cosine、早期倍率×1.5不变）。[固定清单](task19-validation-plan.json)登记全部8个固定卡组任务，各256局、共2048局，取最后计划检查点，逐关报告，不择种子或epoch；L10按固定卡组语义。训练需无其他train脚本且≥14GiB连续30秒后准入，低于2GiB暂停自有完整子树；父线程池的全部task/children也纳入。主数据和旧缓存只读，私有缓存/模型/日志在本工作树，B实现可并行推进，但不改冻结的训练源。

- 2026-10-21，B1完成：新DECK_OPTIONS协议直接读实际profile解锁列表与Board逐卡警告；确定性替换/洗牌、地形覆盖和全局无序卡组留出均接通，默认任务kwargs不变。生成4个d7p训练族（184/185/186/187万起，各8192种子）和每关3套留出；实际32768训练种子审计通过，3682个训练卡组集合与12个留出集合完全分离。地形源码解析只读缓存优化前后生成文件逐字节相同；6项扰动+9项既有回放检查通过。采集对新族记录实际deck/基础deck/完整任务/冻结学生以供whatif，8局v33正式参数采集冒烟已启动，尚待档案加载及实机前缀重放。

- 2026-10-21，A3/输入修订：8局采集证实L10仍有传送带移位补卡/免费种植，原“固定卡组”假设不成立；250个L10种植标签中147个在显示价格上买不起但实际合法。另SEED_NONE(-1)会让旧Python标量编码覆盖wave_count特征。identity改为原始头只屏蔽空槽，act/ranked/native均沿引擎合法集决定可用性；逐帧scene侧缓存与identity_scalars恢复正确公开波数，旧slot/旧缓存签名不改。30项目标检查通过，新增80真实传送带决策（空槽79/免费卡13）最大误差1.64e−7且argmax/动作全同；默认slot两epoch张量/loss继续逐位相同。
- C初次尝试按实际进度归档：停止时已完成3个epoch、可能有下一epoch部分更新，无保存模型、无任何正式identity评估；不是零更新。日志/资源/原计划完整留在本地before_conveyor_input_fix。错误输入导致loss异常，修正后仍按原82标签、38,271档案、seed1、12epoch及同2048局验证，复用已验证缓存，不择种子/epoch。[修订依据](task19-input-amendment.json)、[修订后实际计划](task19-validation-amended-plan.json)保留原计划SHA及新源SHA；样本行数11,051,782与dw3s1完全一致。

- 2026-10-21，B2完成：[8局冒烟证据](task19-perturbation-smoke.json)。L7–L10各2局，v33/V9/K2/B1/v3叶子与正式queue98采集参数一致，全部自然完成；各meta初始牌与种子确定性采样相同，各帧记录实际动态手牌。档案通过v2+cell目标+identity真实加载，共6192行；147个免费种植标签仍可学习（CPU128行CE=1.654且梯度有限）。实机8动作前缀回放使用存档deck/checkpoint通过；8局1胜7负，仅证明采集/归档/加载/回放管线，不作为卡组或新模型质量结论。C修正后训练仍在进行，完成后自动跑原定8×256局，代码/模型/清单/程序SHA在首局前冻结。

- 2026-10-21，C收束入口：新增scripts/summarize_identity_validation.py，只有完整训练与八组各256局的实际文件/种子/配置/胜局和冻结SHA均通过，才写task19-identity-results.json；同时核对dw3s1配方及同种子胜负差，保留基线6worker/另一构建与本次4worker的可比性限制。现有修订训练774795、控制774721及评估等待775575复用，没有启动第二个模型或调整科学计划。

- 2026-10-21，工程证据收束：[task19-engineering-validation.json](task19-engineering-validation.json)保留当前源码与程序SHA、30个唯一目标检查及旧48配置日志身份、24配置默认slot逐位/逐导出相等证明；独立重遍全部32768种子再次确认3682训练初始牌集合与12留出集合零重叠，并逐个核对地形/实际解锁/逐卡警告。未改冻结源，C完整胜局仍等待实际训练/评估完成。

## 问题


- 可选的棋盘卷积与植物embedding共享暂不做：空间卷积是每植物×3×3核，直接共用将改变干线结构；当前只共享出卡/选格两头，不改变旧slot网络。没有改任何其他工作树或tmux队列。
- 选卡警告范围：已接可无头调用的Board::SeedNotRecommendedForLevel原始逐卡标志；OnStartButton依赖mChosenSeeds、mCutScene与模态对话框，未调用其“整组无产阳光/缺防空”等组合警告。原始逐卡标志也未按咖啡豆组合取消夜间植物警告，日间pool因此保守排除菇类；这不是全部合法卡组的枚举。地形硬覆盖始终校验；解锁按任务真实playthrough=2的引擎列表，不用首轮奖励表猜。
- L10语义更正：RESET_V2只规定初始手牌，后续仍是原生传送带行为；初始卡组与原版空手开局也不同。C保留原任务及基线不改物理规则；B留出是初始植物集合，不保证后续动态手牌集合未见。meta.deck供reset重放，逐帧packets供真实身份训练，不用初始deck代替后续槽位身份。
- identity保持--packet-balance .5这个超参数，但按植物身份统计频率；L10动态换卡后实际行权重不等同旧slot频率。C是单种子工程试验，且identity修复了空槽公共输入，不能将结果当作只有输出层改变的严格因果消融。旧slot的SEED_NONE公共标量行为按兼容要求保留，未全局改动featurize_numpy。
