# task96：同配方同种子重训为什么不能复现 r11？

> 主控 2026-10-26 写。执行者：台式机 Codex（plus 号，tmux `codex-clean`；额度用完由 agent_pool 交 CodeBuddy）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/repro`（分支 `repro`，主控已从 origin/pvz-env 拉出）。每个任务最多 2 个游戏进程；**GPU 训练让 qc 先**；同一时间 GPU 上最多 1 个本任务的训练。

## 现象
task84 的变体 A：r11 的 348 个标签，配方与 r11 逐字段相同，seed 1，重训一次。结果 heldout 124/192，而 r11 是 166；对照 d7 是 172。task92（每轮训多个种子再挑选）的前提是“训练随机性是主要来源”。所以必须先分清：
- (a) **训练本身不确定**：cuDNN 非确定性算子、数据加载顺序、多进程预计算；
- (b) **标签内容变了**：同名标签目录在 r11 之后被追加、重标或回填，例如 `backfill_observation_v2`、`relabel_verify`；
- (c) **代码变了**：打包器、编码器或训练脚本在 r11 之后有改动，`encoder_signature` 能否发现？

## 要做的
1. **数据**：对比 r11 训练时的 `source_rows.json` / pack 指纹，与变体 A 打包时的同名标签，逐标签比较行数、内容哈希和 mtime。
2. **代码**：列出 r11 训练时的提交与 A 所用提交之间，`python/` `scripts/` 中训练链的差异。
3. **确定性**：在同一份 pack 上用同一 seed 训 2 次，各跑 1 个 epoch（≤ 15 分钟），比较权重哈希与验证损失。不同就找出非确定性的来源：
   - `torch.use_deterministic_algorithms` 打开后的报错点；
   - DataLoader 的 worker 与 shuffle 种子。
4. **结论与修法**：
   - 如果是 (b) 或 (c)，给出“冻结标签快照、pack 指纹进轮目录”的方案；
   - 如果是 (a)，评估确定性训练的成本（耗时比），并给出建议：是用确定性训练，还是接受随机性、交给 task92 去挑。
   不改生产。
5. 结论写 `MEMORY/docs/training-reproducibility.md`，`git push origin repro`。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式。
- 每完成一小步就提交、push，在“结果”里记一行；有疑问写进“问题”，按最佳判断继续。

## 结果

> 续接复核更正：旧记录把主检出930d776d当成r11运行代码的归因不成立；该版本没有recipe/identity头，无法产生r11现有产物。保留已完成的数据检查与双跑，准确运行修订仍需取证。
- 10-26 CodeBuddy：定位变体 A 产物在 gatepower 工作树 `artifacts/task84/variant_a/`。步骤 1 部分完成：recipe.json 与 r11 逐字段相同（仅 data/out 路径不同，348 标签同集合）；source_rows.json 逐标签行数完全一致（总 16,982,931 行）。但 metrics 显示同 seed 下两轮训练逐 epoch loss 均不同（epoch12：r11 train 3.7252/val 3.5927，A 3.7131/3.5853）——训练确实分叉。r11 训练时间 2026-10-25 08:30–09:12（pack 21 建 + 327 复用）。待办：内容哈希级对比两份 frames（~10.2 GB vs ~10.4 GB）、代码提交差异、确定性实验。
- 10-26 CodeBuddy：两份 recipe.resolved.json 逐字段一致（仅 data/out 路径不同），且两轮 pack 指纹同名 `7d15b915a2192805.npz`（16,982,931 行同）——但该 pack 在 r11 的 frames/packs 里已不存在（pack.log 显示其后有 prune 记录），gatepower 侧现存同名文件仅 81 KB（元数据级，非全量数据 pack），pack 指纹依据待查（大概率是标签行数/元数据而非内容哈希）；下一步需直接对两份 frames 的标签内容做哈希级对比。
- 10-26 CodeBuddy：**(b) 标签内容变更已排除**——pack `7d15b915a2192805` 引用的 348 个段在 r11 与 A 两份 frames 里 sha256 全部相同（~9.6G 实测，无一处差异）；且两轮 train.log 的数据统计逐项一致（rows/plants/seeds/val rows、board_bytes 等、packet/spread 权重）。另发现 pack/段文件名 = sha1(选项字典)（labels/wait_keep/obs/sources 等），不含内容哈希——同名 ≠ 同内容，"pack 指纹进轮目录"方案应改为记录内容 sha256。**(c) 代码已变**——r11 用主仓 `930d776d`（10-17），A 用 gatepower `bec2b9e6`（10-25）：训练器 `scripts/train_rollout_policy.py` 重写（933 行 diff），关键差异：看板数据新增 float16 存储（旧版 fp32）、episode 顺序 shuffle 种子从硬编码 `Random(0)` 改为 `Random(validation.seed)`、行采样/CSR 拷贝路径重写。正在做步骤 3：同 pack 同 seed 双跑 1-epoch（/tmp/exp96/detexp，代码取 bec2b9e6 干净副本），以分辨剩余分歧是否纯非确定（a）。

- 10-26 Codex 续接：已核实CodeBuddy双跑都完成，未重跑。配方除out外相同，同seed=1/epochs=1；21/21权重张量实际不同（并非仅.pt元数据差异），val_loss 3.7510763180 vs 3.7643170536。两次总耗时1224s/592s、epoch耗时890s/334s，存在资源竞争且未记录deterministic标志，不能当作确定性开关成本。临时代码7个训练链文件与bec2b9e6逐字节相同。原r11 recipe使用identity头与recipe入口，930d776d二者皆无，故撤回“r11用930d、训练器改写导致差异”的未证归因；实际训练循环没有DataLoader，CPU torch.randperm由torch.manual_seed(seed)驱动，precompute_workers不是DataLoader worker。证据task96-existing-runs.json；下一步查实际运行来源并做受限确定性诊断，GPU让qc先。

## 问题
- r11 trained.json仅记录模型/配方sha与时间，没有trainer运行revision；recipe.provenance是配方来源，不等于实际训练源码。准确r11源码修订尚未确定，不能据主检出HEAD归因代码漂移。
