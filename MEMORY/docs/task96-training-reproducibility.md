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

> 续接复核更正：旧记录把主检出930d776d当成r11运行代码的归因不成立；该版本没有recipe/identity头，无法产生r11现有产物。保留已完成的数据检查与双跑，部署记录现已定位4f4d217b（评估源码3个SHA交叉确认），它与A的训练器完全相同；逐次训练源码回执仍缺失。
- 10-26 CodeBuddy：定位变体 A 产物在 gatepower 工作树 `artifacts/task84/variant_a/`。步骤 1 部分完成：recipe.json 与 r11 逐字段相同（仅 data/out 路径不同，348 标签同集合）；source_rows.json 逐标签行数完全一致（总 16,982,931 行）。但 metrics 显示同 seed 下两轮训练逐 epoch loss 均不同（epoch12：r11 train 3.7252/val 3.5927，A 3.7131/3.5853）——训练确实分叉。r11 训练时间 2026-10-25 08:30–09:12（pack 21 建 + 327 复用）。待办：内容哈希级对比两份 frames（~10.2 GB vs ~10.4 GB）、代码提交差异、确定性实验。
- 10-26 CodeBuddy：两份 recipe.resolved.json 逐字段一致（仅 data/out 路径不同），且两轮 pack 指纹同名 `7d15b915a2192805.npz`（16,982,931 行同）——但该 pack 在 r11 的 frames/packs 里已不存在（pack.log 显示其后有 prune 记录），gatepower 侧现存同名文件仅 81 KB（元数据级，非全量数据 pack），pack 指纹依据待查（大概率是标签行数/元数据而非内容哈希）；下一步需直接对两份 frames 的标签内容做哈希级对比。
- 10-26 CodeBuddy：**(b) 标签内容变更已排除**——pack `7d15b915a2192805` 引用的 348 个段在 r11 与 A 两份 frames 里 sha256 全部相同（~9.6G 实测，无一处差异）；且两轮 train.log 的数据统计逐项一致（rows/plants/seeds/val rows、board_bytes 等、packet/spread 权重）。另发现 pack/段文件名 = sha1(选项字典)（labels/wait_keep/obs/sources 等），不含内容哈希——同名 ≠ 同内容，"pack 指纹进轮目录"方案应改为记录内容 sha256。**(c) 代码已变**——r11 用主仓 `930d776d`（10-17），A 用 gatepower `bec2b9e6`（10-25）：训练器 `scripts/train_rollout_policy.py` 重写（933 行 diff），关键差异：看板数据新增 float16 存储（旧版 fp32）、episode 顺序 shuffle 种子从硬编码 `Random(0)` 改为 `Random(validation.seed)`、行采样/CSR 拷贝路径重写。正在做步骤 3：同 pack 同 seed 双跑 1-epoch（/tmp/exp96/detexp，代码取 bec2b9e6 干净副本），以分辨剩余分歧是否纯非确定（a）。

- 10-26 Codex 续接：已核实CodeBuddy双跑都完成，未重跑。配方除out外相同，同seed=1/epochs=1；21/21权重张量实际不同（并非仅.pt元数据差异），val_loss 3.7510763180 vs 3.7643170536。两次总耗时1224s/592s、epoch耗时890s/334s，存在资源竞争且未记录deterministic标志，不能当作确定性开关成本。临时代码7个训练链文件与bec2b9e6逐字节相同。原r11 recipe使用identity头与recipe入口，930d776d二者皆无，故撤回“r11用930d、训练器改写导致差异”的未证归因；实际训练循环没有DataLoader，CPU torch.randperm由torch.manual_seed(seed)驱动，precompute_workers不是DataLoader worker。证据task96-existing-runs.json；下一步查实际运行来源并做受限确定性诊断，GPU让qc先。

- 10-26 Codex：metadata补查348段均存在、大小相同、独立inode，r11段mtime全部早于训练；复用CodeBuddy的348段内容SHA结论，不重复9.6GB哈希。qc_setup.log显示10-24 19:47部署4f4d217b，直到r11训练完成后10-25 10:48才deploy12；r11评估的search_play/pvz_promotion/promotion_gate三份源码SHA全部与4f4d217b一致。4f4d217b与A bec2b9e6的train_rollout_policy/pvz_recipe/pvz_training_pack/pvz_value_net字节完全相同，RolloutNet/特征函数AST相同；其余训练链差异见task96-code-provenance.json，不能再把trainer重写当原因。CPU审计脚本task96_provenance.py已保存。另准备单个真实2368行段的受限诊断（最多5个顺序子进程、每个150s），只写repro artifacts，当前等待qc GPU空闲；用它查确定性报错/初始权重/洗牌，不冒充全量数据重训或生产成本测量。

- 10-26 Codex：348段元数据已逐项关联回标签与source_rows计数，两份source_rows原文件SHA同为fd0f9c12…；初版training-reproducibility.md整理已完成数据/代码/双跑结论与冻结方案，明确GPU诊断尚在等qc，不把旧双跑耗时当确定性成本。

- 10-26 CodeBuddy 续接：修好诊断脚本三处无法启动的问题（labels 传字符串、单文件 read_pack 丢 packet store、label_weight 含集外键），CPU 冒烟通过。GPU 9 个用例跑完：初始权重与 randperm 摘要 9/9 相同，未设 workspace 与 `:4096:8` 7 次逐位相同（含 deterministic 关闭），**只把 workspace 换成 `:16:8` 就从第 2 步起分叉**（最终张量 77a3093b vs 282009e3）。即同配置内无 run-to-run 随机性、跨配置才分叉；r11/A/run1/run2 的 epoch1 差 0.1%–1.2% 与“第 2 步级别扰动被 132k 步放大”一致。8-epoch 计时 4 次最终张量全同，strict 10.44/9.30s vs default 7.74/17.05s，qc 并发下测不出确定性开销。结论已写入 `MEMORY/docs/training-reproducibility.md`（含 task92 前提需改写）。未证实的一环：具体哪个环境差异让三次全量跑落到不同算法（cuBLAS 显存不足回退是最可能触发点，但没为复现去压 qc 的显存）。

- 10-26 CodeBuddy 交还 Codex：task96 五步全部完成，`20fe2cdb` 已 push origin/repro，工作区干净、生产未改。交付物 `MEMORY/docs/training-reproducibility.md`（9 个 GPU 用例 + 4 个 8-epoch 计时用例 + 数据/代码/双跑审计 + 6 条修法），证据 `task96-determinism-probe.json` 等四份。唯一遗留写进“问题”：四次全量 run 落到不同 cuBLAS 算法的具体环境差异未证实（最可能：workspace 申请失败回退，run1 890s/epoch 佐证抢 GPU），闭环需固定 `CUBLAS_WORKSPACE_CONFIG` 后重跑一个完整 epoch，qc 优先。

- 10-26 Codex再续接：已读CodeBuddy全部新增提交，不重跑9个小段用例。收紧报告中“历史落差已归因workspace”“同配置全量无随机性”“无报错等于无回退”的超证据结论；纠正12epoch步数为89,532而非132k。短样本计时受并发干扰，成本项尚未闭环。新增完整348段的strict/default/strict控制（固定workspace、每次≤900秒，源数据只读、写入全在repro、proc训练进程准入与让行），等待现有deckvar训练结束后补全量结果。

- 10-26 Codex按主控最新澄清：每任务游戏进程上限，不套全机限制。收回全量控制额外的“等待其他任务训练”准入，只保留qc训练优先、本任务GPU互斥、内存/显存余量；其他任务仅作为计时负载记录，不控制、不阻塞。现有队列尚未启动子训练，核实无子进程后只重启本任务等待器以加载新准入策略，不重跑任何已完成实验。

- 10-26 CodeBuddy 续接（第三段）：把中尺度控制扩到 9 例，补交替对照 default_p1 → strict_q1 → default_p2 → strict_q2（同数据同 seed）。结果更强：固定 `:4096:8` 的 5 例（含压 4 GiB 显存的、含交替重复的）最终张量与逐步 loss 全同；未固定的 4 例两两不同，其中 default_p1/default_p2 同一进程序列、报告空闲显存相同（11793 MiB）仍不同。→ 结论从“显存少会改数”改为“未固定 workspace 时结果依赖运行时状态、每次可能不同，固定后消失”。成本交替对照 strict/default = 1.065，但 default 自身两遍差 1.55 倍，测不出准确比值。证据 `task96-scale-probe.json`（9 例 + cost_pairs），报告已更新。
- 10-26 CodeBuddy 续接（第二段）：复核 Codex 的全量控制并执行。`strict_a` 17:03 启动，读完 348 段后稳定 0.099 步/秒、700 s 内 read_bytes 542 GB（page cache 装不下 9.6 GB spill），折算约 23 h/epoch，900 s 上限触发，未写 trace；证据留在 `artifacts/task96/full_pack/strict_a*`，未重跑。改用同一 348 段的固定子集（前 30 段 / 1,046,710 行，461 步，冻结 bec2b9e6 训练器、batch2048、seed1、epoch1）跑 5 个用例：strict 重复两次逐位相同（`78cfef547345…`，loss 序列 SHA 也相同）；把空闲显存从 9229 压到 4871 MiB 后，未固定 workspace 的 default 两例分叉（`1cf71d20337b…` vs `05b226f28628…`，val 4.7089 vs 4.6690），固定 `:4096:8` 的 strict 在同样压力下仍逐位不变。→ “环境漂移改数”第一次在真实训练里复现，且被固定 workspace 消掉；全量逐位复现仍未被证明（尺度差 16 倍），历史四次 run 的具体显存状态仍无回执。成本：五例 27.66/28.60/22.03/19.69/19.12 s，受顺序与缓存预热主导，测不出确定性开销。报告已更新 `MEMORY/docs/training-reproducibility.md`，新证据 `task96-scale-probe.json`，脚本 `scripts/task96_scale_probe.py`。

## 问题
- 具体是哪个环境差异把 r11 / A / run1 / run2 推到不同 cuBLAS 算法，仍无定论：三次全量跑都没留环境回执，只能从 epoch 耗时（221/224/890/334s）推断 run1 当时在抢 GPU。没有记录workspace申请失败或具体内核，耗时也不能单独证明GPU竞争；资源/算法变化保留为假设，不能据此断言历史原因。
- 诊断只用一个 2368 行段：它能证明“同配置逐位复现、换 workspace 就分叉”，但不能证明 1698 万行上不存在其它规模相关的非确定算子；严格说全量结论仍是“与观测到的分叉量级一致”。
- 生产没有改：所有新写入都在 repro 的 `artifacts/task96/`（ignored）与 `MEMORY/docs/`；`/tmp/exp96` 只是冻结只读快照。
- r11 trained.json仅记录模型/配方sha与时间，没有trainer运行revision；recipe.provenance是配方来源，不等于实际训练源码。已用部署日志和评估SHA定位当时部署版本4f4d217b，但未记录的训练时单文件覆盖无法靠事后Git证明不存在；不能据主检出HEAD归因代码漂移。
- 全量 16,982,931 行的逐位复现仍未证明：中尺度只到 1,046,710 行 / 461 步。全量控制两次尝试（17:03 完整用例、18:20 的 200 步速率探针）都在 9.6 GB spill 进不了 page cache 后退化：10 s/步 / 18 分钟读盘 650 GB，需要 `MemAvailable ≥ 14 GiB` 且无其他 10 GiB 级训练进程才能重试。这是环境限制，不是“全量不可复现”的结论；另因本机 earlyoom `--prefer` 含 python3，继续压内存有杀掉其他任务训练的风险，故主动停止。
- “空闲显存 → 数值”的因果链只观测到两端（压 4 GiB 显存前后张量不同、固定 workspace 后相同），没有采到 cuBLASLt 选核或 workspace 申请失败的 API 回执，机制仍是推断。
