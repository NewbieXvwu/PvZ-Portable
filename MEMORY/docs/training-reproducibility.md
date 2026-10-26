# task96：r11 重训复现审计

2026-10-26，repro 分支。生产代码、qc、门禁、其他工作树均未修改。小样本（9 个用例 + 4 个 8-epoch 计时）与中尺度（30 段 / 1,046,710 行，5 个用例）两轮诊断给出的结论是：

- **同一份数据、同一份代码、同一 seed，在固定运行配置下逐位复现**（小样本 7 次、中尺度 2 次，张量与 loss 全同）。
- **不固定 cuBLAS workspace 时，同样的训练每次结果都不同**：中尺度 4 次未固定配置的最终张量两两不同（含两次报告空闲显存相同、同一进程序列的重复）；而固定 `:4096:8` 的 5 次（含压显存 4 GiB 的两次）**逐位相同**。这是首次在真实训练里直接观测到“运行时状态 → 结果漂移”，且它只在未固定 workspace 时存在。
- 完整 348 段（16,982,931 行）的固定 workspace 控制在**当前主机内存状态下不可行**：9.6 GB spill 进不了 page cache，实测退化到 10 s/步（约 23 h/epoch），不是确定性带来的开销。因此全量逐位复现仍**未被证明**，只能写到“1.05M 行尺度已证明，与历史 221 s/epoch 的量级差一致”。

历史 r11/A 落差的归因仍然**不成立**到“具体某次 run 的空闲显存是多少”这一层：四次全量 run 都没留环境回执。但现在有可复现的机制，不再是纯假设。

## 已有证据支持什么

- **已有同seed全量训练分叉，具体来源尚未闭环。** CodeBuddy 已完成的同数据、同配方、seed=1、1 epoch 双跑，21/21 个实际权重张量不同，val loss 为 3.7510763180 / 3.7643170536。不能用 `.pt` 文件 SHA 不同代替张量比较；本次两者都核验了。受限诊断的9次初始化与排列相同；未设workspace/4096组7次最终张量相同，16组2次彼此相同但与前组不同。它定位一个可行的分叉触发条件，没有覆盖全量训练全部路径。
- **现存输入没有发现标签漂移。** r11 与 A 的 348 标签逐项行数、训练/验证分配一致；CodeBuddy 已比较 348 个引用段 SHA256，全部相同。追加元数据审计显示段大小全同、不是硬链接，r11 的全部段 mtime 均早于训练。
- **“r11 使用 930d776d，训练器重写导致落差”是错误归因。** 该旧版本没有 r11 已使用的 recipe/identity 功能。部署日志定位当时部署版本为 **4f4d217b**，与 A 的 **bec2b9e6** 相比，训练器源码、配方解析、pack 指纹算法、value encoder 完全相同；网络类 AST 也相同。
- 仍不能把 heldout 的 **166→124 / 192** 全部归因于训练随机性，更不能从两次 1 epoch loss 推算胜率方差。r11 的逐次运行源码/环境回执缺失；task84 的多种子完整训练及配对评估才负责量化泛化波动。

## 数据与冻结证据

两轮总计 16,982,931 行，6,605,039 株，55,773 个 seed，验证 1,704,229 行。两份 `source_rows.json` 的文件 SHA256 都是：

`fd0f9c12901b4e4c7015cdf40515b3167c6ba39c791d86145e441d93f1500039`

名义 corpus index 为 `7d15b915a2192805.npz`。r11 侧 index 已被正常 prune，A 侧是 81KB 的段索引，不是损坏或缺数据。A 索引 SHA256：

`1351618d6c10459e9d30b5a2aa70a3a73ddbb05db239fc01bab75a5b6b02b4cb`

逐标签段名、archives/rows/training_rows/validation_rows、大小、双方纳秒 mtime、是否独立 inode 见 [task96-data-metadata.json](task96-data-metadata.json)。348 段内容 SHA 相等沿用 CodeBuddy 在 `9ffc611a` 已完成的 ~9.6GB 检查，本次没有重复全量哈希；该次未提交逐段摘要清单，这是证据保存限制。现存文件与较早 mtime 支持没有漂移，但不能补造训练启动时的不可变内容快照。

`archive_fingerprint` 实际哈希相对路径、文件大小及可选 `training_rows`；故意不含 mtime，也不含文件内容 SHA。pack 名再由 sources 与 labels/wait_keep/obs/cells/aux/packet_head/context/label_weight 等选项导出。**同名只能证明选项/路径/大小摘要相同，不能证明内容相同**；等长重标可能逃过它。

## 代码来源与差异

`qc_setup.log` 记录 10-24 19:47 部署 4f4d217b，21:35 stage4 恢复后重启；下一次 deploy12 是 10-25 10:48。r11 训练在 10-25 08:30–09:12，其后评估回执中的 `search_play.py`、`pvz_promotion.py`、`promotion_gate.py` 三个源码 SHA 都与 4f4d217b 一致。CQ 分支 HEAD 没有因按路径 checkout 而移动，所以只看 cq reflog 会误判。

| 训练链文件：4f4d217b → bec2b9e6 | 实际差异与本次相关性 |
| --- | --- |
| `scripts/train_rollout_policy.py` | 完全相同，SHA `5cc2f20803dc3f525640c76194d0ab6d70b51419d32352719e3dc09c93bc5b4c` |
| `python/pvz_recipe.py`、`pvz_training_pack.py`、`pvz_value_net.py` | 完全相同 |
| `scripts/pack_training_rows.py` | 普通 seed 解析改为 `archive_seed`，兼容 `seed_<demo>_f<future>`；本次直接读已生成且相同的段 |
| `python/pvz_frame_store.py` | 新增 archive_seed、v2size 注册、冻结 fog 签名；encode_frames/encode_student_frames/target_values AST 相同 |
| `python/pvz_rollout_policy.py` | 只改 v2size 原生导出版本号；RolloutNet/features AST 相同 |
| `python/pvz_student_observation.py` | 新增 v2size；raw zombie 三字段读取兼容追加字段；相同已编码段不重编码 |
| `python/pvz_training_rows.py` | SourceRowCounts 使用 archive_seed；完整 diff 见证据，无优化器改动 |

[task96-code-provenance.json](task96-code-provenance.json) 保存部署日志关键行、完整训练链 diff、源码 SHA 与评估交叉核验结果。部署记录比主检出 HEAD 更可靠，但没有训练启动时的逐文件 receipt，无法绝对排除未记录的临时覆盖。此处明确区分已记录部署版本与逐次实际运行源码。

`encoder_signature` 只覆盖 features/featurize_numpy/_col_of/target_values/encode_frames 以及选定数值常量；v2 又有冻结语义签名。它不是训练链签名，不覆盖 optimizer、训练循环、所有 merge/read 路径、网络实现或运行库。即使未来 (c) 发生，它也不保证发现。

## 已完成的全量双跑

复用 CodeBuddy 的两个产物，没有重跑 1698 万行。七个训练链源码文件逐字节匹配 bec2b9e6。两次除输出路径外的配方相同。

| 项 | run1 | run2 |
| --- | ---: | ---: |
| seed / epoch | 1 / 1 | 1 / 1 |
| train loss | 4.2911643558 | 4.3041004138 |
| val loss | 3.7510763180 | 3.7643170536 |
| epoch 秒 | 890.387 | 334.052 |
| 包含读取的总秒 | 1224.317 | 591.676 |

规范张量 SHA（按名称、dtype、shape、实际数值序列化）：

- run1 `5d52761abc8b3240506771a252beee316312ab7d066500ce266db5392bc5722e`
- run2 `148fa6bef2eef889a98a0ae62f3fbdf3d4e441c60d8f16db4750c4c0f8d46f6e`

逐张量差异与原始配方/metrics 见 [task96-existing-runs.json](task96-existing-runs.json)。第一轮 epoch 在 15 分钟内，但含加载共 20.4 分钟，超过任务要求的总时限；没有为了合规而重复实验。两轮未捕获 deterministic 标志及启动环境，资源竞争不同，**1224/592 或 890/334 不是确定性训练成本**。

训练没有 DataLoader。`precompute_workers=8` 指预计算，并非训练 DataLoader worker；读现成 pack 时不是训练洗牌来源。入口先 `torch.manual_seed(seed)`，训练每 epoch 用 CPU `torch.randperm`；验证由排好序的 seed 集与私有 `random.Random(validation.seed)` 划分。受限诊断已记录初始参数和实际 permutation SHA，检验“同 seed”是否真的给出同初始化与同顺序。

## 受限 GPU 算子诊断（已跑完）

脚本 [task96_determinism_probe.py](../../scripts/task96_determinism_probe.py) 使用上述冻结训练器和一个真实标签段：`n20o1_fast_search`，2368 行、32 seeds，段 SHA `806fbdfce4ffa8003eb9a84433a6b911edd3c060ea6cd388864863de23527854`。输入复制并验证到 repro 自己的 ignored artifacts，加载器只替换成读这个段；其余调用实际 trainer.main，batch=128。该实验用来定位算子/种子，不能替代全量训练或泛化测量。

11:59 的首版脚本跑不起来（`labels` 传了字符串而非列表；`read_pack` 单文件分支丢掉 identity head 需要的 packet store；`label_weight` 含新标签集之外的键），已改成 `labels=[label]`、`read_segments([segment])`、只保留本标签的权重，并先用 CPU 冒烟跑通（同一 seed 两次逐位相同）。

正式用例（独立子进程，每次 import torch 前设好 workspace；strict 另开 deterministic algorithms、cuDNN deterministic、benchmark=False）：

| 用例 | workspace | 初始张量 SHA | 最终张量 SHA | 第1步 loss | val loss |
| --- | --- | --- | --- | ---: | ---: |
| strict_no_workspace | 未设 | `2624c16674ce…` | `77a3093bb40d…` | 7.1692876816 | 6.5724385253 |
| default1 | :4096:8 | 同上 | 同上 | 同上 | 同上 |
| strict1 | :4096:8 | 同上 | 同上 | 同上 | 同上 |
| strict2 | :4096:8 | 同上 | 同上 | 同上 | 同上 |
| default2 | :4096:8 | 同上 | 同上 | 同上 | 同上 |
| ws16_a | :16:8 | 同上 | `282009e3e01a…` | 同上 | 6.5716974401 |
| ws16_b | :16:8 | 同上 | `282009e3e01a…` | 同上 | 6.5716974401 |
| nows_a | 未设 | 同上 | `77a3093bb40d…` | 同上 | 6.5724385253 |
| nows_b | 未设 | 同上 | `77a3093bb40d…` | 同上 | 6.5724385253 |

读法：

- **初始权重与 shuffle 顺序在所有 9 次里完全相同**（初始化张量 SHA 一致、`randperm` 摘要 `391b1dcc…` 一致）。`torch.manual_seed(seed)` 确实同时决定了初始化和训练排列，不存在“同 seed 不同初始化”的问题。
- **未设 workspace 与 `:4096:8` 逐位相同**（7 次全部一致，含 deterministic 关掉的 default 两次）。strict 打开后没有抛任何 nondeterministic 算子错误，只说明该样本实际经过的路径未触发确定性错误，不能证明内核未切换、没有内部回退，也不能替代全量检查。
- **只把 workspace 从 `:4096:8` 换成 `:16:8`，最终权重和 loss 就变了**：第 1 步 loss 仍相同，**第 2 步开始出现差异**（6.843989372253418 vs 6.84398889541626），17 步后训练 loss 6.6203269 vs 6.6201138、val 6.5724385 vs 6.5716974。同一 workspace 下重复（ws16_a/ws16_b）逐位相同。
- 本段观察到**组内复现、workspace组间分叉**。这与GEMM算法/浮点归约路径改变相容；尚未采集内核轨迹，不能把具体内核选择与后续放大量写成已证事实。
- 环境：torch 2.14.0+cu132、CUDA 13.2、cuDNN 92400、RTX 5080、driver 617.14、`float32_matmul_precision=highest`（TF32 关闭）。当前包安装元数据早于r11/A，但没有两次历史运行环境回执，不能只凭mtime证明期间未发生环境变化。

### 与全量结果对照

同一 pack、同一 seed、同一 trainer 的四个全量 run（r11、A，以及 CodeBuddy 的两遍 1-epoch），第 1 个 epoch 就已经互不相同：

| 全量 run | epoch1 train | epoch1 val | epoch 秒 |
| --- | ---: | ---: | ---: |
| r11 | 4.3446921 | 3.8311677 | 221 |
| A | 4.3403805 | 3.8103482 | 224 |
| CodeBuddy run1 | 4.2911644 | 3.7510763 | 890 |
| CodeBuddy run2 | 4.3041004 | 3.7643171 | 334 |

workspace实验提供了一个微小数值扰动的实例；尚未用完整训练证明它足以产生这四次全量run的loss或胜率差。全量每epoch为7461个优化步（15,278,702训练行，batch2048），12epoch共89,532步；旧文“132k步”计算无依据，已纠正。不能把差异量级相容当作历史归因。

**未能证实的一环（本次已收窄）**：具体是哪个环境差异让这四个 run 落到不同算法。中尺度控制（见下节）已把“未固定 workspace → 每次结果不同；固定后 → 逐位相同”做成可复现观测，形状与历史四次 run 一致；仍未证实的是这四次 run 当时的具体运行时状态——它们都没留环境回执，事后无法补。run1 的 890s/epoch 只能说明耗时不同，不能区分 GPU 竞争、CPU/I/O/缺页或其他原因。

### 确定性训练的成本

同一段数据跑 8 个 epoch（136 步）四遍：strict 两遍 10.44s / 9.30s，default 两遍 7.74s / 17.05s，**四遍最终张量 SHA 完全相同**（`d115b228…`）。qc 同时在跑，计时噪声（7.7–17.0s）远大于任何 strict 开销，**这个尺度上测不出确定性模式的代价**。这既不能说明生产上零成本，也不能说明有成本；真要上生产，应在完整一个 epoch 上量一次。

本次短样本无法给出可靠的确定性成本比。“未报错”不证明没有算法切换或回退，不能据此预告零成本或小开销。

中尺度控制补了一组**交替**对照（default_p1 → strict_q1 → default_p2 → strict_q2，同数据同 seed，让缓存预热与负载漂移在两种配置间对消）：default 13.51 / 20.93 s（均值 17.22），strict 18.70 / 17.98 s（均值 18.34），**比值 1.065**。但 default 自身两遍就差 1.55 倍（13.5 vs 20.9 s），噪声远大于 6.5%，所以只能说“**没有观测到显著开销，也测不出准确比值**”，不能写“确定性训练零成本”。要给出生产建议，需要在 `MemAvailable ≥ 14 GiB` 且无其他训练时量一个完整 epoch 的 strict/default 交替对照。

运行注意：WSL 下 `nvidia-smi --query-compute-apps` 返回空，脚本里“检测到 qc 就让行”的守卫实际看不到 qc 的 `search_play`（它占 4.5GB 显存）。本次 13 个用例每个 3–10 秒、严格串行、不启动游戏进程，跑完 qc 仍在正常运行（进程数 27、显存占用不变）；但这说明该守卫在本机形同虚设，不能当作“已让行”的证据。

### 全量控制（已尝试，本机内存状态下不可行）

新增 `scripts/task96_full_pack.py` 读完整348段/16,982,931行；使用冻结bec2b9e6训练器，原batch2048、seed1、epoch1、workspace=:4096:8，按strict/default/strict顺序控制。所有memmap/输出/telemetry只写repro，现有输入只读，source_rows须与原控制SHA一致。每个子进程最多900秒；逐次记录初始权重、完整randperm摘要、实际张量与loss摘要、运行源码/资源回执。该控制针对尚未验证的全量路径，不重跑小段诊断。

准入直接枚举/proc中的实际Python训练进程，不依赖WSL可能漏报的GPU客户端列表；qc训练优先、本任务最多一个GPU训练。其他任务照常运行，不作为阻塞条件，仅在资源回执中记录其负载，计时比必须带共享主机负载限制。至少10GiB可用内存、3GiB空闲显存才启动。没有控制其他任务或qc。

**实测结论：这个控制在本机当前内存状态下跑不完，与确定性无关。** 10-26 17:03 启动 `strict_a`：17:04 读完 348 段（16,982,931 行，`source_rows.json` 已写出），随后训练稳定在 **0.099 步/秒（10 s/步）**，700 s 内进程 `read_bytes` 累计 **542 GB**——一个 epoch 只需读 5.5 GB board，这说明 9.6 GB spill 的随机行访问几乎全部落在 page cache 外。按 461 步/1.05M 行换算，全量 8,292 步需约 **23 小时/epoch**，与历史全量 221–224 s/epoch 差两个数量级。900 s 子进程上限触发，`strict_a` 未写出 `trace.json`；`artifacts/task96/full_pack/strict_a/{recipe.json,source_rows.json,telemetry/}` 与 `strict_a.log` 保留为这次失败的证据，未重跑。

原因不是代码而是宿主：本机 25 GiB 内存，实验时其他任务（deckvar 训练 11 GiB RSS、qc/llmimport 的 search_play 与游戏进程）占用 16 GiB，可用 8 GiB，装不下 9.6 GB 工作集；r11 当年 221 s/epoch 说明那时缓存装得下。

18:20 又在 `MemAvailable` 回到 13 GiB 时用 `task96_full_pack.py rate`（只跑 200 步就停）复测了一次，结论相同：训练阶段 18 分钟内 mmap 读盘累计 **650 GB**，进程 RSS 涨到 10.9 GB，200 步没跑完（> 5.4 s/步，全量 epoch 下界 > 12 h），随后主动 TERM 掉。原因是本机 `earlyoom` 的 `--prefer` 列表含 `python3`，再跑下去有杀掉其他任务训练的风险（deckvar 那个进程 RSS 10 GiB），不值得。

**在内存让出之前不要再试全量控制**。可行的启动条件是 `MemAvailable ≥ 14 GiB`、其他任务不占 10 GiB 级 RSS，且接受单次 4 分钟级 epoch；满足不了就只保留中尺度结论。

## 中尺度控制（本次，已跑完）

全量控制在内存上不可行，于是改用同一份 348 段里的**固定子集**做尺度更大的控制：按路径排序取前 30 段，1,046,710 行（是受限诊断那一段的 442 倍），其余全部沿用历史 run 的冻结训练器与配方（batch 2048、seed 1、epoch 1、461 步）。脚本 `scripts/task96_scale_probe.py`，原始回执 `artifacts/task96/scale_probe/<case>/trace.json`，汇总 [task96-scale-probe.json](task96-scale-probe.json)。共 9 个用例：5 个固定配置（含压显存、含两次交替重复），4 个未固定配置。

| 用例 | deterministic | workspace | 启动时空闲显存 | 最终张量 SHA | train / val loss | epoch 秒 |
| --- | --- | --- | ---: | --- | --- | ---: |
| strict_a | 开 | `:4096:8` | 9229 MiB | `78cfef547345…` | 5.3861950943 / 4.7820728496 | 27.66 |
| strict_b（重复） | 开 | `:4096:8` | 9229 MiB | `78cfef547345…` | 同上 | 28.60 |
| strict_squeeze_e（压显存） | 开 | `:4096:8` | 4807 MiB | `78cfef547345…` | 同上 | 19.12 |
| strict_q1（交替） | 开 | `:4096:8` | 11793 MiB | `78cfef547345…` | 同上 | 18.70 |
| strict_q2（交替重复） | 开 | `:4096:8` | 11793 MiB | `78cfef547345…` | 同上 | 17.98 |
| default_c | 关 | 未设 | 9229 MiB | `1cf71d20337b…` | 5.3448550556 / 4.7088792343 | 22.03 |
| default_squeeze_d（压显存） | 关 | 未设 | 4871 MiB | `05b226f28628…` | 5.3235213711 / 4.6689760475 | 19.69 |
| default_p1（交替） | 关 | 未设 | 11793 MiB | `b6e8942cc9d0…` | 5.3489700110 / 4.7110751228 | 13.51 |
| default_p2（交替重复） | 关 | 未设 | 11793 MiB | `9b9f594e2ed9…` | 5.3520673691 / 4.7137228276 | 20.93 |

`squeeze` 的做法：在模型创建前占用 4 GiB 显存（`torch.empty(4 GiB, uint8, device='cuda')`），把空闲显存压到约 4.8 GiB，训练结束后释放。没有触碰 qc 进程，占用期间 qc 一直正常。交替四例（default_p1 → strict_q1 → default_p2 → strict_q2）是为了让缓存预热与负载漂移在两种配置间对消。

读法：

- **固定配置：5 次全同。** strict_a/b/e/q1/q2 的最终张量都是 `78cfef547345…`，每一步 loss 也逐位相同（`losses_sha256` 一致）；初始张量 SHA 与 `randperm` 摘要在全部 9 例里相同。跨 4807–11793 MiB 三种空闲显存都一样。到 1.05M 行 / 461 步为止，固定配置下没有 run-to-run 随机性。
- **未固定配置：4 次全不同。** default_c / default_squeeze_d / default_p1 / default_p2 的最终张量两两不同，val loss 4.7089 / 4.6690 / 4.7111 / 4.7137。其中 **default_p1 与 default_p2 是同一进程序列、同一份数据、同一 seed、报告的空闲显存也相同（11793 MiB）**，结果仍然不同。所以不是“显存少就一定变、显存一样就一定同”，而是**不固定 workspace 时结果依赖运行时状态**，空闲显存只是其中一条已观测到的轴。
- **固定 workspace 就免疫。** strict_squeeze_e 与 strict_a 逐位相同、strict_q1/q2 与 strict_a 逐位相同。修法 5（把 `CUBLAS_WORKSPACE_CONFIG` 写死）不只是让配置可比较，它确实消掉了这个已观测到的、可重复出现的漂移。
- **这正是历史四次全量 run 的形状**：同数据、同代码、同 seed，四个结果互不相同。当时的 run 没有固定 workspace，也没有留环境回执，与这里“未固定 → 每次不同”的行为一致。**但仍不能说已证明历史原因**——没有当时的环境回执，只能说机制相容且已在本机复现。
- **机制仍是推断**：只观测到“运行时状态 → 数值”的因果链两端，没有采到 cuBLASLt 选核或 workspace 申请失败的 API 回执。可以写“与库在运行时状态不同时选择不同实现相容”，不能写“已证明是 cuBLASLt 换核”。

局限（必须一起读）：

- 只有 1.05M 行 / 461 步，不是 16.98M 行 / 8,292 步。它证明的是“这个尺度上固定配置可复现、未固定配置每次都变”，不能外推成“全量逐位可复现”。
- 子集是“按路径排序前 30 段”，不是随机抽样；不同标签的混合比例与全量不同，loss 绝对值不能与全量表对比，只能用例间对比。

## 分歧定位（本次，已跑完）

前面只证明了“未固定就跑不一样”。为了说清**差在哪一步、哪个张量**，在同样的 30 段子集上做了两组更细的观测（`scripts/task96_localize.py`、`scripts/task96_grad_probe.py`，证据 [task96-localize.json](task96-localize.json)、[task96-grad-probe.json](task96-grad-probe.json)）：

1. **模块级（前 6 步）**：两次未固定 run 的模型输入哈希在第 1–6 步全部相同，第 1–3 步 loss 逐位相同；**第 2 步起 `packet_embedding` 的输出哈希不同**（22 个模块里第一个不同的），第 4 步 loss 才出现末位差异（7.658308506 vs 7.658308029）。固定配置的两次对照：6 步 22 个模块全同。
2. **梯度级（第 1 步）**：trainer 在 `loss.backward()` 之前调用 `step_observer`，所以第 2 次回调时手里的梯度正是第 1 步的。两次未固定 run 的第 1 步 forward 与 loss 逐位相同，**梯度里只有 `packet_embedding.weight` 不同，其余 20 多个参数全同**；做完第 1 次更新后，参数里也只有它不同。固定配置的两次对照：梯度一个都不同。

含义：

- 分歧进入点不是初始化、不是洗牌、不是前向、也不是 loss，而是 **embedding 表的梯度累加**——同一 batch 里重复的 packet id 要把梯度散射累加回表行，这个累加顺序在 CUDA 上不固定（`index_add`/`scatter_add` 式原子累加）。前向的查表是确定性的，所以第 1 步前向和 loss 完全相同；梯度先分叉，第 2 步的查表结果才跟着变。
- 这与“cuBLAS workspace 换算法”是**两条不同的路径**：embedding 梯度不走 GEMM。之前观测到的 workspace 敏感性影响的是后面的 GEMM 步；本次定位到的第一步分叉与 workspace 无关，所以单靠固定 `CUBLAS_WORKSPACE_CONFIG` 未必够，`torch.use_deterministic_algorithms` 这类开关才是针对这一条的手段（正在用“只开确定性、不固定 workspace”的两例做对照，qc 训练优先，排队中）。
- 仍未拿到内核级回执（没有 atomicAdd / 排序实现的 kernel trace），所以“原子累加顺序”是**与观测相容的机制解释**，不是已证明的内核行为。
## 冻结与修法建议（生产未改）

1. 每轮先固定标签快照；对 frames/meta/显式训练行清单以及 pack index 和每个引用段保存内容 SHA256。内容清单写入 `round_*/inputs.json`，训练前后核验。不把路径、大小或 mtime 当内容身份。现有不可变段可按内容地址复用，避免重复复制整库；保留到评估回执已固定其依赖。
2. `round_*/runtime.json` 记录实际命令、resolved recipe SHA、Git revision、dirty diff SHA、训练链各源码 SHA（含导入依赖），以及 Python/numpy/PyTorch/CUDA/cuDNN/驱动/GPU、精度与确定性设置、CUBLAS workspace。配方来源 revision 与实际训练运行 revision 分开保存。
3. 显式独立的 CPU generator 管理模型初始化与训练 permutation，验证 split 使用独立 seed；存启动 generator 状态摘要、初始化张量 SHA、各 epoch permutation SHA。以后即使引入 DataLoader，再显式配置 generator/worker_init_fn 以及 Python/numpy RNG；当前不存在 worker 需要补种子的证据。
4. 验收比较实际张量与 loss，模型文件 SHA另作文件完整性校验；不能因 metadata/输出路径差异误判权重不同。相同库/设备下检查逐位复现，跨版本/跨设备另设数值容差与行为评估。
5. 生产训练固定 `CUBLAS_WORKSPACE_CONFIG`（配合 deterministic algorithms / cuDNN deterministic / benchmark=False）写进 `runtime.json` 与启动脚本，作为“同配置可复现”的前提。诊断显示换 workspace 就换算法、换结果；中尺度控制还显示**未固定时同一配置 4 次跑出 4 个不同结果**（val 4.7089 / 4.6690 / 4.7111 / 4.7137，其中两次报告空闲显存相同），固定后 5 次（含压 4 GiB 显存）逐位不变。不写死它，“同配方同 seed 重训”这句话本身没有定义，而且共享主机上任何一次重训都可能落在不同显存状态。改动前先在一个完整 epoch 上量开销。
6. **task92 的前提要改写。** “训练随机性是主要来源”在已观测的两个尺度上都不成立：固定配置的重复训练逐位相同（小样本 7 次、1.05M 行 2 次）。四次全量 run 互不相同，与“未固定 workspace + 不同空闲显存”相容。所以：
   - 多种子挑选测的是**种子方差**（不同 seed → 不同初始化与不同排列，这是真的），不是“同 seed 重训的抖动”。
   - 挑选时**必须**把 `CUBLAS_WORKSPACE_CONFIG` 与确定性开关固定住。否则各种子落在不同显存/算法状态上，挑选结果里混进的是环境差异，不是种子效应，多训几个种子再挑也挑不出真实方差。
   - 对 42/192 的 heldout 落差，仍需要多个完整训练 + 新种子配对评估，不能用单个 r11 称“幸运”就认定原因。

## 复核命令

所有新写入均在 repro 内；`/tmp/exp96` 只是 CodeBuddy 已留的冻结只读训练代码快照。执行前保持 qc GPU 优先。

```sh
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_existing_runs.py
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_provenance.py
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_determinism_probe.py prepare
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_determinism_probe.py run

# 计时用例（8 epoch，strict 与 default 各两遍；结果在 artifacts/task96/determinism_probe/{strict_cost_*,cost_*}/trace.json）
TASK96_EPOCHS=8 /home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_determinism_probe.py child --case strict_cost_a

# 中尺度控制（30 段 / 1,046,710 行，5 个用例，约 3 分钟；已跑完，重复执行会因 trace.json 存在而拒绝覆盖）
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_scale_probe.py prepare
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_scale_probe.py run

# 全量控制（348 段）：当前内存状态下会退化到 10 s/步，别在 MemAvailable < 12 GiB 时启动
/home/newbiexvwu/.venvs/ml/bin/python3 -B /home/newbiexvwu/PvZ-search/.claude/worktrees/repro/scripts/task96_full_pack.py prepare
```

`run` 现在跑 9 个用例（含 `:16:8` 与未设 workspace 各两遍），全部完成后写 [task96-determinism-probe.json](task96-determinism-probe.json)。
