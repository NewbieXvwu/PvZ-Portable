# task162：台式机显卡空档实验执行器

分支 `gpuwindow`，基于 `origin/prodint16`。在台式机做（Codex 账号 A）。

## 背景

决策研究（task157 / 158 / 160）要在生产规模的数据（约 2300 万行）上训练多组对照结构。台式机显卡大部分时间被生产训练占用，但有固定空档：
- 每轮训练包打包期间（v2timer 首次打包约 1.4 小时）；
- 每轮 K=4 个候选训完后的挑选 + 门禁（约 1–1.5 小时，只用 CPU）。

现在这些空档白白浪费。

## 要做的

1. **执行器**：一个常驻的 systemd 用户服务（Nice=10，限内存）。从队列目录读取实验作业（训练命令 + 配方 + 输出目录）。
   - **启动条件**：生产没有训练作业在跑（生产后台执行器的锁 / 进程 / 状态文件，读代码确认哪个最可靠），且显存、内存满足作业申报的需求。
   - **让路**：生产训练一旦准备开始，作业要立即让出。办法是保存检查点后退出，下次空档续跑；不能让生产训练等待。读 `curriculum_queue.py` 的训练准入逻辑，确认生产准入不会因为实验进程占内存而饿死（10-27 曾有双饿事故）。
   - **记录**：每个作业的空档用时、累计进度、结果都要记下来。
2. **训练脚本**：支持断点续训（`train_rollout_policy.py` 是否已支持 `--resume`？不支持就加，默认行为不变），并能按配方切换结构（现有 `arch` 字段，以及 task157 / 158 以后新增的结构）。
3. **评测**：作业训完后在 CPU 上做留出评测（逐行指标：见过 / 没见过的卡组、分地形）。对局评测交给 Mac 的 `bench_play.py`（task160 C 节），这里只要把检查点放到约定位置。
4. 先用两个示例作业跑通：r42 v2timer 训练包上的 CNN64 和 CNN128。

## 约束

- 生产优先，这是硬约束：不改 qc、看门狗、生产配方和状态文件，不 `pkill -f` 含 `curriculum_queue` 的模式。
- 测试只跑 `-k`，加 `ulimit -v 8000000`，先看 `free -g`。不加自校验。
- 分步提交推送 `gpuwindow`。结论写在本文件末尾：空档统计、让路实测（生产等待了多少秒，目标 0）。

## 实施记录（2026-10-29）

- 第一步 `af5849d4`：训练 opt-in 断点续训。保存模型、AdamW、cosine 调度、CPU/CUDA RNG、epoch 洗牌 RNG 和批次游标；`SIGUSR1` 在批次边界保存后退出75。加载期间没有优化器进度，直接退出；超时可从30秒定期断点恢复。原命令默认行为保留。
- 执行器 `scripts/gpu_window_runner.py` 已作为 `pvz-gpu-window.service` 启用。Nice10，MemoryHigh4GiB，MemoryMax5GiB，MemorySwapMax0，CPUQuota150%。启动脚本：`bash scripts/install_gpu_window.sh ~/pvz-logs/gpu-window`。
- 生产 `training.lock` 覆盖 pack/train/select/gate，不是GPU占用锁；`curriculum_state.json` 只随前台边界刷新，实测后台已在训练时仍显示 queued。执行器改读 `/proc` 的真实脚本 argv 与父子关系：只开放生产 pack、select_student、promotion_gate 子进程存在的窗口；有外部训练进程、控制器缺失或阶段转换均关闭。
- 生产准入 `_admit()` 同步看 MemAvailable；不能只等生产训练进程出现才释放主存。因此启动额外预留12GiB主存、8GiB显存，运行时低于预留也让路。r42 的 `training_mode.json` 申报串行准入5,930,622,976字节，近期训练匿名峰值约2.7GiB，双训练加3GiB保留仍低于12GiB。预留让生产准入无需等待实验退出；0.5秒轮询，信号后2秒仍未退出只杀实验自己的进程组。没有给生产加锁或等待协议。
- 已冻结 r42 v2timer 的23,631,235行、1320段，CNN64/CNN128 保留原12 epoch/seed1，init=null。使用独立 pack 索引和合并锁，段及已完成的 mmap 数组只建硬链接，未重打包、未复制14GiB数据、未重放存档。生成命令：`python scripts/prepare_gpu_window.py --recipe ~/pvz-logs/curriculum_state.pipeline/round_000042/recipe.json --state ~/pvz-logs/gpu-window`；该目标目录只生成一次，重复实验用新目录。
- CPU逐行留出评测使用配方的原种子切分；实际公开卡槽类型排序（保留重复卡、忽略槽位）区分训练见过/没见过的手牌，分白天/夜间/泳池/雾/屋顶，L35走夜间特例；空组显示0行和null指标。植物召回、卡/格准确率使用植物行分母，另报动作类型、联合准确率和NLL。GPU训练跳过逐epoch验证，CPU评测可独立中断续接。
- 本分支未包含 task157/158 的新结构或 task160 C节文档；增加 `arch=cnn` 默认与 `module:factory` 入口，后续结构通过同一训练/加载工厂接入，不虚构尚未实现的模型。当前本机交接点为 `~/pvz-logs/gpu-window/mac-handoff/<job>/model.pt`，训练结束即硬链接放置，CPU评测结束写总清单；跨机器仍走HF transfer，不走git。

运行数据（服务实时更新，不进入git）：

- `~/pvz-logs/gpu-window/queue/status.json`：各生产阶段累计秒数、当前作业、观察到的生产内存等待时间。
- `~/pvz-logs/gpu-window/queue/events.jsonl`：空档、启动、让路与退出事件。
- `~/pvz-logs/gpu-window/runs/<job>/window.json`：每次空档用时、退出/让路耗时、累计用时和最近进度；`progress.json` 每30秒写训练进度；`heldout.json` 保存CPU评测。

### 当前结论

执行器和两项完整作业已安装入队。15:01启动时生产正在 r42 seed2 训练，服务按设计保持空闲，尚未经历首个真实空档。因此实验空档使用0秒，真实让路样本0个；没有观察到新增生产内存等待日志，**不能据此声称已验证生产等待0秒**。首个真实窗口、完整CNN训练结果及生产阶段切换让路实测仍待运行数据，勿把小样本测试当成生产实测。

相关小型CPU测试覆盖断点后优化器轨迹一致、默认训练仍可加载、阶段与资源准入、只向实验进程组发信号、CPU评测计数续接及提前交接模型；每次均先看可用内存，`ulimit -v 8000000`，按 `-k` 选择相关项，无全量训练/回放验收。

## 决策研究全部入队（2026-10-29 16:04，台式机）

已实际写入 **16 个训练作业（8组×seed1/2）+2 个准备作业**，并恢复常驻服务。旧 `r42_cnn64` / `r42_cnn128` 两个未启动示例保留原文件、设置 enabled=false，由带共同卡组留出的新作业取代，避免重复训练或混用两种划分。

所有新作业及执行器均锁定代码提交 **`8f99d1e2888f5fa9726621d070bca0e6ea35829e`**，实际执行目录为 `~/pvz-logs/gpu-window/code/<完整SHA>/` 的 detached worktree。训练配方记录同一 SHA；启动/续跑/评测始终调用该目录脚本，模型交接也记录 SHA。后续更新 gpuwindow 分支不会悄悄改变这些作业。

| 优先级（小者先） | 组 / 作业前缀 | 结构 | 种子 |
|---|---|---|---|
| 0 | decision_split（准备） | 冻结r42原帧引用、建立精确行计划与整卡组留出 | 固定划分种子162 |
| 10 | r42_cnn64_s | CNN64，原meanmax/row_pool/identity头 | 1、2 |
| 20 | r42_cnn128_s | CNN128，其他项同CNN64 | 1、2 |
| 30 | r42_resnet6_s | width64，输入卷积+6块双卷积残差块，原三头 | 1、2 |
| 40 | r42_terrain_s | width64，前两层共享，第三层/标量编码/三头按实际地形选1/5专家 | 1、2 |
| 50 | r42_moe_s | width64，整盘路由；1共享+4选2专家，动作头共享，0.01负载均衡损失 | 1、2 |
| 60 | decision_tokens（准备） | 从精确r42输入帧构建所有token组共用的稀疏float32附加数据 | 不训练 |
| 70 | r42_v3_small_s | width96、2层、4头、FF384，关系注意力+SwiGLU+指针头 | 1、2 |
| 80 | r42_v3_medium_s | width192、4层、6头、FF768，同上 | 1、2 |
| 90 | r42_cnn_entity_s | 小型v3编码器+CNN格子分支，实体scatter回格子 | 1、2 |

同组按累计已用空档时间调度两个种子；每段最多1800秒，再从保存的逻辑批次位置续跑。准备作业同样受生产窗口、主存上限和让路机制约束；**入队不等待全量token构建**，CNN系列先跑，三个token组依赖附加数据的ready标记。

数据、输入与评测：

- 共同数据仍是已冻结的r42 v2timer包：23,631,235行、1320段，原标签、软标签、权重来源不变；12 epoch、seed1/2、init=null、有效batch2048。微批次分别为256/128/128/128/64/16/8/16，以逻辑批次统一的权重分母累计梯度和单次AdamW更新；学习率按逻辑更新步数推进。
- v3网络及token契约三份源码来自 **origin/studentv3net 的 `2a8667832bbfe97429aac6501cc931b5e936b17c`**。现有r42包没有token，因此准备作业按包中source_row取原公开帧的 `source_row-1`，补建相同行的token，不重采样、不重标注、不重放引擎。附加数据只存非零float32特征和线性大小索引，不存N²关系。它使用该提交的紧凑帧token编码，**不是后续富obs-v3回填语料**；混合组同时读原v2timer格子输入。
- 共同划分先按地形、固定种子162从有明确卡组的档案选择整套卡组，再将匹配这些完整卡组多重集的所有行从训练排除；其他行保留原seed0/10%对局种子留出。卡组身份忽略槽位、保留重复卡；传送带可变手牌不作为挑选静态留出卡组的依据。缺少可留出组合的地形保留空组，不伪造样本。
- CPU评测对全部留出行统计总体、seen/unseen、五地形及两者交叉组；动作类型、植物召回、卡/格、完整动作准确率与NLL使用各自真实分母，空组为null。MoE另保存5地形×4专家的路由计数。训练时的卡频/正例/分差统计排除卡组留出行。
- 训练结束立即将模型硬链接到约定位置 **`~/pvz-logs/gpu-window/mac-handoff/<job>/model.pt`**，同时放 `recipe.json`、`deck_split.json`、`ready.json`（完整代码SHA、结构、种子）；CPU评测结束另写 `<job>.json` 和 `heldout.json` 路径。Mac按同一代码SHA加载，CNN可走原PVZRN006路径，新结构走bench_play的Python原型路径；跨机器传权重仍走HF transfer，不把权重放git，也未启动Mac对局或部署生产。

运行清单：`~/pvz-logs/gpu-window/decision-study.json`；可复现配置：`configs/decision_r42.json`；入队命令：

```bash
python scripts/queue_decision_study.py \
  --state ~/pvz-logs/gpu-window \
  --recipe ~/pvz-logs/curriculum_state.pipeline/round_000042/recipe.json \
  --revision 8f99d1e2888f5fa9726621d070bca0e6ea35829e
```

本次交付是全部对照的实际入队。服务 active，仍为Nice10/MemoryMax5GiB；恢复时生产仍在训练，当前无实验占GPU，尚无模型或胜率结果。准备/训练/评测完成后，各作业结果落在既有 `runs/<job>/` 与交接目录；当前不能给出真实让路或胜率结论。相关小CPU用例及原断点续训回归通过，未运行生产规模验收。

## seed1优先、实窗预算和磁盘约束（10-29续办）

- 所有8臂seed1优先；seed2除加100优先级外，还依赖8个seed1全部完成，因此不会在某个seed1等数据或暂停时抢跑。task163仅导入seed1/2，7组共14作业，固定其准备代码 `cfa42c1c6d669f849b63bf52df90a2db6aafc072`，在全部16个decision作业及临时数据清理之后执行，优先级1000/1100。该小语料研究保留自己的原配方，不混入r42对照。
- 下一可用窗口先做decision_split，再运行有GPU申报的吞吐探针；探针按地形取r42真实行，完成真实前后向、AdamW、同有效batch2048和相同微批逻辑，预热后逐臂计时，不等全量token附加数据。8臂全部测完才放行正式训练；预算文件一次确定后，所有臂/种子共同使用，检查点记录实际轮数、行比例及预算摘要。无CUDA不生成CPU替代速率。
- 22:51的只读48小时统计：训练进程30.03 h，可调度pack/selection/gate窗口11.27 h（23.48%）；未观察控制器空白不算可用窗口，也不把GPU空闲上限当实际可用率。实测速率按25%余量折算各作业/总GPU小时和日历天；若seed1超过3天，先统一减epoch；一轮仍超过3天才统一固定行哈希抽样，保留完整留出集。全量token构建和CPU评测耗时另计，不能冒充GPU训练小时。
- 额度中断期间实验服务曾停在切换阶段；22:42恢复并拦住未确定预算的训练，生产未动。后续以恢复后的下一实际窗口为测量窗口，尚未产生实测速率。
- 用户磁盘约束：不复制大数据，已有包/合并数组直接引用，原帧继续硬链接。宿主C约15GiB、D约47GiB，不能用WSL虚拟df的818GiB估可用空间。新增token缓存只存统一预算采用的训练行，省掉重复cell索引和全行空记录，CPU留出逐档案产生token；缓存预算至多2GiB，按样本P95×1.5估体积，预算时留13GiB宿主余量、构建低于12GiB即停止。若磁盘要求比3天预算更严格，进一步统一降低所有臂的训练行比例并写入实测报告。
- 新大文件及用途：共同row mask约22.54MiB，原帧/元数据引用计划预计数百MiB；token缓存≤2GiB，实际字节数写token ready.json；模型字节数写window.json。训练成功后即删除无用途的resume.pt并记删除字节；全部decision训练/评测结束后清理token .bin和研究原帧视图，保留生产原件、清单及Mac模型交接。模型权重仍待Mac经HF取用后清理，不提前删除结果。

续办落地：已实际入队14个task163作业（无seed3），seed1/seed2屏障已部署。吞吐报告发布也已安排为准备作业：下一空档测满8臂并确定共同预算后，使用一次性独立检出追加本任务书和当日日志，普通fast-forward推送gpuwindow，再删除该临时检出；仅在当前工作树干净且能fast-forward时同步它，**不移动作业使用的固定代码检出**。因此无需让Agent同步等生产训练数小时，实测速率、GPU小时、折算天数与实际缩减比例会在报告产生后自动落入任务书。发布失败会保留外部report.json/report.md，不假称已发布。

磁盘方案再压缩：54个固定cell token及cell索引不逐行落盘，读取时按相同公开编码恢复；仅存实际训练行的全局行号和变长token。CPU留出顺序读取原帧，既不需要全量token缓存，也不复制大语料。缓存仍≤2GiB；用完连同row mask、二进制source plan和研究硬链接视图一起删除，JSON清单及结果保留。新增代码检出实测各33MiB，旧中间检出在无活动作业引用后移除。

23:08部署收尾：决策作业和执行器重新锁定 **`323b867caf382efa4e4b9aea6829777d1d8d31f8`**；task163仍锁定 `cfa42c1c6d669f849b63bf52df90a2db6aafc072`。启用队列共35项（16个decision、14个task163、split/probe/publish/tokens/cleanup五项准备作业），另两项旧示例仍禁用。服务active，生产仍在训练，实验未占GPU；实测表待下个真实窗口自动追加。新版代码检出33MiB，task163代码检出33MiB；无活动作业引用的f9b5ad75中间检出33MiB已移除。未复制大数据，尚未产生token缓存或模型大文件。

## decision_split 跳过已删除原帧的行（10-30 续办）

- 失败点：r42 冻结包引用 256 份 task20 `d10orig1/2_fast_search` 存档（seed_1880000–1880255，57,970 行，占 0.245%），其原始帧随旧 conveyor worktree 删除且无备份（生产帧目录里指向它们的软链已移到 `~/pvz-logs/retired_d10orig_links/`）。split 在 `os.link` 处 FileNotFoundError，作业 returncode 1。
- 处理（提交 `9974c1a6`）：缺帧的存档**不建硬链接、不建 token、不打分**，但**仍占住它在包里的行号**——行计划保留这些行并推进计数器，`unseen.npy` 把它们标为不训练，留出评测跳过它们，吞吐探针也跳过。没有新增校验步骤或清单文件。
- 顺带修正行计划的枚举顺序：段是按采集轮次追加的，`ORDER BY source` 得到的偏移与包内行号不一致（实测 83,957 个源全部错位），这会让 token 缓存的 `stored_row` 与留出流式 `ArchiveTokens` 把行号解析成别的存档。split / tokens / `ArchiveTokens` / 探针统一改成 `ORDER BY rowid`（插入顺序=包内顺序；已只读核对已冻结的 1164 段完全一致），跳过标记才会落在正确的包行上。评测的“帧是否还在”查**冻结视图**而不是生产目录，避免生产后续裁剪时误删有效留出行。
- 部署：新增固定检出 `code/9974c1a6…`，21 个 decision 作业（split、tokens、16 个训练臂、探针、发布、清理）重新锁定到该提交，原作业 JSON 与配方备份在 `queue-history/9974c1a6…/`；task163 的 14 个作业仍锁定 `cfa42c1c…` 未动。执行器经 `install_gpu_window.sh` 从新检出重装并重启（Nice10 / MemoryMax5GiB 不变），split 从 `progress.json` 的 segment 1164 续跑。生产 qc / 看门狗未触碰，重启后 `production_wait_observed_seconds` 仍为 0.0，服务在等下一个 pack/selection/gate 窗口。旧检出 `323b867c…`（33MiB）已无作业引用，留待磁盘紧张时再删。
- 验收：`ulimit -v 8000000` + `-k "split or tokens or eval or deck or terrain or metric or budget or decision"`（test_decision_queue / test_gpu_window / test_window_budget）15 passed，新增两个小用例分别覆盖 split 缺帧跳过（并验证包行号占位）与 tokens 跳过时行计数推进。未运行生产规模验收。

## 10-30 Codex 续办：纠正合并行顺序与实际部署

- 复查 Qoder 的 `9974c1a6` / `6a116a0f`：保留已删除 d10orig 原帧的跳过处理；**上节关于 `ORDER BY rowid` 的结论撤回**。SQLite 插入顺序等于按 index 追加的段顺序，但 `train_rollout_policy.py::_read_segments` 明确执行 `chunks.sort(key=lambda chunk: str(Path(chunk[0]).parent))`，再把所有输入、标签、seed、权重、卡槽和候选写到共同排序的位置。研究使用的 r42 `corpus/merge/*.npy` 正是该合并结果，不是原段直接拼接。83,957 个源相对原段拼接的偏移不同，不能据此认定相对训练数组错位。实际数组首行 seed=1600000，尚未完成的 SQLite 插入首源却为 seed_7804240，也直接否定了 rowid 对齐合并数组的判断。
- split、tokens、流式 `ArchiveTokens`、吞吐探针统一恢复 `ORDER BY source`（源名均为相同 basename `frames.npz`，与读取器按父目录排序一致）。原帧缺失的行仍占据**合并数组中的**行号，训练屏蔽与评测跳过保持生效。修正 Qoder 小用例：逆序插入的源应在排序后的合并位置被屏蔽，并覆盖逆序 token 源及流式读取；仅这两个相关 `-k` 用例运行，2 passed，Nice10、8GB虚拟内存上限。
- **生产打包/训练不受该 SQLite 偏移问题影响**：生产 `cq` 的读取器同样按 source 父目录排序，来源回调也沿同一 `placed` 顺序执行；生产不读 task162 的 `sources.sqlite`，单段的 `source_archive/source_row` 与各列同时生成，合并时各列同时搬移。不需要重打包、重训或改 qc。Qoder 的 rowid 改动仅影响尚未成功准备完毕的决策研究，没有产生可用研究模型。
- 03:54发现两处实际部署阻塞：systemd 文件已改为 `9974c1a6`，但 MainPID 437 自01:43启动后仍运行 `323b867c`；`enable --now` 不会重启正在运行的服务。与此同时 split 的既有 `window.json` 仍为 failed，重配作业不解除失败状态，故03:26–03:53约27.5分钟pack窗口没有启动研究。安装脚本改为 enable 后显式 restart；仅调整研究服务与研究队列状态，不动生产。
- 03:56:46实际部署 **`b781d8468d19241b59e82513b45e8702cff84fa6`**：研究服务 MainPID 已换为22688，ExecStart指向该固定检出；21个decision作业、16份训练配方和研究清单同步更新，探针命令此前遗漏的 `--revision 323b867c…` 也已纠正。split既有失败状态改为paused，保留原尝试记录和segment1164进度；未生成新的校验、manifest或JSON中间文件。Nice10、MemoryMax5GiB、CPUQuota150%保持。当前生产r52 seed1训练，尚未进入下一研究窗口，吞吐待实际探针记录。

## 10-30 10:01 续办：WSL恢复后的真实进度、首批吞吐与生产来源链

**研究已在真实空档推进，但短探针严重低估端到端耗时；生产未发现这次 source 顺序问题，不需要因此重打包或重训。** 本次只读取既有代码、服务状态和日志，追加本文与当日日志；未改生产或研究运行配置，未新增校验、manifest、JSON文件，也未启动测试、回放或训练。

### 服务和实际推进

- `pvz-gpu-window.service` enabled / active，自09:10:06启动，MainPID **431**，实际argv为固定检出 `code/b781d846…/scripts/gpu_window_runner.py`。Nice=10、MemoryHigh=4GiB、MemoryMax=5GiB、MemorySwapMax=0、CPUQuota=150%均保留。生产Python控制器/打包/训练/挑选脚本仍来自 **`cq`（本次读取HEAD `c84ad7d1`）**，游戏可执行文件为 **deploy19**；不能把deploy19游戏构建当成Python读取器版本。生产日志09:03:09记录r52已激活，09:14:39记录r52/r53待办由r54取代，当前是r54训练，研究正确处于paused / active=null。
- split已完成 **1320段 / 23,631,235行**。八臂探针于 **08:20:51** 完成（重启前），不是本次临时重测；CNN64、CNN128已有断点进度，其他六臂和token准备尚未开始，没有已完成研究模型。
- 重启后 **09:14:58.572–09:27:48.761**，CNN64在生产pack窗口实跑 **770.19秒**。既有metrics日志记录09:16:30为step6 / 12,288行，09:27:00为step17 / 34,816行，期间净增 **11步 / 22,528行**，稳定区间 **35.75行/秒**，证明不是只启动后空等。当前resume仍保存step17；CNN128保存step2 / 4,096行。
- 最新让路：09:27:45.807请求退出，09:27:48.049已观察到生产训练，研究09:27:48.761退出，释放耗时 **2.95秒**，该次为超时强制结束（-9），不是全部优雅退出。`production_wait_observed_seconds=0.0`，未记录生产内存准入等待；这是日志观察值，**不能证明所有阶段切换的生产等待严格为0秒**。
- 自动发布作业08:20:53在`git fetch`返回128后failed，训练不依赖它，故并未阻塞训练。本次fetch已成功，以下将已有报告连同实跑修正手工写入任务书并提交，不新建发布中间文件。

### 首批吞吐与全部decision seed1预算

以下为既有探针：240个真实输入行，五地形各6份档案；每臂预热两步后计时约10秒，真实前后向、AdamW、有效batch2048及既定微批。它覆盖的是小工作集，不是整个训练包的随机访问吞吐。

| 结构 | 探针行/秒 | 现预算单作业纯计算分钟 |
|---|---:|---:|
| CNN64 | 13,043 | 0.47 |
| CNN128 | 8,182 | 0.75 |
| ResNet6 | 6,015 | 1.01 |
| 地形专家 | 3,217 | 1.90 |
| MoE | 2,326 | 2.62 |
| v3 small | 399 | 15.30 |
| v3 medium | 180 | 33.92 |
| CNN + entity | 391 | 15.60 |

- 冻结包原有效训练行 **20,868,626**。既有共同预算因磁盘进一步收缩到 **1 epoch、固定行哈希比例0.0175422848（1.7542%）**，每臂约 **366,083行**，两个种子相同，留出集保持完整。全部decision **8个seed1**按探针合计 **1.49 GPU小时**（含25%余量）；以报告当时近48小时可调度窗口 **11.22小时 / 23.3647%**折算是 **6.38日历小时**。原12轮全行是1019.95 GPU小时 / 181.89日，因此当前已经是显著缩减的数据预算，不能按完整r42训练解释后续结果。
- **6.38小时不能作为当前完工承诺。** CNN64累计34,816行 / 1451.90秒作业占窗时间，只有 **23.98行/秒**；CNN128为4,096行 / 252.54秒，即 **16.22行/秒**（均含加载、续跑及无进度尝试，不含窗口外等待）。CNN64上述稳定区间也仅35.75行/秒。若暂把同样的约36.6万行×8臂代入35.75–16.22行/秒，并保持23.3647%窗口率，得到 **4.1–8.9日历天的训练情景估计**，CNN64累计口径对应6.05天；这不是置信区间，也未实测其他六臂的全语料速度。当前应按**约一周、可能更久**规划，不能报告“seed1半天完成”。
- 完整交付还要另加 **token构建、8臂完整留出CPU评测及其调度**，目前无实测时长，无法给出可靠完整完工上界。上述范围只指decision八臂seed1，不包括排在全部decision之后的task163七臂seed1。现有执行器把训练后的CPU评测也放在受生产窗口约束的作业阶段，不能把它当成窗口外免费工作。
- 现有metrics还显示CNN64最新一次进程`read_bytes`约 **749.26 GB**，其中上述630.10秒区间增读约627.18 GB；输入各列逻辑体积远大于研究5GiB内存上限。这支持**全包随机mmap访问产生读放大/缺页，热小样本探针无法代表实际工作集**的判断，不把它当作已完成的精确性能归因。本次不扩大内存、不加重测、不改运行配方；后续如改善研究读取，应优先在现有内存内读取预算采用的行、减小随机工作集，保留全局行号映射，而不是提高生产机资源占用。

### 83,957个source的偏移问题：逐条读取链结论

这次涉及三种不同坐标：**段内行号、index顺序直接拼接的行号、全局合并数组行号**。Qoder把第二种当成第三种，因而把83,957个source的偏移差异误判为训练数组错位；改`ORDER BY rowid`反而会让研究计划按段追加顺序去索引已排序的合并数组。正确修复已于`b781d846`部署：研究四处统一回到`ORDER BY source`，与本批全部同名`frames.npz`来源的父目录排序一致；缺失原帧的行仍占原合并位置并屏蔽。下面依据**实际生产cq源码**逐项判断，未全量重读训练数据：

| 读取/写入位置（cq路径与行号） | source与包行的对应方式 | 本问题的影响 |
|---|---|---|
| `python/pvz_frame_store.py:16`；`scripts/pack_training_rows.py:80,155,222,255,309` | `frame_paths`按父目录排序；每份档案连续生成各输入列、标签、seed、`source_archive`及`source_row`，共用同一`n`/切片。缓存路径同用`stored['rows'][selected]`，原帧路径的标签row i对应输入frame i-1。 | 正确；最后这一位差是输入/动作时序约定，不是这次source排序错位。 |
| `scripts/train_rollout_policy.py:398–435`，单包`read_pack` | 所有列保留包内顺序；按原`source[]`与`source_archive`计数推进`start`，不对source单独排序，回调和label_source权重作用于同一段。 | 正确；不能把研究的全局排序规则直接套进单包段内偏移。 |
| 同文件`:488–545,571–580,647–683`，多段`_read_segments` | 先由每个段的source列建立局部`start/stop`，再按source父目录排序；`placed`生成全局`pos`，board/scalar/value/candidate、plant/packet/cell/weight/aux、seed和packet_info都搬到相同`pos`。 | 正确；段追加顺序与最终行序不同是设计行为，不是生产特征/标签互相错配。 |
| 同文件`:698–714`，来源回调与source_weight | 回调沿排序后的`placed`执行，并传对应`source_row`；label_source倍率直接乘`weight[entry['pos']:pos+span]`。复用已有merge时也会执行这些回调及私有权重调整。 | 正确；不存在沿原段顺序给已排序数组加权的问题。 |
| `python/pvz_training_rows.py:59,71,115,139,154,179`；训练入口`:732–785` | `SourceRowCounts`的关卡平衡、`LabelWeightAdjustment`的共享包label倍率、`SharedPackAdjustment`的历史L7保留及卡组限权，都在上述回调中累计行段；随后对同一合并数组的weights/is_val/packet列操作。 | 全部沿合并顺序；本次特别覆盖了当前cq新增的共享包权重路径，没有遗漏为“仅来源统计”。 |
| `scripts/select_student.py:44–89`；`scripts/curriculum_queue.py:1748`；`scripts/promotion_gate.py:146–171` | 挑选与门禁加载候选model，按独立任务/seed跑`search_play`，再按seed读取games结果和胜负/屋顶波次；seed-offset是对局种子列表切片。 | 不读训练包、source偏移或研究SQLite；本问题不直接影响挑选/门禁。 |

补充范围：研究`prepare_decision_data.py`的split与tokens、`python/pvz_research_data.py::ArchiveTokens`、`bench_decision_window.py::samples`是使用SQLite累计全局偏移的四处，当前固定代码均为`ORDER BY source`。研究`RowWeights`通过同一合并后observer收集行身份；离线`train_rollout_policy_dilution.py::ImmutableDataCache`记录并原序转发observer，不另建source偏移。其余source_row命中是参数、计数报告、离线诊断/旧验收工具；`human_demo`的同名字段记录原始来源路径，不是按偏移读取训练包的入口。

**影响范围与修法：** rowid错误限于task162研究的卡组/缺帧mask、token行映射、流式留出token和探针配对；旧错误版本尚未产出可用研究模型。生产当前包、训练权重和r52晋升链没有这次错位的代码路径，故**无需改生产、重打包、重训或撤销r52**。今后若改研究来源布局，应沿用训练器的父目录排序语义；需要重建的是依赖错误行序的研究派生物，不应重写生产包。本次不添加任何一致性校验或新清单。
