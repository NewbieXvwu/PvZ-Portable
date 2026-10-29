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
