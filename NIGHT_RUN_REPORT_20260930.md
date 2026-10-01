# 台式机测量与 cap1 seed-0 执行报告（2026-09-30）

A1、A2 已完成并保留原始数字。首次 B0 因工作区代码状态中的缺失文件引用在 2.37 秒后退出；用户修复后，以干净 HEAD 重跑 B0 并完成评估，然后启动 B1。B1 的 seed-0 候选在 5,000 局触发训练器内置 `stage0_no_signal`，按停止条件结束。当前观察未显示该 seed/config 在这 5,000 局内学到胜利信号，不能据此推断所有 RL 配置都不可学。后面的“首次中断现场”保留故障原文；“续跑及最终结果”是本报告当前的最终状态。

## 代码与环境

- 测量/冒烟代码 HEAD：`542ad5a289fda8b65d9415055512e8b4c18520c7`（短号 `542ad5a`），分支 `pvz-env`。
- `git pull --ff-only origin pvz-env` 成功，输出 `Already up to date.`，未发生覆盖冲突。
- 开工前已有两处未提交代码改动：`python/train_pvz_ppo_task_family.py`、`scripts/reachability_audit.py`。本次保留它们；逐文件 SHA-256 核对确认执行期间 `python/`、`scripts/` 中受跟踪代码未改变。故结果绑定 HEAD 和这两处既有改动，不能声称工作区干净。
- Python：`/home/newbiexvwu/.venvs/ml/bin/python`，3.14.4；torch 2.14.0+cu132；CUDA runtime 13.2；`cuda_available=True`。
- GPU：NVIDIA GeForce RTX 5080，16,303 MiB，计算能力 12.0。`nvidia-smi` 报 NVIDIA-SMI 615.78.02 / KMD 617.14 / CUDA UMD 13.4。
- WSL：Ubuntu，内核 6.18.33.2-microsoft-standard-WSL2；`nproc=20`。`free -g`：内存 total 23 / available 21 GiB，swap 8 GiB；磁盘可用约 930 GiB。本次可见内存高于交接文档中的旧 16 GB 描述，以实测为准。
- 资源：`/home/newbiexvwu/.cache/pvz-research-resources`；模拟器 `build/pvz-portable` 存在。输入文件指纹保留在 `logs/input_fingerprints.json`。
- 环境原件：`logs/env.txt`；随 git 交付的同内容副本：[artifacts/t5/perf/night_env.txt](artifacts/t5/perf/night_env.txt)。
- 已读 README 末尾、TODO §7.2 / §9 和 DESKTOP_HANDOFF §9 全部条款。此 HEAD 的 README 末尾没有所述文档导航表，实际末尾为 Thanks；本次未修改文档导航。

## A1：三点墙钟拟合

全部使用 `cpu:18x1`，episode 数分别为 180、720、2,000；任务集、随机种子和默认 `max_actions=4000` 均由既有基准入口决定。仅追加 `--keep-shards` 保留现场，未更改源代码或添加预热轮次。

| 局数 | 基准 pool 墙钟（s） | 整条命令墙钟（s） | 进程树 RSS 峰值（GiB） |
|---:|---:|---:|---:|
| 180 | 23.870 | 27.87 | 21.29 |
| 720 | 51.609 | 54.95 | 22.03 |
| 2,000 | 125.867 | 128.56 | 24.42 |

以 JSON 内 `_measure` 对 `run_seed_jobs` 的墙钟作最小二乘拟合：

`T(N) = 12.596720 s + 0.056398566 s/局 × N`

- 斜率 **56.399 ms/局**；截距 **12.597 s**；R²=0.999276。
- 截距占实测 2,000 局批次 125.867 s 的 **10.01%**；占拟合批次 125.394 s 的 10.05%。
- 本次三点没有复现 5.5 s。该截距是基准路径固定开销的估计，包含启动、初始化、任务调度与收尾，不能单独归因于 pool 初始化，也不据此提出实现改动。
- 另以 `/usr/bin/time` 整条命令墙钟拟合：斜率 55.719 ms/局、截距 16.599 s。该口径还包含 Python 导入及父进程模型准备，故单独报告，不混用。

原始证据：[180](artifacts/t5/perf/pool_intercept_180.json)、[720](artifacts/t5/perf/pool_intercept_720.json)、[2000](artifacts/t5/perf/pool_intercept_2000.json)；计算和残差：[pool_intercept_fit.json](artifacts/t5/perf/pool_intercept_fit.json)。

## A2：并行画像与 worker 选择

每档 500 局，同一轮基准中按 1、8、12、18 worker 顺序执行。争抢放大倍数定义为 `(本轮单 worker 局/小时 × N) / 本轮 N worker 实测局/小时`，包含固定开销。

| 配置 | pool 墙钟（s） | 局/小时 | RSS 峰值（MiB / GiB） | 争抢放大倍数 |
|---|---:|---:|---:|---:|
| cpu:1x1 | 257.666 | 6,985.8 | 2,860.8 / 2.79 | 1.000 |
| cpu:8x1 | 48.657 | 36,993.7 | 10,982.6 / 10.73 | 1.511 |
| cpu:12x1 | 41.120 | 43,774.2 | 15,503.6 / 15.14 | 1.915 |
| cpu:18x1 | 37.726 | 47,713.0 | 22,488.5 / 21.96 | 2.635 |

**下半夜选择 12 worker，每 worker 1 torch thread、CPU rollout。** 12 worker 的 15.14 GiB RSS 比 18 worker 少 6.82 GiB，吞吐保留 91.74%；18 worker 的 2,000 局批次达到 24.42 GiB 聚合 RSS。结合初始约 21 GiB 可用内存，选择 12 为训练主进程、CUDA context 和较大 rollout 批次留余量。500 局画像不能证明 2,000 局训练峰值，原计划还需冒烟验证，但 B0 在采样前失败，故训练内存尚未测到。

RSS 合计会重复计入共享页；7 次整机内存观测中最低 MemAvailable 约 4.71 GiB，swap 使用均为零。这些离散采样不是整机连续峰值测量。基准入口自身每 0.25 s 测得的进程树 RSS 峰值见上表。

仅 workers=1 的口径可跨机器比较。本报告未将任何多 worker 吞吐与旧记录或其它机器比较。原始基准 JSON 的自动最快选择仍保留为 18；执行选择 12 单独记录，未改原始测量。

证据：[parallel_profile.json](artifacts/t5/perf/parallel_profile.json)、[parallel_profile_analysis.json](artifacts/t5/perf/parallel_profile_analysis.json)。原有 `artifacts/t5/throughput.json` 未改动，未变更其门槛或放行字段。

## 首次 B0 中断时的状态（修复前；历史现场）

> 本节只记录首次失败状态；成功重跑和 B1 的最终结果见文末“续跑及最终结果”。

## B：冒烟与逐段结果

B0 使用用户要求的 `cap1`、初始化 seed 0、rollout 200、最多 1,000 局、指定 initialization-note 和本轮选择的 12 worker；外层设置 30 分钟 timeout，实际 2.37 s 即以退出码 1 结束。未使用门禁忽略参数。

| 阶段 | 实际局数 | 实测命令耗时 | 最近 policy_loss / value_loss | 评估胜率与 95% Wilson 区间 |
|---|---:|---:|---|---|
| B0 冒烟 | 0 | 2.37 s | 无（未更新） | 无（n=0，未评估，区间不可计算） |
| B1 第 1–6 段 | 未启动 | 无 | 无 | 无 |

检查结果：

- `artifacts/t5/night_smoke/runs/run_1/gameplay_model_v1_ppo.pt`：未生成。
- `artifacts/t5/night_smoke/learning_curve.json`：未生成。
- 冒烟评估：未执行；原始逐种子结果未生成。
- `artifacts/t5/night_smoke/training_state.json`：未生成。
- `artifacts/t5/night_seed0/` 长跑产物：未生成。

任何一项未出就停止，故没有放大训练。冒烟没有完成一局，无法以冒烟实测推算整夜局数；未用 A2 的纯 rollout 吞吐替代训练端到端吞吐。B1 没有启动，不能给出逐段 loss 或胜率。学习曲线不存在，“是否上升”为无法判定；未评估不等于胜率 0%。运行不足 30 分钟，未到长跑每 30 分钟进度记录周期；停止状态写入 `logs/progress.log`。

## 异常原文及现场

启动前的既有 trainer 改动把 `FAILURE_ANALYSIS.md` 加入 protected_paths，但仓库中该文件不存在。B0 在计算受保护文件指纹时失败，尚未写训练状态或建立训练 run。

```text
2026-09-30T19:59:43+08:00 B0 START workers=12 rollout_episodes=200 max_episodes=1000
Traceback (most recent call last):
  File "/home/newbiexvwu/PvZ-Portable/python/train_pvz_ppo_task_family.py", line 1225, in <module>
    main()
    ~~~~^^
  File "/home/newbiexvwu/PvZ-Portable/python/train_pvz_ppo_task_family.py", line 752, in main
    protected_before = {path: sha256_file(path) for path in protected_paths}
                              ~~~~~~~~~~~^^^^^^
  File "/home/newbiexvwu/PvZ-Portable/python/pvz_common.py", line 39, in sha256_file
    with path.open("rb") as file:
         ~~~~~~~~~^^^^^^
  File "/usr/lib/python3.14/pathlib/__init__.py", line 772, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/newbiexvwu/PvZ-Portable/FAILURE_ANALYSIS.md'
2026-09-30T19:59:46+08:00 B0 END exit_code=1
```

既有改动完整 diff 在 `logs/preexisting_code_changes.patch`；代码完整性核对在 `logs/code_integrity.json`。本次未修复或回退任何 `python/` / `scripts/` 代码，未补造缺失文档，未删失败种子或改评估/任务/阈值。未触发阶段 0 门禁拒绝；未绕过任何门禁。完成的基准没有观察到 OOM、worker 死亡或停滞。

全部 4,900 个测量分片留在 `artifacts/t5/night_measurement_shards/`，总计约 595.4 MiB；冒烟目录和全部日志保留。训练检查点、曲线与 training_state 在失败前未产生，不能作为已交付项。

## 交付位置与推送范围

- git 入库：A1/A2 原始 JSON、拟合/选择/执行摘要、环境文本副本及本报告。执行摘要：[night_execution_summary.json](artifacts/t5/perf/night_execution_summary.json)，内含原始冒烟错误与停止原因。
- 本机日志：`logs/env.txt`、`logs/git_pull.log`、`logs/perf.log`、`logs/perf_wallclock.txt`、`logs/smoke.log`、`logs/smoke_wallclock.txt`、`logs/smoke_exit_code.txt`、`logs/smoke_validation.json`、`logs/progress.log`、`logs/memory_samples.jsonl`、代码 diff/指纹/完整性记录和 git 收尾日志。
- 按交接文档与既有 .gitignore，原始 `.log` 和分片保留执行机，不强制入库；环境和失败原文已另存可推送证据。没有可复制到 curves 的训练曲线，未创建虚假的曲线文件。
- 交付提交 SHA 保存在本机 `logs/delivery_head.txt`，最终 push 结果见 `logs/git_push.log`；本报告的代码 HEAD 是产生测量的版本。


## 续跑及最终结果（2026-09-30）

用户修复问题后，当前工作区处于干净的 `b09d2b2`；训练入口的 `protected_paths` 与 HEAD 一致，不再引用缺失的 `FAILURE_ANALYSIS.md`。原先 2.37 秒失败的 traceback 保留在 `logs/smoke.log`，没有删除。

B0 第二次运行使用原定参数和 12 worker，进程退出码 0，墙钟 **385.22 秒**，完成 **1,000 个训练局**。检查点 `artifacts/t5/night_smoke/runs/run_1/gameplay_model_v1_ppo.pt` 存在，曲线含起点和训练后评估，1,280 条评估任务完成。训练后 held-out cap3 ×1.0 为 **33/640 = 5.16%**（95% Wilson **3.69%–7.15%**）；stage0 cap1 为 **24/320 = 7.50%**（95% Wilson **5.09%–10.92%**）。全冒烟吞吐为 **9,345 训练局/小时**；最后一次 rollout + PPO update 的口径为 **22,277 局/小时**。前者含启动和最终评估，按六小时粗略外推约 **56,100 局**；后者不含评估，仅表示训练更新速度。训练 rollout 本身最近一批为 0/200 胜；该入口未重测新架构的未训练策略，曲线零点沿用 T4 门禁资料，故冒烟率上升不能单独证明策略学习。

B1 分段情况如下。第 1 段计划 8,000 局，实际在 **5,000 局、660.58 秒** 后由现有训练器的停止条件终止。共完成 3 次更新：

| 段 / 更新 | 实际训练局数 | policy_loss | value_loss | rollout / update 墙钟 |
|---|---:|---:|---:|---:|
| 第 1 段 / 1 | 2,000 | 0.1283 | 0.0201 | 121.9 / 91.1 s |
| 第 1 段 / 2 | 2,000 | −0.0308 | 0.0023 | 124.8 / 87.9 s |
| 第 1 段 / 3 | 1,000 | −0.0011 | 0.0017 | 69.0 / 46.5 s |

训练 rollout 胜数为 **0/5,000**。5,000 局最终评估：held-out cap3 ×1.0 为 **0/640**，95% Wilson 区间 **0%–0.60%**；stage0 cap1 为 **0/320**，区间 **0%–1.19%**；参考 cap3 任务为 **0/960**。学习曲线在 0 和 5,000 局的 held-out pass rate 均为 0；曲线上升为 **否**。当前训练器状态是 `stage0_no_signal`，自动生成的 `gates/T5.json` 为 `fail`。

该停止条件明确阻止在同样的零胜信号下继续加样本；因此第 2–6 段没有启动，也没有忽略门禁。结论限于一个初始化种子、一种结构和 cap1 课程：**这轮没有显示 RL 学到胜利策略的证据；它尚不能回答 RL 在此环境中普遍能否学会。** 没有删种子、修改任务集或改变阈值。

内存观测共 12 次，可用内存最低约 5.72 GiB，swap 使用 0；单个训练主进程 RSS 峰值约 5.26 GiB。没有 OOM、worker 死亡或采样停滞。冒烟与长跑日志分别出现 72 条和 60 条 `INFO: RegEmu: Couldn't open '/tmp/pvz-env-…/registry.regemu' for writing`。评估任务仍全部返回、两次训练命令均以 0 退出；保留该 warning 原文，不推测其成因。

[续跑证据摘要](artifacts/t5/perf/continuation_after_repair.json) 包含 B0/B1 原始胜数、Wilson 区间、loss、内存和停止原因。最终曲线已复制到 [night_seed0 曲线](artifacts/t5/curves/night_seed0.json)。
训练状态 `artifacts/t5/night_seed0/training_state.json`、全部分片及 `logs/` 留在执行机（本机没有）。
检查点 `artifacts/t5/night_seed0/runs/run_1/gameplay_model_v1_ppo.pt`（14 MB）**当时被提交进了 git** ——
这是 2026-10-01 审计认定的违规项，后续检查点一律不进 git（见 [AGENTS.md](AGENTS.md) §2）。
`gates/T5.json` 是训练器自动生成的失败结果，和性能证据及曲线一起推送。
