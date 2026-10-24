# task61：让 qc 的后台训练装得进台式机——段合并逐段拷贝 + 内存估算按实测校准

> 主控 2026-10-24 11:15（台式机时间）写。执行者：台式机 CodeBuddy（`cb-qcalloc` 会话，接着 task59 做）。工作树 `/home/newbiexvwu/PvZ-search/.claude/worktrees/trainmem`（分支 `trainmem`，从 origin/pvz-env 拉出，主控已建好）。最多 1 个训练进程用于实测；**不碰 tmux `qc`、不改 cq 工作树、不部署**。**这是当前最急的任务**：qc 的学生更新整条链卡在这里。

## 为什么
- 11:06，round 2 后台打包完成：202 个标签，**14,821,421 行**，段目录 7.8 GiB。
- 训练随即卡在 `_admit()`，日志不断刷 `background waiting for memory: available=12–14 GiB required=31.20 GiB`。
  - 需要量来自 `estimate_peak() = 3 × 紧凑段字节 + 2 GiB`。
  - 台式机 WSL 一共只有 25.4 GiB，所以这个条件**永远满足不了**。
- 连带两个后果：
  - `reservation()` 每组从采集里扣 25 GiB 的后台余量，qc 前台只能开 1 个 worker；
  - 控制器退出时要等后台任务，于是停止标志也停不下来。主控已撤掉部署脚本，qc 先以 1 个 worker 维持。
- 实测参考：task23 的 dw4i 训练，1364 万行，子树 RSS 求和峰约 15.9 GB（含共享页），最低可用约 4 GB。所以 3 倍明显偏大，但即使按实测约 2 倍，也要 17 GiB 左右，加上 6 GiB 预留还是装不下。
- 根因在 `train_rollout_policy.py` 的 `read_segments()`：
  - 先把 202 个段全部读进内存（`parts`，约 1 倍紧凑语料）；
  - 再按全局档案顺序 `np.concatenate` 出一份新的（又是约 1 倍）。
  - 峰值是两份加模型。

## 要做的
1. **段合并改成逐段拷贝。**
   - 先只读各段的计数元数据（`source`、`source_archive`、`offsets` 等），按现有的全局档案顺序（`chunks` 排序）算出每个 chunk 在目标数组里的位置；
   - 预分配目标数组：CSR 的 `indices` / `values` / `offsets` / `plane_of`，以及 `join` 的各列、`sparse` 列；
   - 一次只载入一个段，把它的全部 chunk 拷到目标位置，然后释放。
   - 目标数组优先用现有的 `_spilled_array`（文件后备的 memmap，task32），让匿名内存接近"一个段 + 小量"；若训练随机取批时 memmap 明显变慢，用实测说明，再选普通数组（峰值约 1 倍语料 + 一个段）。
   - **合并结果必须与现行 `read_segments` 逐位相同**：写测试，用几个小段，比较每一列；再在 round 2 的真实 pack 上比较若干列的哈希。
2. **实测峰值**：
   - 用 round 2 的真实 pack 只做"读入 + 建 batch 若干步"，不必训完。pack 在 `/home/newbiexvwu/pvz-logs/curriculum_state.pipeline/frames/packs/0d95f2ff1c6803b6.npz`，段在同目录 `segments/`，**只读**。
   - 报改前 / 改后的峰值：进程 RSS 与 `MemAvailable` 最低值，以及读入耗时、每步耗时。
   - 改前那一次若内存不够，就只测改后，写清楚。开跑前 `MemAvailable` 至少要比估算峰值多 3 GiB，不够就等，不要挤爆 qc。
3. **估算按实测校准**：
   - `curriculum_queue.py` 的 `estimate_peak()`：系数用实测值（例如 紧凑字节 × 实测倍数 + 实测常数），写清楚依据；
   - 每次后台训练结束，把实测峰值记进 state，下次估算优先用最近的实测值；
   - **不可能满足的需求不能静默死等**：如果 `estimate + 预留 > MemTotal`，记一条 ERROR，并让该训练任务失败退出、写原因，不要卡住控制器退出。
   - 后台训练进行时，采集让路的规则保持现状（`reservation` 扣余量），只是余量要按新估算算。
4. 全部测试通过：`python/test_curriculum_*.py`、训练相关测试，以及新加的逐位比较测试。结果写本文件，**主控合并并部署**。

## 规则
- 只用绝对路径；不 `cd` 进别的工作树；`PVZ_NO_POPUP=1`；`OMP_NUM_THREADS` 按需。
- 禁止 `git add -f`；`.pt` `.npz` `.log` 不进 git；不写字面密钥；不手写打法规则或公式（内存估算系数来自实测，可以）。
- 不改训练的语义：同样的行、同样的顺序、同样的划分和洗牌。
- 每完成一小步就提交、`git push origin trainmem`，在"结果"记一行；疑问写"问题"，按最佳判断继续。

## 结果

- [x] 步骤 1（commit 78a69103）：read_segments 改两遍逐段拷贝。第一遍只读每行元数据与各成员
  npy 头（段是未压缩 np.savez，zipfile 读头不解压数据），按全局档案序预计算每个 chunk 的
  目标行位与各流条目前缀（board/scalar CSR 的 nnz 前缀；value/cand 段内没有 offsets 成员，
  由 counts 累积）；第二遍一次载入一个段，按位置拷贝后释放。目标数组在 data 给出时用
  _spilled_array 文件后备 memmap（匿名内存 ≈ 一个段），否则普通数组。**逐位相同**：40 段
  真实子集（2,560,231 行）逐列 sha256 与改前实现一致（heap 与 memmap 双模式），哈希常量
  固化在 python/test_incremental_pack.py::ROUND2_SUBSET_HASHES；合成语料逐列位比较
  （含 planes/plane_of/observer 行序）；既有 verify/assert_payload_equal 全过。
- [x] 步骤 2（实测峰值，round 2 真实 pack：202 段、14,821,421 行、紧凑段 7.73 GiB，只读）：
  - 改前全量**未跑**：需 ~2× 语料匿名 ≈ 16-18 GiB，超过"MemAvailable 比估算峰值多 3 GiB"
    的开跑线。子集外推：40 段峰值 3299 MB ≈ 2.3× 子集语料 → 全量 ≈ 16-17 GiB，与 task23
    独立实测（dw4i 1364 万行峰值 15.9 GB）吻合；旧 3× 公式给 31.2 GiB 确实永远装不下。
  - 改后 memmap（默认，data 给出）：读入 34.5 s，峰值 RSS 8.54 GiB（大头是可回收文件页），
    **MemAvailable 最低 15.66 GiB**，RssAnon 峰值 **1.59 GiB = 0.205×紧凑语料**；
    建批 256×50 步 p50 16.0 ms / p90 75.1 ms。
  - 改后 heap（对照）：读入 33.5 s，峰值 RSS 8.54 GiB（全匿名），MemAvailable 最低 9.5 GiB；
    建批 p50 5.7 ms / p90 18.0 ms。
  - **决定：训练走 memmap**。理由：匿名峰值 1.6 vs 8.5 GiB——heap 的估算+预留（6+9≈15 GiB）
    又逼近卡死线，而 memmap 的 required（6+4≈10 GiB）在 qc 同时采集时也总能满足；批构建
    慢 ~2.8×（p50 16 vs 5.7 ms）是可接受的代价，若实际训练中证明太慢，可选方案见"问题"。
- [x] 步骤 3（commit ba4b2e5f）：estimate_peak 按实测校准——训练口径改为**匿名内存**
  （语料数组是文件页，可回收不算需求）：优先用最近一次后台训练的实测 RssAnon 峰值
  （_command 每 2 s 轮询子进程树 RssAnon，写入 report 的 pack/train_peak_anon_bytes，
  随 trained.json/complete.json/state training_jobs timing 落盘，重启时读回）；没有实测时
  回退 0.25×紧凑段字节 + 2 GiB（实测 0.205×，留余量；常数覆盖模型/优化器/批次）。
  reservation/_admit 语义不变，只是余量按新估算算。**不可能满足的需求不静默死等**：
  _admit 开头检查 required（预留 6 GiB + estimate）> MemTotal 时直接 RuntimeError 带数字，
  该训练任务失败退出，poll() 变成控制器可见错误，停止标志与退出不再被卡。
  round 2 数值检验：required = 6 + max(打包, 0.25×7.73+2 ≈ 3.9) = 9.9 GiB < MemTotal 25.4
  ✓ 不再死等；实测全过程 MemAvailable 最低 15.66 GiB，qc 前台不受挤压。
- 全部测试 68 过 + 4 subtests（test_curriculum_queue/pipeline/incremental_pack/
  identity_training/value_widen_training/compact_loader/roof_panel）。

## 问题

- memmap 建批 p90 75 ms（写回高峰期），若训练耗时不可接受，可选：(a) 给 read_pack 加
  heap 开关（一行参数，实测数据齐）；(b) 训练前把段目录 cat 进 page cache 预热。
- estimate_peak 的回退常数（2 GiB）按 torch 模型/优化器的经验余量取值，未逐模型实测；
  首次真实训练后 state 里的实测值会自动取代它。
- 段目录 7.73 GiB 的 memmap 页在训练期间靠 page cache 命中；若采集把 cache 挤掉，
  建批会掉速但不失败——这也是"采集让路"仍按估算扣余量的原因。

## task61 追加（2026-10-24）：训练 45 分钟/epoch 的瓶颈定位与修复（branch `trainspeed`）

现象：round 2 后台训练（14,821,421 行，bs=2048，cuda，threads=1）epoch 1 用了 2747s
（≈45.8 分钟/epoch，12 epochs ≈ 9 小时）；task23 同量级语料只要 ~6 分钟/epoch。
训练进程 CPU 99.5%、GPU 5%、majflt 50k → 怀疑 memmap 随机 gather。

### 1. 实测（200 步 × 2 模式，真实 round 2 pack，`/tmp/task61_step_measure.py`）

| 模式 | load | build p50 | fwd/bwd p50 | 整步 | anon 峰值 |
|---|---|---|---|---|---|
| memmap（修复前） | 23.3s | 238.6 ms | 7.4 ms | 245.9 ms | 2.69 GiB |
| heap（修复前） | 19.7s | 224.1 ms | 7.5 ms | 231.6 ms | 9.33 GiB |

**heap 只快 6%** —— memmap 随机 gather 不是瓶颈（分解见下），"按 MemAvailable 自动选 heap"
用 9.3 GiB anon 换 6% 不值得，未采用；memmap 保持默认。

### 2. 分解定位（`/tmp/task61_decomp.py`，memmap）

boards.batch 222.2 ms 里 **per-identity plane 循环 218.4 ms（98%）**；
主 CSR scatter 2.0 ms、scalars.batch 1.2 ms、packet_info 0.9 ms、小张量 0.5 ms。
根因：round 2 有 **5155 个 plane**（每个归档一组常量格），2048 行随机批含
~1382 个不同身份 → 每步 ~1382 次（H2D 拷贝 + GPU index_put）微 kernel，纯启动延迟。
task23 当年语料归档数少，循环可忽略 —— 这就是 6 分钟 → 45 分钟的差值来源。

### 3. 修复（`CompactBoards.batch`，scripts/train_rollout_policy.py）

plane 展平一次（concat cells/values + offsets，round 2 共 1.5 MiB），按设备缓存；
每批一次融合 gather+scatter（ragged repeat 算位置，单次 index_put_）。语义不变：
每行恰属一个 plane、plane 内格唯一，与逐身份循环逐位等价
（`test_compact_boards.py::test_batch_with_planes_matches_dense` 按 int32 位型比对）。

| 模式（修复后） | build p50 | 整步 p50 | anon 峰值 |
|---|---|---|---|
| memmap | **12.1 ms** | **19.5 ms**（12.6×） | 2.69 GiB |
| heap | 9.9 ms | 17.4 ms | 9.33 GiB |

外推 6518 步/epoch ≈ **2.1 分钟**（原 45.8）。修复在 merge 模式之外，memmap 模式直接受益。

### 4. 重启语义（杀训练进程 + 控制器重启）

代码审读 + 单测（`test_curriculum_queue.py::TrainingRestartTest`）：
- `__init__` 把所有 status != 'done' 的训练任务重置 'queued' 并重新提交（实测 running/failed 重跑、done 不动）；
- `training.lock` flock：锁随活进程，旧子进程未死则新控制器阻塞，不会并发训同一快照；
- 训练完成（trained.json 已写）、评估前被杀：重启后校验 model/recipe sha 通过则**复用报告直接进评估，不重训**；
  sha 不符（半截文件/被改动）→ 拒绝复用，报错终止；
- 训练中途被杀（两个 json 都没有）→ 从头重训，无重复（什么都没完成过）。

### 5. 结论

- 已修复并合入 `trainspeed`（commit 3345e1cf）。生产 round 2 本次训练跑完后，
  下一轮起 epoch 应降到分钟级；无需改 recipe 或 merge 模式。
- heap 自动选择不做（数据不支持）；`--merge-mode` 开关也不加 —— 唯一瓶颈已消。
