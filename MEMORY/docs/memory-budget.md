# 内存账本：学生训练、采集 worker、价值网络（task32，2026-10-22）

台式机 WSL 只有 28 GB，2026-10-22 21:18 被挤满到 `wsl.exe` 连不上。这份文档记的是：钱花在哪、
每一项改动省了多少、以及改完之后 1500 万行训练 + 16 个采集 worker + 价值网络怎么在 28 GB 里排。
所有数字都在 Mac（48 GB）上实测；外推到 1500 万行的地方会写明是按实测的每行字节数线性外推。

## 1. 现状账本（改前）

### 1.1 学生训练装载（`scripts/train_rollout_policy.py`）

生产配方（`--wait-keep .1 --obs v2 --packet-head identity --cell-soft-weight 3`），
200 档案 / 218 385 帧 / 61 903 行，逐数组 `nbytes` 记账（`scripts/measure_training_memory.py`，
明细见 `MEMORY/docs/task32-baseline.json`）：

| 组成部分 | 字节/行 | 1500 万行 |
|---|---:|---:|
| 棋盘 CSR（indices uint16 + values float16） | 382 | 5.34 GiB |
| 棋盘/标量行指针 | 8 | 0.11 GiB |
| 标量 CSR（indices uint16 + values float32） | 120 | 1.68 GiB |
| `packet_info`（identity，54×4 float32） | 160 | 2.24 GiB |
| 格候选 `cands`（float16 12×3） | 72 | 1.01 GiB |
| 目标 `values`（float32 55） | 44 | 0.62 GiB |
| keep 掩码（Python bool 列表） | 28 | 0.39 GiB |
| 标签 plant/packet/cell/weight | 24 | 0.34 GiB |
| seed_of + 索引张量 | 24 | 0.34 GiB |
| **合计** | **1024** | **14.32 GiB** |

### 1.2 引擎（`build/pvz-portable -env`，每个采集 worker 一个）

`vmmap` 物理占用 91.9 MB、峰值常驻 231.5 MB、启动 1.05 s。堆栈归因（`MallocStackLogging=lite` + `heap`）：
`main.pak` 私有副本 43.9 MB、`DefMapReadFromCache`（已编译动画定义全量预读）25.5 MB、
贴图像素（`LoadingImages`/`LoadingFonts` 资源组）约 60 MB。

### 1.3 采集 worker（Python 侧）

`scripts/measure_collection_memory.py`：解释器 23.0 MiB → 导入 `search_play` 24.4 MiB →
装载学生 + 价值网络（torch 导入）**205.0 MiB** → 开引擎并 reset 后本进程 210.3 MiB，
引擎子进程 57.2 MiB（改前 220 MB 量级）。**每个 worker 约 425 MiB**，16 个 6.6 GB。

torch 的 180 MiB 在 fork 模式下是全 worker 共享的（父进程先装载，子进程 COW），
所以 16 个 worker 的 Python 侧实际只多付各自的堆，不是 16×205 MiB。

### 1.4 价值网络训练（`scripts/train_value_net.py`）

训练棋盘按样本存稠密 `(92, 6, 10) float16` = **11 040 B/样本**；16.6 GB 被 OOM 杀就是这一项。

## 2. 每项改动省多少

### 2.1 学生训练：紧凑行存储（唯一实现，无开关）

| 改动 | 省 |
|---|---|
| keep 掩码位打包（`np.packbits`） | 28 → 3.5 B/行 |
| 标签 dtype（plant uint8、packet/cell int16） | 24 → 7 B/行 |
| `PacketStore` 按列拆 kind int8 / active uint8 / progress float32 / affordable uint8 | 160 → 70 B/行 |
| 棋盘索引按格子数降到 uint8（92×6×10 = 5520 格）、标量索引 uint8 | 382 → 361 B/行（含下面平面） |
| 每档案恒定平面（可通行车道那 50/134 个非零） | 棋盘非零 134 → 84/行 |
| 索引张量 int32、`values` 一次预分配 | 小项 |

格候选同样只存排名靠前的几个（`CandidateStore`：uint8 计数 + 6 B/候选）：**72 → 25.9 B/行**。

目标 `values` 同样只存非 NaN 的候选（`ValueStore`）：**44 → 17.6 B/行**。

结果：**1024 → 593 B/行，14.32 → 8.29 GiB**（1500 万行）。实测 66 993 行
`data_memory` = 棋盘 24.18 + 标量 7.25 + packet 4.69 + 其它张量 0.67 + 格候选 1.73 + 目标 1.18 = 39.70 MB。
验证集分块从 4096 降到 1024，训练进程峰值 **2.43 → 1.62 GB**（-33%，权重逐位相同；
`val_loss` 是「各块均值按行数加权」，本来就随分块变，数字不可跨这个提交比较）。
门槛：同数据同种子，改前（`--compact-data`）与改后 21 个权重张量 sha256 **全同**；
峰值 2.457 → 2.423 GB，耗时 34 s 不变。

`--data-memmap`（把大数组写文件再只读映射）**没有保留**：2000 档案实测它只把峰值从
2.65 降到 2.64 GB，却要在临时盘上占 8.3 GB；28 GB 预算里紧凑行放得下。

### 2.2 引擎：无界面模式只读尺寸 + pak 映射 + 动画定义按需加载

| 改动 | 效果 |
|---|---|
| 图片只解析文件头取宽高（PNG IHDR / GIF / JPEG SOF / TGA），不分配像素 | 贴图像素归零；回退顺序与完整加载一致（`IMAGE_BLANK` 只有 `images/blank_.gif`） |
| `main.pak` 改 `mmap` 只读映射，XOR 改在 `ReadAt` 里按读取解 | 每进程少 43.9 MB 私有副本；16 个进程共享同一份页缓存 |
| 已编译动画定义不再全量预读，第一次用到时加载 | 每进程少约 25 MB |

| 指标 | 改前 | 改后 |
|---|---:|---:|
| 峰值常驻（`ru_maxrss`） | 231.5 MB | **60.0 MB** |
| 空闲物理占用（`vmmap`） | 91.9 MB | **21.4 MB** |
| 启动到 reset 完成 | 1.05 s | **0.31 s** |

验收：同一动作序列重放，第 7 关 300 决策（到第 6 波）+ 17/27/47 关各 150 决策，
两版 `frames` 摘要**逐字节相同**；同种子 64 局、每局 400 决策、学生自己下（最深到第 13 波，
没有一局在 400 决策内结束，所以是深局不是短局），两版**每局逐帧摘要都相同**。
64 局上：峰值常驻 222.9 → 60.6 MB，启动 1.048 → 0.343 s，总耗时 89.8 → 44.2 s。

### 2.3 采集 worker

引擎 220 → 60 MB 是这一项的全部收益：**每 worker 425 → 268 MiB**，16 个 6.8 → 4.3 GB。
Python 侧 205 MiB 里 180 MiB 是 torch 本身，fork 后共享，没有再压的空间（除非去掉 torch）。

### 2.4 价值网络

训练棋盘改走同一套 CSR 行存储：**11 040 → 约 330 B/样本（21 倍）**；
`train_value_branches.py` 合并多份 dump 时写文件再映射，不再 `torch.cat` 复制一份。

## 3. 28 GB 里怎么排

| 用途 | 改前 | 改后 |
|---|---:|---:|
| 学生训练（1500 万行数据 + torch/训练现场） | 14.3 + 2.5 = 16.8 GiB | 8.3 + 2.5 = **10.8 GiB** |
| 16 个采集 worker（Python 共享父进程 + 16 个引擎） | 0.2 + 16×0.22 = 3.7 GiB | 0.2 + 16×0.06 = **1.2 GiB** |
| 价值网络训练（补采 dump + 训练现场） | 2–4 GiB | **1–2 GiB** |
| 合计 | 23–25 GiB（贴顶，2026-10-22 就是这样挤爆的） | **13–14 GiB** |

余下约 14 GB 给操作系统页缓存、CUDA rollout 服务和偶发的第二批采集。

## 4. 复现

```bash
# 学生训练逐组件记账（外推到 1500 万行）
python3 scripts/measure_training_memory.py --data artifacts/hard_mac/frames \
  --labels <label,...> --cache-dir /tmp/cache --archives 200 --json out.json

# 采集 worker 的常驻构成（Python + 引擎）
python3 scripts/measure_collection_memory.py

# 引擎：同一动作序列 / 整局学生重放，逐帧摘要 + 峰值 + 启动
PVZ_NO_POPUP=1 python3 scripts/engine_replay_compare.py \
  --executable <旧二进制> --executable build/pvz-portable --decisions 300
PVZ_NO_POPUP=1 python3 scripts/engine_replay_compare.py \
  --executable <旧二进制> --executable build/pvz-portable \
  --student artifacts/student/v25/model.pt --games 64 --decisions 400
```


## 5. 当前分布（前 15 项及占比，2026-10-23）

### 5.1 学生训练数据（593 B/行，1500 万行 8.29 GiB）

| 项 | B/行 | 占比 | 为什么不能再压 |
|---|---:|---:|---|
| 棋盘 CSR（uint16 索引 + float16 值，87 非零/行） | 361 | 60.9% | 每档案去重后仍有 468–629 个不同格子（>256），uint8 索引放不下；索引值已是 float16 原值 |
| 标量 CSR（uint8 索引 + float32 值，20 非零/行） | 108 | 18.2% | 实测只有 41% 的非零能无损装进 float16（11/28 个通道全通道可装，占 15% 的非零），收益 <2% |
| packet 分列（int8/uint8/float32/uint8） | 70 | 11.8% | progress 列只有 62% 能无损 float16 |
| 格候选（稀疏） | 26 | 4.4% | 已只存排名靠前的候选 |
| 目标值（稀疏） | 17.6 | 3.0% | 已只存非 NaN |
| plant/packet/cell/weight/is_val | 10 | 1.7% | uint8/int16 |

### 5.2 引擎（malloc 7.9 MB，物理占用 21 MB）

| 项 | MB | 占 malloc |
|---|---:|---:|
| `DefMapReadFromCache`（本关真正用到的动画定义） | 4.08 | 51.9% |
| `SingleDataElement`（定义树节点） | 0.97 | 12.4% |
| pak 记录表（3195 个文件名） | 0.48 | 6.1% |
| `ParseCommonResource`（资源表字符串） | 0.34 | 4.3% |
| `FontLayer::GetCharData`（字形度量） | 0.31 | 4.0% |
| `ImageRes`（837 个资源描述） | 0.26 | 3.3% |
| `ListDataElement` | 0.24 | 3.1% |
| `vector<unique_ptr<DataElement>>` 节点 | 0.24 | 3.1% |
| `std::string` 赋值 | 0.23 | 2.9% |
| `ImageFont::GenerateActiveFontLayers` | 0.18 | 2.3% |
| `SexyAppBase::SetString` | 0.16 | 2.1% |
| `GetSharedImage`（共享图片表） | 0.09 | 1.1% |
| `FloatTrackSetDefault`（动画轨道） | 0.07 | 0.9% |
| `AddMemoryImage` | 0.02 | 0.3% |
| 字符串拷贝构造 | 0.03 | 0.3% |

### 5.3 停止条件：连续三项 < 2%

| 尝试 | 实测收益 | 结论 |
|---|---:|---|
| 格子头梯度检查点（`torch.utils.checkpoint`） | 0%（1619 vs 1621 MB） | 峰值不在被检查点的张量上，已回退 |
| 标量按通道降到 float16 | 1%（15% 的非零可无损） | 不划算 |
| packet progress 降到 float16 | 0%（62% 可无损，但必须 100%） | 不划算 |

前 15 项里没有可证明的冗余：棋盘/标量/packet 都已到「索引 1–2 B + 值 2–4 B × 非零数」的地板，
引擎的 4 MB 是本关实际在用的动画定义。再往下要么改训练配方（batch、精度），要么改标签含义，
都不属于「内存优化」。

## 6. 还剩什么（没做，按收益排序）

1. **Python 侧 205 MiB/worker 里 180 MiB 是 torch**：只有 fork 共享能压，Windows 上无解。
2. **引擎 21 MB 里还有 `PoolEffect`/`ZenGarden`/`ReanimatorCache` 的初始化**（约 0.4 MB）：
   与贴图相比不值得改，且 `PoolEffect` 在 `Board::Update` 里每帧建粒子，要跳过它得动更新循环。
3. **训练现场的 1.6 GB**（torch + 一次 2048 行前向/反向的激活）没动：它不随数据量增长，
   再压要改 batch 或做检查点，前者改配方、后者实测无收益（见 5.3）。

## 7. 磁盘存量（档案与日志，2026-10-23）

内存账本之外的存量账。两项都**不丢字节**：转换后的 npz 里存着原始 JSON 文本，轮转后的
`.gz` + 明文尾巴拼起来就是完整日志。

### 7.1 旧 JSONL 档案 → frames.npz

```bash
# Mac（全量，先 --dry-run 看清单）
python3 scripts/convert_frames_npz.py --data /Users/newbiexvwu/PvZAgent/artifacts --jobs 4 --dry-run
python3 scripts/convert_frames_npz.py --data /Users/newbiexvwu/PvZAgent/artifacts --jobs 4 --delete \
    --report MEMORY/docs/frames-npz-report.json --manifest MEMORY/docs/frames-npz-converted.txt.gz

# 台式机（WSL）：同一条命令，换 --data
python3 scripts/convert_frames_npz.py --data ~/PvZAgent/artifacts --jobs 4 --dry-run
```

删原文件前逐档过五道校验：npz 里存的字节与 JSONL 原文逐字节相同、每一帧解析出的 dict 全等、
行索引恰好等于有标签且未被拒的帧号、`encoder` 签名是当前版本、六个数组的行数与行索引一致。
默认跳过 60 分钟内改过的档案（不碰正在写的采集）。`--delete` 只删已通过校验的那些，
每个删掉的文件（原大小 / npz 大小 / 状态 / 路径）记在 manifest 里。

### 7.2 日志轮转

```bash
# Mac
python3 scripts/rotate_logs.py --root /Users/newbiexvwu/PvZAgent/artifacts --dry-run
python3 scripts/rotate_logs.py --root /Users/newbiexvwu/PvZAgent/artifacts

# 台式机（WSL）
python3 scripts/rotate_logs.py --root ~/PvZAgent/artifacts
```

默认 `--min-mb 2 --keep-mb 1`：超过 2 MB 的日志把"除最后 1 MB 之外"的部分压进
`<name>.log.gz`（每次轮转追加一个 gzip member，所以长期写的日志最终是一个 .gz），
最后 1 MB 留明文给 `tail`/`grep`，切点对齐到行首。改过 120 分钟内的文件跳过（可能还在写）。
2026-10-23 实测：104 个文件 463.3 MB → 明文 109.0 MB + gz 28.8 MB，**省 325.5 MB（70%）**，
9 个正在写的跳过。

**2026-10-23 实测（Mac，`--jobs 4 --delete`）**：6 658 个档案（另有 1 个教师诊断档在第 13 步冒烟时已转换删除，
共 6 659）全部通过五道校验后删原文件，**失败 0**；7 509 703 263 B JSONL（7.00 GiB）→ 563 481 079 B npz（537 MiB），
**7.50%**，省 6.95 GB，耗时 196 s。`artifacts` 从 11 GB 降到 6.2 GB（含 §7.2 的日志轮转）。转换后 Mac 上
`frames.jsonl` 剩 0 个；读取端实测：`read_frames(npz)` 与原文逐帧全等（577 帧样本）、`frame_path` 优先选 npz、
`cache_episode` 直接复用 npz 里的行缓存不再重算。逐档清单（原大小 / npz 大小 / 状态 / 路径）在
`MEMORY/docs/frames-npz-converted.txt.gz`（6 658 行，sha256 见 `MEMORY/docs/frames-npz-report.json`）。
台式机尚未转换，命令同上。

### 7.3 档案容器：zlib → LZMA

`python/pvz_frame_store.py:_save` 现在写 `ZIP_LZMA`（原来 `np.savez_compressed` 是 zlib）。
一局档案里 72% 的字节是 frames JSON（每帧一行状态），LZMA 能跨整局建模重复，12 档样本上
**235 KB → 140 KB（59.6%）**。`np.load` 不用改（zipfile 解码），训练包保持 `np.savez` 的
`ZIP_STORED`（`read_pack` 要 `mmap_mode`），`python/test_frame_store_container.py` 钉住这两条。

```bash
# 存量一次性重压（逐档把所有数组读出来重写，逐个数组 dtype/形状/值全等才原子替换）
python3 scripts/convert_frames_npz.py --data /Users/newbiexvwu/PvZAgent/artifacts --recompress --jobs 8
```

**2026-10-23 实测**：30 120 个 npz 里 22 205 个被重压（7 915 个跳过：训练包是 ZIP_STORED、
已重压过的、120 分钟内改过的），**5 195 921 622 B → 3 105 601 019 B（59.8%），省 2.09 GB，失败 0**，
8 并发 1 970 s。代价：写一局档案 0.055 → 0.63 s、读一档 4.4 → 12 ms（一局游戏几十秒到几分钟）。
`artifacts` 总计 **11 GB → 4.3 GB**（7.1 转换 + 7.2 日志 + 7.3 LZMA）。
