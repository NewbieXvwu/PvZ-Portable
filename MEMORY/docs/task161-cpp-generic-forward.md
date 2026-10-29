# task161：C++ 通用前向模块（按配置组装）

分支 `cppnet`，基于 `origin/prodint16`。在台式机做（Codex business 账号）。

## 背景

用户决定：生产里的网络前向能用 C++ 就用 C++。采集、续走、门禁都在 C++ 引擎里跑。

下一代学生的结构还在决策中（见 task157 学生 v3、task158 地形 MoE、task160 决策研究），候选有：
- 现行 CNN；
- 加宽 / 加深的 ResNet；
- 实体 / token 注意力（含关系偏置）；
- CNN + 实体混合（scatter）；
- 各种 MoE。

为了避免每换一次结构就重写一遍 C++，先把一套**按配置组装的通用前向**做好。现行导出格式是 PVZRN002–006，C++ 前向在 `src/EnvironmentRollout.cpp`。task153 实测：现行 CNN 在 C++ 中约 61 µs/次；一局根节点与续走的调用比约 1 : 132。

## 要做的

1. **模块库**（C++，CPU，单线程，batch 1 为主，也支持小批）：
   - 卷积（含 1×k）；
   - 行池化 / 全局 mean·max 池化；
   - Linear、LayerNorm / RMSNorm；
   - 多头注意力，带按头查表的关系偏置：种类对、行差桶、列差桶、同格，并支持掩码；
   - SwiGLU / ReLU FFN；
   - MoE FFN，两种路由：
     - 共享专家 + top-k 路由；
     - 按外部 id 硬路由（地形）；
   - scatter：实体特征按 (行, 列) 加回格子；
   - 指针头：query · key 给卡包 / 格子打分；
   - 现有三头（种 / 等、卡包、格子）的输出约定。
2. **自描述权重格式**：文件头写明模块图与超参，C++ 按描述组装。Python 侧写导出器：给定 `nn.Module` 和图描述，导出可被 C++ 读的文件。现行 RolloutNet 也能用这套格式表达，用来对拍旧路径。
3. **对拍**：Python 与 C++ 在随机输入和真实帧上逐项误差 < 1e-5；现行 RolloutNet 用新格式时，动作与 PVZRN006 路径一致。
4. **性能**：报告各示例结构的单次前向 µs，与现行 61 µs 对比；用编译器向量化、预打包权重、避免分配，必要时可选 int8。示例结构至少包括：
   - 现行 CNN64；
   - CNN128；
   - 6 块残差 CNN；
   - token 注意力：2 层和 4 层，宽 128，token 数 64 和 128；
   - 混合结构；
   - 5 专家硬路由 MoE。
5. 接入点：引擎能按配置文件加载这种模型，作为根节点策略或续走策略。旧格式保持可用，默认不变。

## 约束

- 台式机是生产机：
  - 编译用独立 build 目录，Nice=10，并行 ≤ 4；
  - 测试只跑 `-k`，加 `ulimit -v 8000000`，先看 `free -g`；
  - 不碰 qc / 看门狗 / `stop_curr` / 生产二进制，不 `pkill -f` 含 `curriculum_queue` 的模式。
- 不加自校验。分步提交推送 `cppnet`，结论写在本文件末尾。
- 与 task157（Mac codex-a，分支 studentv3）协调：它负责富输入与网络移植，本任务只负责前向库。接口说明写清楚，方便它接。

## 阶段 1：通用模块与导出接口（2026-10-29）

代码入口为 `src/GenericForward.{h,cpp}`、`python/pvz_generic_forward.py`。
`GraphSpec` 显式列出输入、权重、节点、各输出形状；`export_graph(nn.Module, spec, path)`
按 `state_dict` 名字导出引用的参数，不依赖 tracing、TorchScript、ONNX 或 Python 运行时。
`rollout_graph(net, batch=1)` 可表达现行 RolloutNet 的全部原生观测版本、两种卡包头、行池化。
`python/pvz_generic_examples.py` 提供 CNN64/128、6 块残差、2/4 层 token、混合和 5 专家示例。

### 给 task157 的接入约定

- 所有张量连续 float32，离散 id 也用 float32 表示整数。卷积为 NCHW，token 为 `[B,T,D]`。
  图固定形状（包括 batch）；小批用同一模块另导出 B=2/4 的图。每个 worker 独占 Graph 与 scratch。
- `Graph::Load(path)` 一次组装图与缓冲，`Get(name).data` 绑定输入，`Run(firstStage,lastStage)` 前向。
  stage 0 编码、1 种/等与卡包、2 给选中卡包的格子；stage 2 不得更改编码，编码只执行一次。
- 三头名字与形状：`plant:[B,1]` 为种/等原始 logit（>0 种），`packet:[B,10]` 原始卡包 logits，
  `cell:[B,54]` 按 row*9+col 的原始格子 logits。`selected_packet:[B]` 输入为槽位 id。
  legality 与 identity 卡包平局规则继续由引擎动作选择器负责。
- 富输入的字段、可见性与 token 生成由 task157 负责。本库不推断 token 的游戏语义，也不改采集协议。
  新的原生输入适配器只需把数组填进 Graph 输入；编码器、注意力和头由图表达。
- `relation` 的输入元数据 `[B,T,3]` 为 kind,row,col。kind 查 `[H,K,K]` 表；行差/列差按 query-key
  差截断到 `[-R,R]` / `[-C,C]`，分别查 `[H,2R+2]` / `[H,2C+2]` 表；最后一桶为未知坐标。
  同格偏置为 `[H]`，只对两坐标均有效的同格 token 相加。额外关系或逐对掩码可用 `add` 加到 bias。
- `attention` 接收 q,k,v:`[B,T,D]`、关系 bias:`[B,H,T,T]`、valid:`[B,T]`；有 1/sqrt(D/H)
  缩放。valid<=0 的 key 不参加 softmax，query 输出零；整批无有效 key 时输出零。
  逐对 mask 用 bias=-inf。`ffn` 激活 0=ReLU，1=SwiGLU（up 输出前半是 gate、后半是 value）。
- MoE 路由输入是外部 logits 或外部 expert id。top-k 对所选 logits 做 softmax 并归一化；平局按专家 id
  升序。共享专家无 gate，直接求和；硬路由取一个专家，支持每个 token 不同 id。
- `scatter` 将 `[B,T,C]` 加到 NCHW 网格，坐标 `[B,T,2]`，重复格累加、越界忽略。
  `pointer` 对 `[B,D]` query 与 `[B,T,D]` keys 点积，可配置缩放。

### PVZGN001 二进制布局

小端，无压缩；字符串为 u32 字节数+UTF-8，向量为 u32 元素数+连续元素。

1. 8 字节 magic `PVZGN001`，i32 metadata 向量，u32 tensor_count。
2. 每个 tensor：name 字符串，i32 role（0=中间量、1=输入、2=权重），i32 shape 向量；
   role=2 后跟 product(shape) 个 f32。各张量的隐式 id 就是记录顺序。
3. u32 node_count；每节点为 op 字符串、i32 stage、i32 output id、i32 input ids 向量、
   i32 参数向量、f32 参数向量。节点按依赖和 stage 排序，不支持原地覆盖。

算子 i32 参数：conv=`padH,padW,strideH,strideW[,1表示稀疏输入通道]`；concat=`axis`；permute=`轴序列`；
pool=`归约末尾轴数,0均值/1最大/2均值拼最大`；norm=`0 LayerNorm/1 RMSNorm`（f32 epsilon）；
attention=`heads`；ffn=`激活`；moe=`激活,0topk/1硬路由,路由专家数,topk,共享专家数`；
broadcast=`每个输出轴对应的输入轴，-1为新增轴`（可丢弃输入的单例轴）；
slice=`axis,start,length`；relation=`行差半径,列差半径`。pointer 的 f32 参数为缩放；
mask 的 f32 参数为填充值（mask 输入<0 的连续尾块被覆盖）。其余算子无参数。
linear 输入 x,w[,bias]；conv 输入 x,w[,bias]；norm 输入 x,scale,bias（RMS 无 bias 时导出零向量）；
FFN 输入 x,up_w,up_b,down_w,down_b；MoE 输入 x,route，随后每个路由专家再共享专家的上述四组参数；
scatter 输入 grid,entities,coords；embedding 输入 ids,table（负 id 映射 0）；select 从 `[B,T,D]`
按 `[B]` 槽位取 `[B,D]`；relation 输入 meta,kind_table,row_table,col_table,same_table。

独立驱动可直接编译或用 `-DPVZ_GENERIC_DRIVER=ON` 构建。输入文件按 role=1 的记录顺序串联 f32，
输出按节点顺序串联，供逐中间量对拍；生产不输出中间文件、不添加前向自检。
阶段 1 相关测试：CNN64 各中间量最大绝对误差 2.69e-7；B=2 attention（含全掩码）、
RMSNorm、ReLU FFN、共享+top-k MoE 和 B=2 identity CNN 对拍通过，单项约 2 秒。

## 阶段 2：引擎接入（2026-10-29）

`EnvironmentRollout::Load` 按 magic 分流，新格式供现有 `ROLLOUT_PREDICT` 根节点策略及
`ROLLOUT`/`ROLLOUT_VALUE` 续走使用；旧格式与默认导出不变。能力字段新增
`generic_forward_version:1`。Python `env.load_rollout_net(path)` 接受新格式；命令行导出：

```bash
python python/pvz_rollout_policy.py checkpoint.pt student.bin --generic
```

原生适配器支持可选输入 `board:[1,C,6,10]`、`scalars:[1,S]`、`packet_info:[1,10,4]`、
`terrain:[1]`，必需 `selected_packet:[1]`。metadata 为 `[观测id1..9,task_context0/1,identity0/1]`。
terrain 为白天0、夜间1、泳池2、雾3、屋顶/最终Boss4；可用 broadcast 扩展到每 token。
task157 增加富输入时在 `LoadGraph` 绑定新输入，在 `Encode` 填数组即可，不需要改算子库。
独立 token 示例使用合成 token 输入，因此需 task157 的富输入适配器后才能直接用于游戏。

在本 worktree 的 `build-cppnet` 独立编译，nice10、-j3。真实占用局面选 3 种布局
（v1/slot，v2/context/identity/非等宽embedding，v2timer/row_pool/identity CNN64），
12 个决策逐中间量与 torch 原模型对拍，最大绝对误差 4.77e-7；旧路径/新路径完整动作一致。
每种布局各做一次 600 tick 原生续走，两路径终态 hash 一致。测试约 2 秒，地址空间上限 8,000,000 KiB。

## 阶段 3：性能、最终对拍与交付结论（2026-10-29）

采用 CBLAS 矩阵核和编译器向量化，float32、不量化、不启用 fast-math；OpenBLAS 加载时限制单线程，
Accelerate 设置 VECLIB_MAXIMUM_THREADS=1（调用方也应在进程启动前设置）。每个图加载时分配各节点
输出、im2col、注意力分数、FFN/MoE 工作空间；前向没有堆分配。首层按实际非零通道缓存打包权重，
活跃通道集变化才更新；im2col 只拷有效连续行，padding 从加载起保持零，1×1 直接读原数组。
只依赖权重的切片/reshape 在加载时折叠；格子头的全局/卡包投影只算一次再广播，避免每格重算。
硬路由 MoE 按专家收拢 token 做批量 GEMM，再按原位置回写；共享专家批量执行。
这是前向实现优化，未增加游戏策略规则或启动/续跑自检。

最终测量：本台式机 i7-12700F / WSL x86_64，GCC 15.2 Release / LTO / -ffp-contract=off，
OpenBLAS 0.3.32，nice10，batch=1、单线程，每项热图 200 次均值，包含编码及全部三头，
不含特征构造、模型加载、文件 I/O。保持生产运行，未绑核或停服务。完整记录在
[task161-cpp-generic-forward-results.json](task161-cpp-generic-forward-results.json)。

| 示例 | 稠密随机输入 µs | 8 个非零输入通道 µs | 稠密耗时 / 历史 61 µs |
|---|---:|---:|---:|
| 现行 CNN64（v2timer / meanmax / row_pool / identity） | 308.4 | 213.3 | 5.06 |
| CNN128（相同头与观测） | 879.6 | 686.9 | 14.42 |
| ResNet，stem + 6 个双卷积残差块，宽64 | 1167.0 | 1129.2 | 19.13 |
| 2层 token / 宽128 / 64 tokens / 4头 | 1666.6 | — | 27.32 |
| 2层 token / 宽128 / 128 tokens / 4头 | 3818.5 | — | 62.60 |
| 4层 token / 宽128 / 64 tokens / 4头 | 3491.4 | — | 57.24 |
| 4层 token / 宽128 / 128 tokens / 4头 | 7394.3 | — | 121.22 |
| CNN + 2层 token + scatter | 1930.7 | — | 31.65 |
| 1层 token + 5 专家硬路由 SwiGLU | 1239.8 | — | 20.32 |

注意力示例包含 pre-LN、关系表、Q/K/V/输出投影、4倍隐藏宽度 SwiGLU FFN 和条件指针头；
MoE 示例让 64 个 token 的外部 id 混合覆盖 5 个专家，不等同于全局一个地形 id 的最优批量。
“8通道”是同一随机输入只保留前8个通道的可重复稀疏算例，不冒充真实帧速度。
61 µs 是任务书引用的历史数值，并非本次同输入/同负载的旧路径实测；上表比值仅作量级参照。
**通用 CNN64 仍未达到该历史速度，不建议据此切换生产默认。** 本任务不设速度晋升门槛；
交付的是可配置前向能力与真实计时，后续可继续做融合/内核优化，不以降低精度换取对拍通过。

最终随机对拍涵盖上表 9 类结构及3个稀疏变体，比较每个节点输出；最大绝对误差 **2.03e-6 < 1e-5**。
CNN64/128 额外直接对照原 RolloutNet 三头，而不只对照图解释器。优化后的真实帧/旧路径对拍
最大误差 **3.73e-7**，12 个决策动作仍相同，三次短原生续走终态相同。
小批 B=2、全掩码、RMSNorm、非均匀 top-k/平局/共享专家的针对性测试通过。
所有测试均独立 `-k`、8,000,000 KiB 上限，单项2–3秒；没有全量数据重放、训练或生产改动。

复现基准（独立驱动，不启动游戏）：

```bash
nice -n 10 cmake -S . -B build-cppnet -G Ninja -DCMAKE_BUILD_TYPE=Release -DPVZ_GENERIC_DRIVER=ON
nice -n 10 cmake --build build-cppnet --target generic-forward -j 3
free -g
ulimit -v 8000000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 300 nice -n 10 "$HOME/.venvs/ml/bin/python3" \
  scripts/bench_generic_forward.py --iterations 200 --sparse-cnn --output build-cppnet/results.json
```

task157 可直接从本分支取 `GenericForward`、导出器和上述 ABI；富 token 的原生数组适配、
训练器结构和配方仍由它负责。当前新格式已能在现有原生特征适配器下作为根节点/续走策略使用，
独立富 token 示例需接入相应富输入适配器。旧导出/旧模型、生产默认、qc、看门狗均保持原状。

## 续任务阶段 4：热点剖析与可选卷积后端（2026-10-29）

按用户续任务继续推进 61 µs 目标。独立工具显式追加 `profile` 可输出逐节点计时，不进入生产热路径。
原稀疏算例主要热点为第二层卷积约69 µs、1×5卷积约40 µs，首层约14 µs；没有发现每轮重新分配
本库缓冲。新增固定形状搬运计划/定长内联复制、位 OR 稀疏通道扫描和 batch1 GEMV。

可选 `PVZ_GENERIC_DNNL=ON` 使用 oneDNN FP32 卷积：权重与 user scratchpad 在加载时准备，
conv_relu 明确表达融合激活，相邻卷积保留内部布局；调试驱动在计时外 materialize 中间量。
默认 OFF，未改变旧路径/生产依赖。此机仅在 build-cppnet 下解包 Ubuntu oneDNN 3.9.1 依赖，
没有系统安装、没有改生产服务。自写 Winograd 试验收益不稳定，未保留。

增加显式开发命令 `ROLLOUT_BENCH N`，同一真实帧只构造一次特征、计时完整编码+三头。
初次同机真实帧测得旧路径169.6 µs、通用192.5 µs；因此历史61 µs不能冒充本台式机此次旧路径实测，
且此阶段还没有达到用户目标。可选后端的合成8通道CNN64约131–145 µs。
真实帧对拍最大误差3.58e-7、动作与三次短续走一致，约4秒；性能继续优化。
