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

算子 i32 参数：conv=`padH,padW,strideH,strideW`；concat=`axis`；permute=`轴序列`；
pool=`归约末尾轴数,0均值/1最大/2均值拼最大`；norm=`0 LayerNorm/1 RMSNorm`（f32 epsilon）；
attention=`heads`；ffn=`激活`；moe=`激活,0topk/1硬路由,路由专家数,topk,共享专家数`；
broadcast=`每个输出轴对应的输入轴，-1为新增轴`（可丢弃输入的单例轴）；
slice=`axis,start,length`；relation=`行差半径,列差半径`。pointer 的 f32 参数为缩放；
mask 的 f32 参数为填充值（mask 输入<0 的连续尾块被覆盖）。其余算子无参数。
linear 输入 x,w[,bias]；conv 输入 x,w,b；norm 输入 x,scale,bias（RMS 无 bias 时导出零向量）；
FFN 输入 x,up_w,up_b,down_w,down_b；MoE 输入 x,route，随后每个路由专家再共享专家的上述四组参数；
scatter 输入 grid,entities,coords；embedding 输入 ids,table（负 id 映射 0）；select 从 `[B,T,D]`
按 `[B]` 槽位取 `[B,D]`；relation 输入 meta,kind_table,row_table,col_table,same_table。

独立驱动可直接编译或用 `-DPVZ_GENERIC_DRIVER=ON` 构建。输入文件按 role=1 的记录顺序串联 f32，
输出按节点顺序串联，供逐中间量对拍；生产不输出中间文件、不添加前向自检。
阶段 1 相关测试：CNN64 各中间量最大绝对误差 2.69e-7；B=2 attention（含全掩码）、
RMSNorm、ReLU FFN、共享+top-k MoE 和 B=2 identity CNN 对拍通过，单项约 2 秒。
