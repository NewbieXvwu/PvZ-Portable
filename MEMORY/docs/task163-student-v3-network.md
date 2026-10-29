# task163：学生 v3 网络与训练接入（task157 的并行分支）

分支 `studentv3net`，基于 `origin/prodint16`。在台式机做（Codex 账号 A，第二个会话）。

## 分工

- task157（Mac codex-a，分支 `studentv3`）：富输入 obs v3（C++ 产出 token）与存档回填。
- task161（台式机 Codex business，分支 `cppnet`）：C++ 通用前向库。
- 本任务：Python 侧的**网络与训练接入**。

## 要做的

1. **移植旧 `GameplayModelV1` 编码器**（`git show pre-cleanup-20261019:python/pvz_agent_model.py`）：
   - 保留：12 种 token、种类 / 类别 / 变体嵌入、行列嵌入、关系偏置（种类对 + 行差桶 + 列差桶 + 同格 + 目标链接）、pre-LN、SwiGLU、卡包 / 格子指针头；
   - 去掉：GRU、PPO critic、事件等待；
   - 输出对齐现行训练器的三头（种 / 等、卡包、格子）。
2. **结构可配置**：宽、层、头数；可选按地形硬路由，或共享 + top-k 的 MoE FFN；可选与 CNN 混合（scatter 把实体加回格子）。都做成 `pvz_recipe` 的字段，默认不变，旧检查点逐位不变。
3. **输入**：
   - 先用现有紧凑帧能给的 token：卡包、植物、僵尸精确 x 和速度、投射物、格子；
   - 留好接口，obs v3 富 token 就绪后直接替换（与 task157 约定 token 数组格式，写在本文件里）。
4. **训练**：打包 / 训练 / 加载路径接通；导出走 task161 的通用格式（接口就绪前先留桩）。
5. 小测试：默认关闭时逐位不变，各结构前后向能跑。

## 约束

- 台式机是生产机：
  - 测试只跑 `-k`，加 `ulimit -v 8000000`，先看 `free -g`；
  - 不在台式机用 GPU 训练（等 task162 执行器就绪）；
  - 不碰生产。
- 不加自校验。分步提交推送 `studentv3net`。结论写在本文件末尾。

## 阶段 1（2026-10-29）：Python 编码器与输入接口

- `pvz_student_v3.StudentV3` 保留旧编码器的 12 类 token 词表、三种语义嵌入、行列嵌入、32 维特征投影、五项关系偏置、pre-LN 与 SwiGLU。去掉 GRU / critic / 事件等待。状态池化排除 seed_packet；输出 plant logit、10 卡包指针、选中卡包条件下的 54 格指针。
- 配方 opt-in：`obs="v3", packet_head="identity", width=192, network={"architecture":"token","layers":4,"heads":6,"ff_width":768}`。`network.ffn` 为 `dense` / `terrain` / `moe`；后两种支持 `shared_experts`（默认 1），top-k 另有 `experts` / `top_k`。`network.hybrid=true` 用 CNN 格子特征加实体 scatter。未启用时仍走现行 CNN；配方解析不导入 torch。
- 小 CPU 测试 7 项通过（约 4 秒），涵盖四种结构的前后向、目标链接梯度、padding 和配方约束。无训练或生产操作。

### 提供给 task157 的 token 数组契约（待富输入分支接入）

紧凑帧路径为 `frame_tokens(frame, context)`；若帧含 `tokens_v3`，直接消费数组，不再逐实体组装。当前未获得 task157 的最终产出实现，以下是本分支提供的接口，合并时由该分支按此输出或加一个适配层，不能假定双方已联调。

`tokens_v3` 对象：

| 字段 | 形状 / 语义 |
|---|---|
| `schema_version` | 1 |
| `ids` | N×5，int16：[kind, category, variant, row, col]；category / variant 是原类型 + 1，0 表示未知，范围 0..127；row/col 未知为 -1 |
| `features` | N×32 float32，缺少字段填 0，不估计隐藏值 |
| `target_indices` | N int32，公开目标 token 的帧内下标，无目标为 -1；不使用引擎对象 id |
| `packet_index` | 10 int32，每个槽位的 token 下标，空槽为 -1 |
| `cell_index` | 54 int32，按 row×9+col 顺序的格子 token 下标 |
| `terrain` | 标量 0..4：白天草地、夜间草地、泳池、雾池、屋顶；用实际 scene flags，不能按关卡号取整（L35 例外） |

kind 编号沿用旧模型：global=0, profile=1, cell=2, plant=3, zombie=4, projectile=5, defense=6, grid_item=7, seed_packet=8, zombie_roster=9, lane=10, lane_enemy=11。卡包特征不包含槽位编号。

特征槽沿用旧 `observation_tokens` 的布局：植物 0/1 行列、2 血量比例、3/4 当前/最大血量÷3000；僵尸 0/1 行列、2/3 x÷900/y÷700、5 本体血量÷2000、10 vx÷5、11 减速÷1200、13 冰冻÷1200、17 啃食、20 on_board、21 总血量÷2000、22/23 头盔/盾牌血量÷2000、24 体型（末四项为紧凑帧补充）；投射物 0/1 行列、2 x÷900、9 伤害÷500；卡包 1 价格÷500、2 冷却÷3000、3 总冷却÷3000、4 active。其余富特征槽由 task157 接入时明确，变更已有槽的含义需要新 schema，不能静默替换。

紧凑帧可提供的类型是 global/profile占位、cell、lane、plant、zombie（精确 x/vx）、projectile（v2 子块可用时）、defense、grid_item（graves）、seed_packet、zombie_roster（context 开启时）。缺少的状态/坐标、目标链接保持空；不推算轨迹，不增加策略公式。N 不截断；打包保留变长数组，训练只补齐当前 minibatch。

## 阶段 2（2026-10-29）：打包、训练、加载与 Python 出手

- `pack_training_rows --obs v3 --packet-head identity` 打包 token。v3 不复用缺少 token 的旧 episode 特征缓存；只在构建缺失标签段时读取帧，已有不可变分段照常复用。token 按行变长保存，构建中用临时文件流式累计，避免囤积逐行 Python 对象；不保存 N² 关系矩阵。
- `train_rollout_policy` 的现行分段合并器合并 token offsets、数据及帧内指针，保持原来的全局档案行顺序。合并后的 token 数组与其他列一样落在训练数据目录自己的 `.merge` memmap 中；复用时不重新解析 token 或旧档案。仅当前 minibatch 补齐 token。
- `make_student` 按 recipe.network 构造 CNN 或 StudentV3。现行损失（种/等、卡包、格子、可选辅助头/格子软标签）接通；checkpoint 自带完整 network 配置；`load_student`、同构 warm-start、`act` 与 `ranked_plants` 支持 v3。`StudentHistory(v3)` 是无状态兼容入口。
- **task161 导出按任务允许留桩**：`export_student(v3)` 明确报 `NotImplementedError`，不创建半成品文件。`export_v3(net,path,graph)` 已按 task161 当前 `export_graph(nn.Module, GraphSpec, path)` 签名留接入点。已读取 cppnet worktree 的接口初稿：还需把特征投影 SiLU、任意索引 gather、masked state pooling、目标链接偏置与 token 准备表达为兼容图/原生输入；未把另一并行分支的未提交文件复制到本分支，未宣称已有可部署的 v3 原生模型。旧 PVZRN002–006 导出不变。
- 当前的僵尸紧凑帧 `health` 是本体+头盔+盾牌总和，放新槽 21；有 v2 头盔/盾牌细节时才填本体槽 5。速度/减速/啃食等前缀槽与旧富 token 对齐，缺失的最大血量、状态等不伪造。富输入直接消费 `tokens_v3`，不会再走紧凑实体循环。

### 使用示例（仅说明，未启动实际训练）

在现有训练 recipe 中显式改这些字段，其他数据、标签和权重项仍由同一 recipe 给出：

```json
{
  "obs": "v3",
  "packet_head": "identity",
  "width": 192,
  "pool": "mean",
  "row_pool": false,
  "batch_size": 16,
  "validation_batch_size": 32,
  "network": {
    "architecture": "token",
    "layers": 4,
    "heads": 6,
    "ff_width": 768,
    "ffn": "dense",
    "hybrid": false
  }
}
```

`width` 为现有配方字段，不另加一份宽度。`ffn="terrain"` 用 5 个地形专家 + `shared_experts`；`ffn="moe"` 用可配置 `experts`、`top_k`、`shared_experts`，每 token 只执行选中专家。top-k 同分按专家 id 升序，便于对接通用 C++ 实现。`hybrid=true` 将实体加回 6×10 CNN 格子；格子指针按每行前 9 列取出，保留 54 格顺序。

打包仍使用现行命令与相同 recipe 的行选项（data / labels / wait_keep / context / aux_horizon / cells / label_weight）；训练入口仍是 `scripts/train_rollout_policy.py --recipe /绝对路径/recipe.json`。更换宽/层/FFN/混合结构不改变 token pack 的语义，因此不因结构变化重新打包。

### 交付结论

- 10 个新增小 CPU 测试通过（约 3.3 秒）：四类结构前后向、padding、token 数组入口、跨标签分段合并与复用、一次真实 trainer optimizer step（包含辅助头/软格子标签）、加载与合法动作出手、导出桩，以及默认 CNN 的权重/三头输出/旧格式导出逐位一致。
- 另各跑一个与共享改动直接相关的旧配方与分段合并测试，均通过；全部测试先看 `free -g`，带 `ulimit -v 8000000`、超时及单线程限制。未跑全量测试、全量存档重放或真实语料训练；未碰生产进程、二进制或 GPU 训练。
- task157 的富输入生产与 task161 的原生图适配仍由对应并行任务衔接；本分支提供可消费数组的网络、打包/训练/加载/Python 推理路径。未做胜率、吞吐量或晋升结论。

## 续接阶段 3：task157 的实际富输入（2026-10-29）

合入 task161 `origin/cppnet`（5e502c93）的 PVZGN001 依赖。task157 输入代码取自 `origin/studentv3` 的 e66d878a；只引入公开 token、回填及数组适配，不引入其被分工取代的 Python 网络初稿。

新增显式 `token_input="obs_v3"`：直接消费 `v3.version=3 / v3.tokens[N,38]`，前 6 列是 kind/category/variant/row/col/target，后 32 列直接作为特征，不转换槽位、不重复加 1。固定 cells=2..55、cards=56..65；空卡槽 category=0，key mask 无效，卡包指针为 −1。与阶段 1 的 `tokens_v3` 紧凑原型是不同契约，旧默认 `token_input="compact"` 保留；富模型遇到缺 v3 的帧报需回填，不静默回退。pack 指纹只为新格式增加该字段，旧指纹保持不变。检查点记录 token_input，同构续训禁止跨格式复制。

观测开关、帧保存、search_play 的 `--obs-v3` 与 Python 学生评估入口接通。实际富槽位以 [task157 输入契约](task157-student-v3-token-encoder.md) 为准；阶段 1 槽位说明仅适用于 compact 原型。

## 续接阶段 4：完整 PVZGN001 导出（2026-10-29）

`pvz_student_v3_export.student_graph` 导出完整的嵌入/特征 SiLU/逐层关系注意力/SwiGLU 或 MoE/可选 CNN scatter/三指针头。GenericForward 增加通用 SiLU、批内 gather、带掩码均值池化；embedding 可选偏移（行列未知位），relation 支持两桶同格表及可选目标链接表。旧节点参数和旧图继续有效。权重快照在导出时合并 gate/value，前向不拼接权重。

新图 metadata 的 observation id=10 表示 task157 富输入；线格式仍是 `v3.version=3`。`export_student` 和 CLI（含 `--generic`）自动导出新格式，可用 `--token-capacity` 指定固定容量，默认 256。容量不足明确报错，不截断实体。compact 原型 checkpoint 明确拒绝按富输入导出，以免无声改变模型输入。

原生 `EnvironmentRollout` 增加 id10 输入适配：直接从 PublicObservationTokens 填 float32 原始数组、有效位、排除卡包的池化位、帧内指针和实体 scatter 坐标；编码/种等卡包仍在 stage0/1，stage2 可多次给定卡包算格子。导出的 batch>1 图供独立驱动；原生游戏适配器是 batch1。

完整模型小型独立 C++ 对拍（batch2、含空卡槽/补齐/目标链接/重复 scatter、24宽2层3头）四结构均通过，三头对原始 StudentV3 的最大绝对误差：dense 2.09e-7、地形 1.64e-7、top-k 2.02e-7、hybrid 2.99e-7，均 <1e-5。约2秒，无训练/存档重放。真实富帧与引擎适配将在本次交付收尾记录。

## 续接阶段 5：task162 作业交付（2026-10-29）

合入 task162 的 `gpuwindow`（8ef62176），沿用其原有信号让路、模型/优化器/RNG 断点、独立 frozen-pack/merge 路径。没有安装/重启服务或修改生产队列。统一 `create_student` 与本分支的 `make_student`，`arch=student_v3` 按 network 构造模型；先前没有 arch 的本分支 v3 检查点仍可加载。CPU 评测同样传入 token minibatch。

可提交的矩阵为 `configs/task163_student_v3_jobs.json`；准备入口为 `scripts/prepare_student_v3_jobs.py`。7 类结构 × seed 1/2/3：

| 名称 | 结构 | batch |
|---|---|---|
| cnn64 / cnn128 | 现行 v2timer、meanmax、row_pool、identity | 256 / 128 |
| token128_2 | 128宽、2层4头、FF512 | 16 |
| token192_4 | 192宽、4层6头、FF768 | 16 |
| terrain5 | 128宽、2层4头、5个地形 FF256 专家+1共享专家 | 16 |
| moe5_top2 | 128宽、2层4头、5个 FF256 专家、top2+1共享专家 | 16 |
| hybrid128_2 | token128_2 + CNN scatter | 16 |

每作业12轮、cosine学习率，init=null。资源申报为 host 5GiB / GPU 4GiB、每空档最多900秒、每30秒断点；这是保守作业预算，并非完整训练实测。执行器仍执行自身生产余量/显存门禁。CPU评测batch=32（token）/256（CNN）。所有结构复用同一训练档案划分与抽样；CNN与v3分别仅打包一次，结构和随机种子变化不重新打包。

准备命令（仅创建独立作业，不入运行队列）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ~/.venvs/ml/bin/python3 scripts/prepare_student_v3_jobs.py \
  --source /绝对路径/task157_obs_v3_7e702892 \
  --state /绝对路径/新的独立作业目录
```

输入 manifest 的 train 档案单独硬链接为训练语料；训练内部另保留固定种子的10% validation，但 GPU 作业跳过逐轮验证。seen/unseen/terrain_sample 全部隔离为外部 CPU 留出集，逐行评测不抽掉等待行。`eval_window_policy --manifest --eval-data` 使用源 manifest 的分组与地形，terrain_sample 不冒充 unseen；旧的训练包内 seed/deck 留出方式保持可用。它们是该155局数据上的结构作业，**不是2300万行生产规模实验，也不是等参比较或旧 task157 初稿的数值复现**。

本机生成目录为 `/home/newbiexvwu/PvZ-search/artifacts/student/task163-window`，其中 `jobs/*.json` 是 task162 接受的绝对路径 argv 作业，`recipes/` 是逐作业冻结配方，`prepared.json` 记录包路径与划分。未复制到正在运行的生产空档执行器队列；供主控按实验安排取用。

### 最终输入/原生验证与中转清理

- HF真实富输入抽白天、屋顶、雾、原生泳池传送带各一帧，四种结构均对拍原始 StudentV3 三头；最大绝对误差 **7.16e-7 <1e-5**。
- 本 worktree 独立 Release 引擎 `build-studentv3net`，nice10/-j2；四种结构通过真实引擎 id10 适配，三头最大误差 **1.80e-7**，合法动作与 Python 一致。没有替换生产二进制。
- 小CPU作业测试覆盖7结构配方生成、v3一次 SIGUSR1 中断/断点接续、外部 manifest CPU评测及三组行数；默认CNN检查点/前向/旧导出仍逐位一致。测试全部按相关 `-k`、8,000,000KiB虚拟内存上限执行，没有GPU训练或全量重放。
- HF 155局已下载，并硬链接到作业自己的 train/heldout 语料目录。已按用户要求删除远端 `transfer/task157_obs_v3_7e702892/`，删除提交 `1e69acb1727f145f9ef94455144947159f9af509`；本地数值档案保留供后续作业使用，未提交数据/检查点/日志。

### task157 冻结后的接手收尾（2026-10-29）

采集 `--obs-v3` 改为独立稀疏附件，紧凑帧逐位不变；打包器直接读附件。大语料复用 task162 冻结行计划生成 mmap `decision_tokens_v3`，训练器与 CPU 评估通过 `--tokens` 读取，支持中断续训。纯 token 可复用 v2timer 行 pack，混合 CNN 需 v1/v3 网格 pack。实现和磁盘/回填抽样结论统一追加于 [task157 任务书](task157-student-v3-token-encoder.md)。未部署或改生产 qc。

## 2026-10-30：tokenstore，deploy18 附件的分块训练存储

从 `origin/deploy18:e60a951d` 新建 `tokenstore`（含生产源码 `35fedfaf`）。deploy18 只移入了附件采集，本次选择性接回 task163 的 Python 编码器、配方、检查点与打包/训练接口，保留 deploy18 的 label_keep、共享权重调整等新逻辑，没有整支合并旧工作树。没有修改 C++、生产采集配置、队列、模型或二进制。

### 格式与接入

采用 task157 方案 A：**按行稀疏，最多32行独立 zlib level3 压缩，按需解码**。复用已有 `pvz_token_blocks` 的 TKB1 数值编码，保留 float32 位模式（含负零）、公开目标链接、完整实体数及帧内指针；每局边界提前收块。

token 直接存在原有不可变 label segment NPZ 内，仅三个成员：

| 成员 | 内容 |
|---|---|
| `token_data` | uint8压缩字节；每块前置uint32压缩长度，块内为已有稀疏数值列 |
| `token_offsets` | int64，监督行数+1；每行所在块的字节地址，同块行共享地址，末项为总字节数 |
| `token_rows` | uint8，每行在块内的序号0..31 |

不增加 manifest、内容哈希或 token sidecar JSON；沿用现有 pack 的内嵌 meta/索引。富附件身份只读现成 ZIP 目录的CRC/长度，不重放、不解压全库。`obs=v3` 的包合同增加 `token_storage=sparse-block32-v1`，避免误用早期稠密原型段；CNN默认合同不变，旧原型保留兼容读取。

打包每次只读一局已有稀疏附件，选中输入行 `i−1` 放入至多32行缓冲；沿用原有等待抽样、拒绝行排除、显式training_rows及各类权重。分段合并仅平移地址、复制压缩字节和块内行号，**不解压token，不生成全局密集特征或未压缩稀疏列**。合并后的三个NPY与其他训练列一起mmap，后续训练复用缓存。每进程16MiB LRU只缓存解压后的**稀疏**块，仅当前minibatch复原密集特征。原有档案行序、全局随机抽样和验证划分不变。纯token网络跳过无用的board/scalar batch解码，hybrid仍读网格。

仍用原打包器和训练器，在**独立训练视图**运行：

```bash
python scripts/pack_training_rows.py --data /独立视图 --labels 标签列表 \
  --obs v3 --packet-head identity --token-input obs_v3 --precompute-workers 1
python scripts/train_rollout_policy.py --recipe /独立配方.json \
  --pack /独立视图/packs/包.npz --merge-dir /独立合并缓存
```

配方关键字段：`obs="v3", token_input="obs_v3", packet_head="identity", pool="mean", row_pool=false, network={"architecture":"token","layers":2,"heads":4,"ff_width":512}`，以及现有width/data/labels/out字段。更换宽层、FFN或种子不重打包。本入口不依赖旧decision_tokens、sources.sqlite或`--tokens`，未接回GPU空档服务/作业生成器；也没有因此给deploy18原生引擎增加新网络前向。

### 真实deploy18样本实测

只读硬链接8份完成的c53附件、原帧及已有meta到 `/home/newbiexvwu/PvZ-search/artifacts/student/task163-tokenstore-c53/data`，没有重放或回填。源库为 `artifacts/student/frames_v2`，覆盖L19、L23、L24、L26、L28开局组；种子为210000753、210000762、210100776、210200767、210300745、210300756、210300761、210400749。7个标签段，共**10,787监督行、1,485种植行**。

| 磁盘项目 | 实测字节 | 同分布外推2,300万行 |
|---|---:|---:|
| 原采集附件（保留，硬链接不另复制） | 2,354,374 | 5.02 GB |
| 单份token，含索引，270.25 B/行 | 2,915,156 | 6.22 GB |
| 完整pack索引+7段，含网格/标量/监督/token | 7,170,081 | 15.29 GB |
| 完整合并缓存，含上述训练列 | 6,956,557 | 14.83 GB |
| 新增pack+合并缓存合计 | **14,126,638** | **30.12 GB** |

GB按10⁹字节。**6.22GB是单份token，不是训练路径总盘位。** pack和合并缓存各存一份，token部分外推合计12.43GB；原附件继续随源帧保留。全量行地址+块内序号约207MB/份，已计入token数字。这批夜间/泳池开局样本不能代表全地形容量，混合语料预算仍参考task157较大的压缩估算。没有生成全量包，也没有假定历史2,300万行已有富输入。

| 项目 | 实测 |
|---|---:|
| 首次打包，含原帧/标签/网格/token构造 | 4.367 s，2,470行/s |
| 首次分段合并 | 0.136 s |
| 重新打开已有合并缓存 | 0.0175 s |
| 随机4096次加载，禁用解码缓存 | **3,512行/s** |
| 顺序加载全部行，16MiB稀疏缓存 | **22,909行/s** |
| 128宽/2层/4头/FF512，batch8，真实trainer的12步CPU优化 | **2.481 s，38.7行/s** |

加载包含块读取/解压、选中行复原、padding和CPU tensor转换；文件在OS页缓存，**不是冷盘或GPU吞吐**。训练单线程，12步loss均有限，只说明链路可用，不作收敛/胜率结论；总墙钟含加载/建网/优化器/保存3.776s。保留按种子划分的2714验证行，速度试跑用`--skip-validation`，未跑整轮验证。

可复现入口：`scripts/bench_tokenstore.py --out /新的独立目录 --archives <上述原帧路径...>`，默认4096次随机加载、12步训练，遥测定向实验自身目录。实测在`free -g`后以Nice10、CPU单线程和`ulimit -v 6000000`执行；这个地址空间上限只用于小样本。全量mmap地址空间大于驻留内存，正式大语料作业应使用执行器的实际内存约束，不能照抄小测试的6GB地址空间上限。

相关小`-k`测试通过：附件到分段/合并/batch/缓存复用及位模式保持、真实trainer一步含hybrid/辅助头/软格子标签、默认CNN两种旧输入的权重/输出/导出兼容。没有全量测试、全量重包或GPU训练。样本、实验缓存、配方及检查点保留在上述独立目录，不进入git。

## 2026-10-30：块级打乱与多进程预取，加载超过50k行/s

训练器对新稀疏token包默认使用 `pvz_token_loader.TokenBlockLoader`：每轮打乱压缩块顺序，块内仅打乱该块入选的至多32行，worker按此顺序整块解压、批量复原。源监督行、标签、权重及训练/验证划分不变；**SGD行序有意改为块级随机**。任务由主进程按训练随机种子生成，结果按提交顺序消费，不随worker完成先后改变顺序。原CNN与旧稠密原型沿用原路径。

默认配方新增 `token_workers=4, token_prefetch=2`；设 `token_workers=0` 可保留旧随机行加载作对照。这两个执行参数不进入pack身份，不重打包。worker用spawn启动，进程池跨epoch保留；只打开现成的token_data mmap，不读源对局。全局行计划只需要块起点、选中行布尔位与块序列，不建立全量密集特征。

每个任务约512个入选行，按batch_size的整倍数切分，最后一个batch可以较短。worker一次将稀疏块批量复原到临时共享内存，**每个minibatch独立取自己的最大token长度补齐**，不按整个预取任务的最长序列补齐。主进程直接将共享数组视图转为CPU tensor，不通过pickle传输密集数组，再按需要送训练设备。4×2个在途任务加当前任务，最多约4608行有界预取；没有整轮预解码。共享缓冲在消费后释放，提前达到max_steps时关闭迭代器、回收未消费缓冲并结束worker，不让生产背负后台解码进程。

没有改存储格式；**新增常驻数据磁盘占用0字节**，同一份token仍为2,915,156字节。不新增manifest、哈希、JSON旁文件或任何校验。没有重打包、重放、全量测试或GPU训练；只运行本次吞吐测量与真实trainer的12步CPU训练。

### 同一c53样本的前后吞吐

直接复用上一节8局/10,787行的 `task163-tokenstore-c53/data/packs/897d54c254306e50.npz` 和已有合并缓存。batch=16，OS页缓存热；计入块解压、数值复原、padding、共享内存生产/消费、CPU tensor视图转换及每轮行计划，不计网络前后向或GPU传输。原随机行对照禁用解码LRU，模拟全量随机行低命中；原顺序读取使用16MiB LRU。

| 路径 | 本轮实测 |
|---|---:|
| 上一节旧随机读取历史记录 | 3,512行/s |
| 本轮同次运行重测旧随机读取，全10,787行 | **4,299行/s**，2.509s |
| 本轮旧顺序读取，全10,787行 | 32,460行/s，0.332s |
| 新块打乱+4进程预取，首次epoch含spawn启动 | 6,086行/s，1.772s |
| 新路径进程池就绪后，连续10个完整epoch | **95,850行/s**，107,870行/1.125s |

稳态各轮0.107–0.118s，约91k–101k行/s，超过50k目标；相对同次重测旧随机读取约**22.3倍**。单纯加载按同速外推2,300万行约**240秒/epoch**，这不是整轮模型训练耗时，也不是全量冷盘实测。小样本的首次epoch被一次性worker启动主导，不能用其6k行/s代表长作业稳态，也不能把稳态95k当作含启动的小样本首轮速度。

实现中2个worker曾测得55.6k行/s，但并发负载上升时复测只有38.3k行/s；因此最终改为更紧的按batch补齐与4个低优先级worker，采用上表的最终实测，不以早期单次最好值交付。

同一真实recipe的128宽/2层/4头网络、batch8、单线程CPU，使用新默认路径完成**12个optimizer step（96行）**，训练段2.252s（42.6行/s，含worker启动及提前停止回收），总墙钟约4s；输出在 `task163-tokenstore-c53/train-prefetch4`，仍隔离2714验证行且本次用`--skip-validation`。不作收敛或GPU速度结论。

复现命令如下；去掉`--recipe`即可只测加载，不运行优化器。默认4个worker、每worker预取2任务、10个稳态epoch，统计仅打印stdout。trainer遥测也落在独立输出目录。

```bash
free -g
ulimit -v 6000000
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
nice -n 10 /home/newbiexvwu/.venvs/ml/bin/python3 scripts/bench_token_prefetch.py \
  --data /home/newbiexvwu/PvZ-search/artifacts/student/task163-tokenstore-c53/data \
  --pack /home/newbiexvwu/PvZ-search/artifacts/student/task163-tokenstore-c53/data/packs/897d54c254306e50.npz \
  --recipe /home/newbiexvwu/PvZ-search/artifacts/student/task163-tokenstore-c53/recipe.json \
  --train-out /home/newbiexvwu/PvZ-search/artifacts/student/task163-tokenstore-c53/train-prefetch4
```

资源约束同上一节：Nice10及单线程环境由子进程继承，小样本每进程虚拟地址空间上限6,000,000KiB；正式全量mmap不能照抄该地址空间限制，应沿用执行器的驻留内存约束。未改生产采集、队列、配置、模型或二进制。
