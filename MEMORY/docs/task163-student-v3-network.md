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
