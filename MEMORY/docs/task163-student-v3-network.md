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

特征槽沿用旧 `observation_tokens` 的布局：植物 0/1 行列、2 血量比例、3/4 当前/最大血量÷3000；僵尸 0/1 行列、2/3 x÷900/y÷700、5 血量÷2000、9 vx、10 减速÷2000、13 啃食；投射物 0/1 行列、2 x÷900、9 伤害÷500；卡包 1 价格÷500、2 冷却÷3000、3 总冷却÷3000、4 active。其余富特征槽由 task157 接入时明确，变更已有槽的含义需要新 schema，不能静默替换。

紧凑帧可提供的类型是 global/profile占位、cell、lane、plant、zombie（精确 x/vx）、projectile（v2 子块可用时）、defense、seed_packet、zombie_roster（context 开启时）。缺少的状态/坐标、目标链接保持空；不推算轨迹，不增加策略公式。N 不截断；打包保留变长数组，训练只补齐当前 minibatch。
