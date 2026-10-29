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
