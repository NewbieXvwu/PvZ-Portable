# task192：卡牌静态属性离线臂

- 基于 `origin/gpuwindow` 的 `22648b35`（固定研究代码 `5ef7ad93`）；`card_attributes=false` 默认关闭，仅 CNN identity 头显式开启。关闭时权重、稠密/稀疏前向与固定版本逐位一致；无生产配置或 C++ 前向改动。
- 启动时从本检出的 `gPlantDefs`、`Plant::IsNocturnal/IsFungus/IsAquatic/IsFlying/IsUpgrade/IsSpiky`、`IsUpgradableTo` 和 `PlantSubClass` 生成 54×65 float32 矩阵（14,040 字节）。数值为基准费用/冷却/发射间隔的按列最大值归一化；升级底座为身份 one-hot，仅表示必要底座，不替代合法性判定。矩阵只驻内存，不落 JSON。
- 卡槽及所选卡的落点输入拼接属性；棋盘追加“属性矩阵 × 原植物 one-hot×血量”投影。保留原输入，旧模型照常加载；新臂仅用 Python 推理。
- `cnn64_cardattr_s1` 从现有 `r42_cnn64_s1` 配方派生，只开启属性（另改输出/代码来源）；复用同一包、mask、全局行哈希及预算：目标 49,152 行、1 epoch、seed1、同一约 32,768 行留出前缀。无重打包、重新划分或新增中间清单。
- 对照与新臂在同一输入遍历中评分，原 CNN64 结果不改写；新臂现有 `heldout.json.comparisons` 报 unseen、五地形及交叉分组。准确率差为新臂−CNN64，NLL 差越负越好；前缀报告保持 `evaluation_complete=false`。
- 10 项相关 `-k` CPU 测试通过（约6秒），先看 free -g 并设 `ulimit -v 8000000`。待固定提交后入 gpuwindow 队列，严格排在现有 CNN64 臂完成之后；尚无泛化收益结论。
