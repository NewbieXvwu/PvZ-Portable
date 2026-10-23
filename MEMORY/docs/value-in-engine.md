# task37：价值前向与静态候选下沉引擎

基线：`c790588d`，独立 vcpp Release 构建；只操作本工作树，游戏进程不超过 4。
实现与验收进行中，未达到停止条件。task30 价值编码按 `14713bd4` 的定义导入：模型 v3 使用编码 1，模型 v4 使用编码 2，不改变学生编码签名。

静态候选语义：从 root 恢复、执行动作、立即打分；种植和铲除推进 0 tick。静态收入使用根观测的 sun_income_rate，而非分支累计收入。叶子收入使用完整推演的 6000 tick 窗口（保留决策边界和终点裁剪）。

待验收：四场景各 256 帧、v3/v4 logits <1e-5；逐帧动作重放；本机老师自身噪声底线；三任务各 32 局 beta=1 配对；耗时与 RSS；连续三项收益 <2% 后停止。

已实现独立 `EnvironmentValueNet`，未改 `EnvironmentRollout.cpp`，无共用数学工具需要协调。PVZVN001 头为 magic[8] + little-endian uint32 width / encoder_version / context_dim，参数按 ValueNet.state_dict 顺序 float32。上下文维度仅接受 0/35，拒绝坏头、截断、非有限数及尾随数据。

首次 oracle：`python/test_engine_value.py`，双编码 × L7/L14/L21/原版L10 ×16时点（128个局面，随机 width64 上下文网络）；Board 特征最大误差 0（逐位相同），logit 最大误差 7.48e-9，静态概率最大误差 6.01e-8。检查非法候选空值、0 tick种植、分叉后 root 恢复相同。该测试尚不替代真实 frames_v2 各256帧及训练权重验收。
