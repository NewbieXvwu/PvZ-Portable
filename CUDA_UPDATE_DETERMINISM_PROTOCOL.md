# 相同实际数据的CUDA更新确定性诊断（运行前冻结）

`observation_interrupt_audit_v1`严格比较失败，首次差异发生在SIGKILL之前的更新1，
相同起点、RNG、动作结果与计数未分离。原fail不改为pass。
这里独立诊断同输入的CUDA数值复现，不是新增奖励/能力候选，也不续跑失败探针。

固定读取连续臂实际初始化完整检查点与update 1全部20个原始NPZ，共1452决策。
每次从同一模型、空AdamW、完整Python/NumPy/Torch CPU/CUDA/分配RNG状态开始；
相同32宽两层模型/input_flags=7、R0、原lr=1e-4和全部PPO字段、FP32/dense。
CUBLAS_WORKSPACE_CONFIG保持:4096:8不变，只比较torch.use_deterministic_algorithms
关闭/开启两组，每组两次同数据完整更新。开启时warn_only=False，未知非确定操作
报错即保留失败，不隐藏警告或降低要求。每个优化器步骤保存全部参数的摘要，
最终保存完整参数/AdamW/RNG；比较必须逐项严格相等。

全量保存四次结果，无胜负选择、无温度修改、无正式矩阵修改。只有关闭组分离而
开启组严格相同时才支持这一运行条件假设；不能据此锁定某个具体算子或证明完整
中断恢复已通过。若支持，另注册新版本实际中断协议并重跑，保留v1。
启动时至少1024MiB当前空闲VRAM；与正式矩阵并行，墙钟不作独占机器速度。

依据为PyTorch官方确定性API说明，而非恢复流程已经错误的断言：
https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html
