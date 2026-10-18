# CUDA 续走模拟器：设计与实测

分支 `cuda-rollout`，只替代老师的续走。真实对局与帧档案仍由原版模拟器生成。
交接材料在 `~/pvz-logs/qoder_cuda_task.md`、`qoder_goal_addendum.md`、`qoder_answers.md`。
本报告在 2026-10-18 接手时重写；旧版本是阶段 0 前的设计，不能代表当前实现。
逐轮历史测量见 `~/pvz-logs/cuda_status.md`；机制来源见 [规格目录](../../cuda/spec/)。

## 当前结论

| 验收 | 当前证据 | 状态 |
|---|---|---|
| 阶段 0 | OpenBLAS 单线程 90.18 条/秒；20 线程折算 1804；门槛 9018。交接前 27200 道 15518 条/秒，即门槛 1.72 倍 | 已通过，性能修改后需复测 |
| 阶段 1 | f198afa 重编译通过；425 个根的 5668 个价值特征全部逐位相同；96 个强制种植候选与原版及主机参考零差异 | 搜索门槛测量中，尚未通过 |
| 阶段 2 | 尚无合格的 128 局配对与 DAgger 墙钟证据 | 未通过 |

阶段 1 的判据是搜索一致率至少为另一组 CPU 随机未来的一致率减 3 个百分点。
阶段 2 的判据是 z15 前 128 局配对 McNemar p > 0.1，胜局差不超过 4，并报告真实 DAgger 墙钟。
2026-10-18 接手时旧 CUDA 评估仍在运行已删除的旧二进制，保存其 29 局 / 21 胜后停止。
原记录保留在本地 `artifacts/cuda/handoff_pre_f198afa_eval/`，不计入验收。

## 实际架构

一条续走一个线程，游戏状态按字段连续（SoA）。每轮先由 PreKernel 推进无需网络的等待，
把需要前向的泳道压缩为连续激活槽；cuBLAS 完成三层卷积与选格头的 1×1 卷积；
PostKernel 完成上下文、动作头、选动作和动作执行。StoreKernel 回收统计与叶子特征。
每条泳道有独立随机流。批内调度不应改变同一根与未来的结果。

`cuda/SimImpl.inc`、`SimTick.inc`、`NetForward.inc`、`Rollout.inc` 由 nvcc 与 g++ 双编译。
主机参考与 GPU 的差异用于定位设备侧问题；主机参考与原版的差异用于定位机制问题。
主机参考支持 ASan。常规搜索通过 Unix socket 连接 `build/rollout_cuda --serve`，
一个服务进程汇聚各 worker 的候选；已有服务框架仍需实测汇聚规模与吞吐。
客户端在原版恢复根局面后执行 `CUDA_DUMP`，只把续走交给服务。

叶子直接输出价值网络特征 `[92×6×10 board][113 scalars][35 context]`。
Python 用同一个价值网络在 CPU 上打分，终局取真实 1/0。
学生权重常驻 GPU；真实根局面、候选与未来种子逐请求传入。
当前 `.cudaroot` 是 v5，旧 v4 必须明确拒收。

## 支持范围与仍须补齐的机制

范围是冒险第 7 关白天草地，卡组 `[0,1,2,3,4,5,7]`，普通、路障、旗帜、撑杆僵尸。
已有向日葵、豌豆/寒冰/双发、坚果、樱桃、土豆雷、行走/啃食/伤害/死亡、
天降阳光、自动拾取、卡槽、波次倒计时与割草机代码；存在代码不等于已逐项验真。

接手静态审查已确认的缺口：

- 撑杆起跳与落地尚未实现，当前按步行处理；不能称为全部机制已对齐。
- 土豆雷的三段武装动画仍待核对；实现只读倒计时，与规格描述不完全一致。
- GPU 使用导入的未来波表，尚未按 CPU RESEED 重新抽组成；必须补齐以免搜索看见已定未来。
- 原版动画行走目前用平均位移常数替代，事件时刻与几何仍需短视界逐项核验。
- 其他地形、植物与僵尸不在本任务范围；v2lite 历史特征尚未接入。

f198afa 对卡槽刷新做了正确方向的修正：导入原始 refreshTime；种植时写入实际冷却；
计数严格大于 refreshTime 后激活并清零。原版还有 mRefreshing 标志，当前用正冷却值推断，
仅可在当前关卡范围内视为等价，不能宣称覆盖传送带与老虎机。

## 搜索语义

CPU 与 GPU 都必须保留用户选择的 successive halving。
交接实现曾在 CUDA 后端静默关掉 halving；这会改变候选获得的随机未来数及最后动作，
已修正。需要全网格对照时，两臂显式选同一全网格配置。
不通过加宽搜索换取更好的 GPU 批量数字。

## 接手复测（2026-10-18，第 14 轮）

本地详细产物：`artifacts/cuda/codex_r14/`，不提交二进制、模型、语料或滚动日志。

- nvcc sm_120 -O3 与 g++ ASan 构建均通过。
- 根特征：425/425 完全一致，每根 5668 个 float。
- 强制首着：96/96 与原版完全一致，GPU 与主机参考完全一致。
- 修正回归测试默认语料 v4→v5 后，3 项设备/服务回归通过，6.012 秒。
- 同 425 根 ×8 个未来的 3400 条完整续走：零错误标记，CPU 输 8，GPU 输 10。
  叶子价值均值 CPU 0.452、GPU 0.432；逐对差 -0.020，标准差 0.167。
  随机流不同允许逐对不同，但该均值偏移尚未解释，不能据此宣告一致。
- 本批 3400 道 2983 条/秒，是较小批量下的正确性运行，不替代 27200 道阶段 0 门槛。
- 正在运行 6 种子、2 worker、3 未来、2500 tick、全网格的阶段 1 对照。

## 构建与验证入口

CUDA 暂时是独立可选构建，不会让无 CUDA 的主构建依赖 nvcc：

```bash
/usr/local/cuda/bin/nvcc -arch=sm_120 -O3 -std=c++17 --expt-relaxed-constexpr \
  -I. -lcublas -o build/rollout_cuda cuda/RolloutCuda.cu
g++ -fsanitize=address -g -O1 -std=c++17 -I. -o build/sim_host cuda/SimHost.cpp
```

协议布局：`scripts/cuda_layout_check.py`。
根编码及叶子分布：`scripts/cuda_leaf_compare.py`。
原版强制首着：`scripts/cuda_first_action_check.py`。
服务合批与根数组扩容：`python/test_cuda_serve_merge.py`。
搜索门槛：`scripts/sim_skip_agreement.py search --cuda-rollouts`。
老师与 DAgger：`scripts/search_play.py --rollout-backend cuda`。

## 资源与交付

常规 CPU 并行度不超过 4；显存不超过 8 GB；本任务内存不超过 6 GB。
available < 6 GB 时不启动测量；不制造换页。正式训练使用 GPU 时让出性能测量窗口。
服务测量 A 窗口最多 30 分钟、最多 12 worker，开始前 available ≥ 6 GB。
只提交代码、规格、KB 级结论和文档；不 push、不合并，不使用 git add -f。
