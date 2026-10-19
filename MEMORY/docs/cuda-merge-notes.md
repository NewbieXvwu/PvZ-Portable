# CUDA 合并说明（待主控审核）

`cuda-merge` 从 `cuda-rollout 3d74462` 合入 `origin/pvz-env adbbb2f`；合并提交 `f4c6eae`。
训练器、学生/缓存实现及上游更新的测试、文档保留主线；当日日志冲突按 append-only 保留双方内容。
只推 `cuda-merge`，不更新 `pvz-env`；正式 `~/pvz-cuda` 的源码与 build 未改动。

混采入口现默认 **poll=100µs、推测减半、精确复核预取**，没有减少样本或复核次数。
`PVZ_CUDA_POLL_US=0` / `--no-prefetch-verify` 可分别关闭；`--no-speculative` 自动关闭预取，
显式同时要求 `--prefetch-verify` 则报错。视界仍2500，近似关闭，普通 CPU 入口仍默认CPU。
训练时用 `PVZ_GPU_PAUSE_FILE` 的 touch/rm 让GPU采集在局间让路；旧服务需要重启才加载新代码。

首次在独立目录构建：
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPVZ_CUDA_ROLLOUT=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --target pvz-portable rollout_cuda_small -j2
```
正式混采一条命令（资产沿用台式机目录）：
```bash
PVZ_GPU_PAUSE_FILE=/tmp/pvz_gpu_pause scripts/mixed_dagger_round.sh \
  --student ~/PvZ-search/artifacts/student/v29/model.pt --task day_l7_z17_train \
  --seed-offset 15000 --games 16 --frames ~/PvZ-search/artifacts/student/frames \
  --frames-label cuda_review --out ~/PvZ-search/artifacts/hard/cuda_review --cpu-workers 3 --gpu-workers 1
```
分布式worker的GPU分片执行器如下；任务文件提供路径、LABEL、OFFSET、GAMES、FRAMES、OUT，
`PY` 指向worker自己的venv，资源留在worker目录；偏移必须来自controller台账，不能手选重复段。
```bash
PVZ_CUDA_POLL_US=100 PVZ_CUDA_PRIVATE_LIMIT=1200 PVZ_CUDA_COOPERATIVE_HEADS=1 \
"$PY" scripts/search_play.py --rollout-backend cuda --cuda-binary "$PWD/build/rollout_cuda_small" \
  --cuda-speculative-halving --cuda-prefetch-verify --rollout-policy "$STUDENT" \
  --value "$VALUE" --static-value "$STATIC_VALUE" --manifest "$MANIFEST" --task-id day_l7_z17_train \
  --student-candidates 2 --verify-samples 9 --dagger-beta 0 --workers 1 \
  --executable "$PWD/build/pvz-portable" --resource-dir "$RESOURCE_DIR" \
  --seed-offset "$OFFSET" --seeds "$GAMES" --frames "$FRAMES" --frames-label "$LABEL" --out "$OUT"
```
controller/网页的GPU开关由分布式分支接入；此处提供底层采集命令，不连接或修改朋友机器。

**5070 Ti 二进制复用**：两卡均cc12.0，本次产物含 `sm_120` cubin，设备码可兼容，依据
[GPU列表](https://developer.nvidia.com/cuda/gpus)与[NVIDIA二进制兼容规则](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#binary-compatibility)。
但这是Linux x86-64 ELF，只能在匹配ABI的Linux/WSL用，不能直接在原生Windows/Mac用。
本次 `readelf/ldd`：glibc≥2.38、GLIBCXX≥3.4.21、CXXABI≥1.3.9，需 `libcudart.so.13`、
`libcublas.so.13`、`libcublasLt.so.13`、系统C/C++运行库及驱动的 `libcuda.so.1`。
CUDA13要求R580或更新驱动，旧驱动对新特性的限制见[官方兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。
本机验收是5080/驱动617.14/Toolkit13.4；5070 Ti未实机验收。Ubuntu22.04的默认glibc不满足此预编译产物，
应本机构建；运行现成二进制不需nvcc。`pvz-portable`另需其 `ldd` 所列游戏运行库、游戏资源和Python依赖。
二进制、模型、帧不进git，模型沿现有分发通道；此任务没有额外上传大文件。

风险：预取多算未来，收益不保证覆盖所有候选数/复核率/并发；训练与采集仍抢GPU；跨机器不承诺浮点逐位一致。
CPU/GPU原本使用不同未来流，本次只要求优化前后GPU等价，不声称跨后端逐字节一致。

验收正在新worktree独立执行：CPU与最新主线各2局、GPU与冻结档案2局、全部Python/原生测试。
最终结果在完成后补齐；本地原始证据为 `artifacts/cuda/task7/`，不入git。
