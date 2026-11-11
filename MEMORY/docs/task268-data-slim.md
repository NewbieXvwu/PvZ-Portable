# task268：训练数据瘦身

2026-11-11，分支 `dataslim`，基于生产 `pkg238` 的 `b131933b`。代码与抽样测量已完成，未部署。台式机生产文件和 `cq` 工作树保持只读，实验全部写在 `~/pvz-logs/task268/`。原始测量与样本标识见 [task268-measurements.json](task268-measurements.json)。

## 容量与代价

全量容量沿用主控 11-11 提供的口径。样本容量以实际字节计量，全量压缩结果为外推。

| 项目 | 当前 | 改后 | 预计回收 | 代价 |
|---|---:|---:|---:|---|
| tokens_v3 旁车 | 31.7 GB | 日常新增为 0，按需生成单局 | 31.7 GB | 50 局回填合计 260.5 秒，含示范前缀恢复 |
| segments | 41 GB | 约 10.5 GB | 约 30.5 GB | 新段写入与首次并入共享文件时增加压缩、解压 CPU 时间 |
| 逐局特征 cache | 17.5 GB | 已打包部分删去，仅保留尚未打包部分 | 上限 17.5 GB | 主动重打包时重新提取特征，五局样本每局 0.88–2.12 秒 |
| 训练实际读取的 shared-v2 | 约 11.4 GB | 保留 | 0 | 取批路径和载荷不变 |

**旁车与分段合计预计回收约 62.2 GB，加上已打包缓存，合计上限约 79.7 GB。** cache 的实收量由维护命令输出，未把尚未打包的缓存计入可删除量。源 `frames.npz` 和 `meta.json` 保留。WSL 已配置稀疏虚拟盘，释放的完整块可归还 C 盘，生产新增数据会抵消部分净回收量。

## tokens_v3 回填证据

从生产目录按遍历顺序筛选，共检查 2,642 份元数据，选取五种地形各 10 局，共 50 局、40,297 帧。其中 33 局从 tick 0 开始，17 局由示范前缀恢复后续走，包含人工操作时间线、GPT 示范和 drill 起点。完整输入及示范依赖副本约 40 MB。

正式入口 `pvz_research_data.backfill_tokens()` 用当前生产 `artifacts/production/imitater1/pvz-portable` 重放，复用 `archive_task()`、`restore_prefix()` 和 `replay_actions()`，调用原有 `capture_frame()`、`frame_tokens()`、`SparseTokenWriter`。十个稀疏数组的形状、dtype 与原始字节全部一致：`ids/counts/columns/values/offsets/feature_offsets/packet_index/cell_index/terrain/target_indices`。50 局生成的压缩旁车共 8,795,528 字节，合计耗时 260.515 秒。

初测有一局 L39 的旧元数据没有 `environment_task`，仅在 `task` 中保存 `fog_observation=human_v2`。原 `archive_task()` 的旧格式分支遗漏了这一字段，第 17 帧开始出现特征差异。已修正共用恢复入口，正式回填复测该局全部 722 帧一致。

生产 `search_play` 没有独立的 resume 归档类型，非零起点通过 `demo_archive/demo_decision` 表达，未来随机流由 `seed_<原种子>_f<未来种子>` 表达。为单独验证中途续走，又从五种地形各选一局，在第 64 帧截断，用原档案作为前缀来源，通过同一恢复入口回填后半局。起点 tick 分别为 3480、3540、3360、3420、3600，共 5,198 帧，逐帧所有 token 字段一致。这五例是从生产局构造的续走样本。

回填依赖源帧、元数据及元数据引用的示范前缀。保留相应规则版本的引擎和示范源，回填时使用对应版本。本次只删除旁车的命令交由主控执行，不删除这些依赖。

按需回填一局：

```bash
~/.venvs/ml/bin/python3 ~/PvZ-search/.claude/worktrees/cq/python/pvz_research_data.py \
  --archive /完整路径/seed_123/某标签 \
  --executable ~/PvZ-search/artifacts/production/imitater1/pvz-portable
```

默认原子写入该局 `tokens_v3.npz`，也可用 `--out` 指定输出位置。该入口只在显式调用时重放，不进入采集、训练或启动预检。

## 分段压缩与训练读取

选择保留压缩分段，供重建、旧配方读取和重新打包复用。32 个随机生产分段的输入共 106,877,805 字节，压缩后为 27,374,575 字节，占原体积 **25.613%**，减少 **74.387%**。包含 180,239 行。

同一批数组分别经原 `np.savez()` 和新生产写入函数落盘，再完整读取全部成员，单线程、热缓存测量如下。新写入采用标准 ZIP Deflate level 1，读取仍由 `np.load()` 完成。

| 32 段合计 | 原未压缩格式 | 新压缩格式 |
|---|---:|---:|
| 字节 | 106,877,805 | 27,374,575 |
| 写入时间 | 0.090 秒 | 0.737 秒 |
| 完整读取时间 | 0.096 秒 | 0.350 秒 |
| 写入吞吐，按未压缩输入计 | 1,189 MB/s | 145 MB/s |
| 读取吞吐，按未压缩输入计 | 1,119 MB/s | 305 MB/s |

这是小样本、已缓存条件下的容器成本，未测冷盘全量吞吐。新增段立即使用新格式，旧段由迁移器转换。压缩和迁移共用 `pvz_training_pack.atomic_savez()`，迁移逐成员复制原始 `.npy` 字节，不转换浮点值，也不在内存装入整个分段。

同一子集构建 shared-v2 用时 1.415 秒，共享载荷为 45,916,788 字节。原位转换 32 个分段用时 1.049 秒，回收 79,503,230 字节。转换后全部 `.bin` 的 inode 和大小不变，禁止加载分段载荷的条件下，复用共享文件耗时 0.0237 秒。固定 18 批随机行号、每批 2,048 行，去掉前两批后，`BatchSource.prepare()` 中位数从 **38.58 ms** 变为 **38.17 ms**，批数据逐位一致。

压缩分段只在首次并入 shared-v2 时解压，epoch 从相同 `.bin` 取批，沿用现有原生解码和预取。因此本改动未增加 epoch 取批的工作量。未运行整轮 GPU 训练，未把样本取批时间换算成新的 epoch 时间。按本次迁移速度线性外推，41 GB 约需 6.7 分钟，文件目录操作与生产争用会延长时间。

旧 `index.json` 的路径、文件大小和 mtime 签名继续可读。迁移先从 ZIP 中央目录补充各成员的名称、CRC 和原始字节数，原子发布一次索引，然后逐段转换。读取器遇到容器大小或 mtime 改变时，用这些目录信息识别相同内容，保留原共享记录。已并入的源段暂时缺失，或者选择旧集合的子集，也可复用对应物理记录。真实内容改变仍走既有重建路径。需要重建却缺少源段时，会在删除共享文件前报错。

分段发布与迁移最终替换共用原有 `shared.lock`。压缩在锁外进行，锁内核对原文件代际并替换，避免覆盖并发打包的新代际。训练已打开的映射保持有效。内存准入的冷估算改为累计 ZIP 成员原始字节数，压缩体积不会导致低估训练内存，已有实测峰值仍优先使用。

## cache 的读取者与剪枝

`v2timer` 由 `cache_episode/read_episode_cache` 服务特征提取，可能引用 `v2sleep` 父缓存。`cells` 服务候选格标签，`packets/identity_scene` 服务身份卡头与场景标量。这些数据被打包器写进 segments，训练从 segments 或 shared-v2 读取，不读取逐局 cache。父缓存和视图都按同一源帧绝对路径的哈希命名。

五种地形各复制一套生产缓存，共 20 个文件、577,832 字节。每局读取四类缓存耗时 0.017–0.040 秒，从保留帧重新生成四类缓存耗时 0.88–2.12 秒。缓存能加速重打包，已发布且复用的段不需要它。

生产原有 `--prune` 路径现在于新段原子发布后，直接删除本段源帧哈希对应的本地缓存，涵盖根目录与各特征目录。它只处理本段，不扫描历史缓存库。整个打包持有 cache 的 `.pack.lock`，剪枝与正在读缓存的打包不会交叠。

存量清理由同一入口的 `--prune-only` 完成，默认最多删除 4,096 个缓存。它只处理已有完整分段覆盖的源帧，尚未打包的源帧缓存保留。目录整理和源帧路径解析放在锁外，每批最多删除 256 个文件后释放 cache 锁，让生产打包有机会继续。维护模式不构建分段，也不删除其他 pack 索引。中断后重新执行即可继续，已删缓存不会被重复处理。

## 主控部署与存量迁移

本任务没有运行 deploy26，没有修改生产数据。以下步骤由主控在部署后执行。

1. 使用现有 deploy26 流程，在组边界更新 `cq` 的本分支代码和采集启动命令。新队列启动参数应去掉 `--obs-v3`，保留 `--native-games`。只更新磁盘上的 `qc_launch.py` 不会修改已经运行的队列参数。共用引擎规则和训练解码器未改动，无需为本任务重编游戏引擎。
2. 新读取器生效、上一组旧采集结束后，再删除旧旁车。删除只涉及可回填附件：

```bash
nice -n 19 ionice -c3 find ~/PvZ-search/artifacts/student/frames_v2 \
  -type f -name tokens_v3.npz -delete
```

3. 启动分段迁移服务。脚本一次只转换一个文件，临时压缩文件写完才替换原文件。单段原始成员总量超过 1,990,000,000 字节时停止，保留原文件，确保含容器开销的额外文件低于 2 GB。当前抽样分段远小于此上限。启动时及累计转换约 1 GB 后执行 `df -B1 /mnt/c`，可用量低于 20,000,000,000 字节就退出。脚本使用单独迁移锁，阻止重复启动。

```bash
free -g
df -h /mnt/c
systemd-run --user --unit=task268-segments --collect \
  --property=MemoryMax=6G --property=MemorySwapMax=0 \
  --setenv=HTTPS_PROXY=http://127.0.0.1:7890 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  nice -n 19 ionice -c3 ~/.venvs/ml/bin/python3 \
  ~/PvZ-search/.claude/worktrees/cq/scripts/slim_training_segments.py \
  --data ~/pvz-logs/curriculum_state.pipeline/frames
journalctl --user -u task268-segments -f
```

每个转换文件会输出本文件和累计回收字节数。用 `systemctl --user stop task268-segments` 中断，源文件仍是完整旧代际或完整新代际。重新运行同一命令，会清理本迁移器遗留的临时文件并跳过已压缩段。迁移期间采集和训练继续运行。

4. 分段迁移结束后，用同一打包程序清理存量缓存。以下命令一次覆盖最多一百万个文件，分批让出锁。它会扫描已发布分段的小型来源列，耗时取决于文件目录操作，不进行帧重放或特征重算。未测全量删除墙钟，不能用五局缓存测量外推这个时间。

```bash
systemd-run --user --unit=task268-cache --collect \
  --property=MemoryMax=6G --property=MemorySwapMax=0 \
  --setenv=HTTPS_PROXY=http://127.0.0.1:7890 \
  --setenv=OPENBLAS_NUM_THREADS=1 --setenv=OMP_NUM_THREADS=1 \
  nice -n 19 ionice -c3 ~/.venvs/ml/bin/python3 \
  ~/PvZ-search/.claude/worktrees/cq/scripts/pack_training_rows.py \
  --data ~/pvz-logs/curriculum_state.pipeline/frames \
  --cache-dir ~/pvz-logs/curriculum_state.pipeline/cache \
  --prune-only --prune-cache-limit 1000000
journalctl --user -u task268-cache -f
```

输出 `cache_pruned/cache_bytes` 是实际删除数量与字节数。若只清某份快照覆盖的缓存，可额外传 `--out /完整路径/某打包索引.npz`。

## 回退

先停止两个维护服务即可停止后续转换和存量删除，已完成的压缩段是标准 NPZ，源帧未动。恢复采集旁车时，在组边界通过原启动流程重新带上 `--obs-v3`。已删旁车按单局入口回填，已删特征缓存由现有打包器按需重建，日常训练继续复用已发布段。

若只回退采集或剪枝行为，保留本次 shared-v2 读取兼容改动，可直接使用已压缩段。**整体切回旧读取器之前，需要在组边界把当前快照的索引签名绑定到压缩文件的新大小和 mtime，否则旧读取器会全量重建。** 以下命令只更新小型索引，不读取数组，不改 `.bin`。`PVZ_PACK` 必须指向准备交给旧读取器的已发布快照索引：

```bash
export PVZ_DATA="$HOME/pvz-logs/curriculum_state.pipeline/frames"
export PVZ_PACK="当前快照打包索引的绝对路径"
PYTHONPATH="$HOME/PvZ-search/.claude/worktrees/cq/python" ~/.venvs/ml/bin/python3 - <<'PY'
import json, os
from pathlib import Path
import numpy as np
from pvz_training_pack import shared_lock
data, pack = Path(os.environ['PVZ_DATA']), Path(os.environ['PVZ_PACK'])
with np.load(pack, allow_pickle=False) as stored:
    selected = {str((pack.parent / name).resolve()) for name in stored['segments'].tolist()}
with shared_lock(data):
    marker = data / '.merge/shared-v2/index.json'
    index = json.loads(marker.read_text())
    rebound = {}
    for signature, record in index['segments'].items():
        name = json.loads(signature)[0]
        if name in selected:
            stat = Path(name).stat()
            record['stamp'] = [stat.st_size, stat.st_mtime_ns]
            rebound[json.dumps([name, *record['stamp']])] = record
    index['segments'] = rebound
    temporary = marker.with_suffix('.building')
    temporary.write_text(json.dumps(index))
    temporary.replace(marker)
PY
```

随后再由 deploy26 切换旧代码。旧读取器仍保留其原有的集合变化重建规则，后续切换其他历史快照时会按旧行为处理。回退不需要把压缩段解压回 41 GB。

## 改动与验收范围

修改 `pvz_training_pack` 的共用 NPZ 写入和共享锁，新增 `slim_training_segments.py`。训练器增加压缩无关的成员身份识别，队列冷内存估算采用原始成员字节。打包器扩展既有剪枝路径，启动器取消默认旁车采集。回填入口放在现有 `pvz_research_data`，恢复循环统一到 `drill_start.replay_actions`，并修正旧雾档案的恢复上下文。

Mac 上 11 项直接相关定向测试通过，涵盖共享追加、内容变化、压缩迁移、源段缺失、转换中断、cache 剪枝与重建、冷内存估算、旧雾档案上下文和启动命令。后续维护模式调整仅重跑直接 cache 用例，并用 CLI 小样本删除了 12 个缓存。均使用 `timeout 300` 和文件大小上限，没有整套运行受限测试，没有在 Mac 跑对局。

初次选择测试时还有两条旧断言失败：一条要求改变 `label_weight` 就新增分段，与基线 `pack_options()` 已将权重移至加载阶段的行为冲突，基线两种权重生成的选项相同。另一条对内存预留的期望为 18 个建造者，基线未改动的 `pack_memory_limit()` 实际计算为 16。两条旧断言未在本任务修改。

台式机实验单进程运行，使用 `nice -n 19 ionice -c3` 与 systemd `MemoryMax=6G/MemorySwapMax=0`，代理通过 `--setenv` 传入。每次实验前查看 `free -g` 和 `/mnt/c` 空间，可用内存约 13–14 GiB，C 盘约 27 GB。分段副本与共享文件的同时占用低于 300 MB，结束前剩余实验目录 42,739,149 字节，已取回测量文件，并已删除整个实验目录。
