# task62：2026-10-24 HF 归档索引

按主控 17:05 批准，tar/分片是本次退役归档方式；不是日常同步通道。仓库用 `$PVZ_HF_REPO`，鉴权用 `$HF_TOKEN`。每一行只有在固定 revision 下载后 SHA256 回验通过才登记；完整 MANIFEST 记录原始相对 home 路径、文件字节数/SHA256、软链目标、tar/分片字节数与 SHA256。tar 不跟随外部软链，取回依赖见文末说明。

| 名称 | 原位置 | HF MANIFEST 路径 | 固定 revision | 文件数/原始逻辑 B | tar 部件与 SHA256 |
|---|---|---|---|---|---|
| hard_branch | `~/PvZ-search/artifacts/hard/branch` | `archive/2026-10-24/hard_branch.MANIFEST.json` | `64688af9c6861b94bd74fa3d159265d8ab5ce25f` | 3/2452148684 | `archive/2026-10-24/hard_branch.tar` 2452162560 B / `0f881dadd2cdf5f096e6d3578341ae4ffc651cc6fbda00ee91a4af7ffa45b6ee` |
| hard_branch3 | `~/PvZ-search/artifacts/hard/branch3` | `archive/2026-10-24/hard_branch3.MANIFEST.json` | `474831852f2e4594a062e779f87715e6f337355a` | 96/1681885408 | `archive/2026-10-24/hard_branch3.tar` 1682094080 B / `c0082c38e22748135f6f5f19e0dbb159ca054273f8b2e891278bba17c400a0c8` |
| night | `~/PvZ-search/artifacts/night` | `archive/2026-10-24/night.MANIFEST.json` | `82108867dac86e39d5ad0bed729757eaf248d75c` | 4136/4048206862 | `archive/2026-10-24/night.tar.part001` 4000000000 B / `e4a3913aaeff8a8fc027f67da580c603e85862057ffcaafeabbc58869f2e2b76`；`archive/2026-10-24/night.tar.part002` 62126080 B / `2f557267e45fdc44cfe2d81a0d6d4e63caaa4216be324b75a3e39dc95dd206fd` |
| relabel | `~/PvZ-search/artifacts/relabel` | `archive/2026-10-24/relabel.MANIFEST.json` | `ba30a8018bb6354acf0745721e69ec8644de7b0f` | 4281/367775924 | `archive/2026-10-24/relabel.tar` 383887360 B / `73016be40513cd9d507812c8d213cab7b3f85d7fdced42cdafb43f9587299f0d` |
| portable_artifacts | `~/PvZ-Portable/artifacts` | `archive/2026-10-24/portable_artifacts.MANIFEST.json` | `6a6978e1446bf411a137b83e47d0e354066b52d1` | 5219/433859326 | `archive/2026-10-24/portable_artifacts.tar` 445194240 B / `a57e0774d8e08e86a2ea1165b216678985120d43bcafd05eeee177613ee60789` |
| pvz_lane | `~/pvz-lane` | `archive/2026-10-24/pvz_lane.MANIFEST.json` | `dab418c1b8d1bd9f45a5ffb8483d484cdae4e918` | 4102/653269613 | `archive/2026-10-24/pvz_lane.tar` 666961920 B / `0fa0d208c5dee86b9fe991d464fb7a0be39ef4f6366b628ac4c3c8035ff6ae9a` |

## 取回方法

先下载该行 MANIFEST 和其中列出的所有部件，固定该行 revision；逐部件校验 SHA256。若分片，按 MANIFEST 的 part 序号拼接成 tar。然后在恢复目标 home 目录执行 `tar -xpf <完整tar> -C "$HOME"`，按 MANIFEST 再验全部普通文件字节数/SHA256 和软链目标。归档含 `PvZ-search/...` 等相对 home 路径，不在缺少源依赖时直接启动训练。示例下载：

```python
import os
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id=os.environ["PVZ_HF_REPO"], token=os.environ["HF_TOKEN"],
    filename="<本表 MANIFEST 或部件路径>", revision="<本表固定 revision>",
    local_dir="./task62-restore", force_download=True)
```

`train_value_widen.py` 等保留原本默认路径，需要使用退役数据时先从 HF 取回。旧帧归档只含未被正式帧同 inode 覆盖的文件；恢复完整旧视图时，MANIFEST 的 `excluded_shared` 指明从 frames_v2 同路径重新链接的文件。工作树里的绝对软链保持原目标；conveyor/dw4/frames_v2 属于仍保留的生产源，v5 指向 vnet 的证据需同时取回 vnet 归档；forks/hard 旧帧引用需取回 student_frames_v1 的对应原始帧，或按索引记录的改链路径恢复。
