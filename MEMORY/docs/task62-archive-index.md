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
| student_frames_v1 | `~/PvZ-search/artifacts/student/frames` | `archive/2026-10-24/student_frames_v1/student_frames_v1.MANIFEST.json` | `782c24f44ac9c2ec6847580c683fd10bd58ccb10` | 53004/17786124845 | `archive/2026-10-24/student_frames_v1/student_frames_v1.tar.part001` 4000000000 B / `176968aeec4b527bdc40f9462462a760bc64488b992d2ab08d89601770f3acb9`；`archive/2026-10-24/student_frames_v1/student_frames_v1.tar.part002` 4000000000 B / `1135d4d5b4fb6967d4faf374d6c935d6d76a80723bf76f556dd1894275341d17`；`archive/2026-10-24/student_frames_v1/student_frames_v1.tar.part003` 4000000000 B / `af2cadb03a1b3a8affa83181fcdfb57f707c0fd0f2d0d58adc9e28a89b88c202`；`archive/2026-10-24/student_frames_v1/student_frames_v1.tar.part004` 4000000000 B / `81ad5cc5d742313db47c0a5072204ead68c0933a42e99080eb42fc7b475dce64`；`archive/2026-10-24/student_frames_v1/student_frames_v1.tar.part005` 1924782080 B / `8d1b05a9366fa6968008db2d989a1e471d066a55361644b9f525c42f45442033` |
| worktree_advgap | `~/PvZ-search/.claude/worktrees/advgap` | `archive/2026-10-24/worktrees/worktree_advgap.MANIFEST.json` | `6529e3c1d3df9caad3e45c1bb6012ddf3f1058db` | 49/26582014 | `archive/2026-10-24/worktrees/worktree_advgap.tar` 26695680 B / `76bcbff2fcd908e0610b5dda65dc4e8cd4ce1ccf2070195be2f62600b4f564c2` |
| worktree_b3 | `~/PvZ-search/.claude/worktrees/b3` | `archive/2026-10-24/worktrees/worktree_b3.MANIFEST.json` | `3abfd4eba6ebed56dcfdbc785d08a08c15aa8af0` | 4192/964062552 | `archive/2026-10-24/worktrees/worktree_b3.tar` 973547520 B / `42d42ab4e8eb38df4cd7d182f9b159d7cb038d47119f23974fc23e387a4f9be0` |
| worktree_curr | `~/PvZ-search/.claude/worktrees/curr` | `archive/2026-10-24/worktrees/worktree_curr.MANIFEST.json` | `7d28d6efc21c60aa1c154ef11b018fbc247a4370` | 231/34510431 | `archive/2026-10-24/worktrees/worktree_curr.tar` 35112960 B / `c7bb328e71d3270c7e44b8cdbd676a862fea3fb382c904418fe166f0d7dab243` |
| worktree_curr39 | `~/PvZ-search/.claude/worktrees/curr39` | `archive/2026-10-24/worktrees/worktree_curr39.MANIFEST.json` | `0ebd46e3d55c9bcc7ad46006859e5735b6b2ffaa` | 11/12087722 | `archive/2026-10-24/worktrees/worktree_curr39.tar` 12124160 B / `a50c1935b3f73b0bf89bbaf6b6dd199b101ca4f650a0a2959125ec546c300436` |
| worktree_fog | `~/PvZ-search/.claude/worktrees/fog` | `archive/2026-10-24/worktrees/worktree_fog.MANIFEST.json` | `34c8df7f9cb8109634c7984ae9eae88dd9e672f1` | 27583/326303754 | `archive/2026-10-24/worktrees/worktree_fog.tar` 385177600 B / `ce9f64b5999f8f5746846aa286da920be62ed5aa7cdc003eefdb98210fd66595` |
| worktree_forks | `~/PvZ-search/.claude/worktrees/forks` | `archive/2026-10-24/worktrees/worktree_forks.MANIFEST.json` | `fcae351a8a3feed9b52870b17adc65c51eea36af` | 22108/402432091 | `archive/2026-10-24/worktrees/worktree_forks.tar` 449228800 B / `993cd06bef160833880b07a03417ba0174aafdf6536fbaebe040d52413f32345` |
| worktree_identity | `~/PvZ-search/.claude/worktrees/identity` | `archive/2026-10-24/worktrees/worktree_identity.MANIFEST.json` | `b860b84b75e56f829823dd84825cdcb15a00d244` | 701/157892834 | `archive/2026-10-24/worktrees/worktree_identity.tar` 159610880 B / `1cd713fe5e41b4dd3d42cadf0a47bfd63117955df0af2651e064b16e393d722c` |
| worktree_keep | `~/PvZ-search/.claude/worktrees/keep` | `archive/2026-10-24/worktrees/worktree_keep.MANIFEST.json` | `80ec6f9744e49fa3147efe2253ab2cd9a7d13847` | 235/39815068 | `archive/2026-10-24/worktrees/worktree_keep.tar` 40407040 B / `67a32a42c2307e802145f96efa3e8477e5eff44a2affd1bd00fd8f2044a71a6a` |
| worktree_l20diag | `~/PvZ-search/.claude/worktrees/l20diag` | `archive/2026-10-24/worktrees/worktree_l20diag.MANIFEST.json` | `045b451d95ead555609cc82ff16bf67dac89d52b` | 207/24172902 | `archive/2026-10-24/worktrees/worktree_l20diag.tar` 24750080 B / `ea627d5bd3de2c477606ea3d71eeb4a3de946841643454c33dd962ae74ac0773` |
| worktree_l20gap | `~/PvZ-search/.claude/worktrees/l20gap` | `archive/2026-10-24/worktrees/worktree_l20gap.MANIFEST.json` | `a3ed1f1a14f944bd38a3df29175321b114e39541` | 219/36286337 | `archive/2026-10-24/worktrees/worktree_l20gap.tar` 36935680 B / `bfb74d21f59bdeed46fe6a3fc66125cd5ccbc88ca063a80b46551516b0742955` |
| worktree_macpool | `~/PvZ-search/.claude/worktrees/macpool` | `archive/2026-10-24/worktrees/worktree_macpool.MANIFEST.json` | `7a43f56302a431cd8a5dc1f751f66ecc4ed63830` | 115/12427084 | `archive/2026-10-24/worktrees/worktree_macpool.tar` 12687360 B / `a8426e7605950f08cabfc74146047dfb23e793c927abd4f4183a2a1696f12461` |
| worktree_n10p | `~/PvZ-search/.claude/worktrees/n10p` | `archive/2026-10-24/worktrees/worktree_n10p.MANIFEST.json` | `1a437b2c2eeff7f8a4b3d5c69a695138e66c5b6a` | 207/27099956 | `archive/2026-10-24/worktrees/worktree_n10p.tar` 27627520 B / `88593c61286c3766050331ee3c9bf432118d94e863675dcb49c7f8f5669dcc01` |
| worktree_night | `~/PvZ-search/.claude/worktrees/night` | `archive/2026-10-24/worktrees/worktree_night.MANIFEST.json` | `8f54f30fb196b50990d13c66c48f03d371940bca` | 376/27993973 | `archive/2026-10-24/worktrees/worktree_night.tar` 28979200 B / `bed9e2faee68ccb058ea416d6aac2e1ee2a9d782a7252d8d8020b0f2155fd239` |
| worktree_nightn10 | `~/PvZ-search/.claude/worktrees/nightn10` | `archive/2026-10-24/worktrees/worktree_nightn10.MANIFEST.json` | `b015518230e9f2bd3eb07b82cbd7632146b7812e` | 171/26113324 | `archive/2026-10-24/worktrees/worktree_nightn10.tar` 26552320 B / `ca50841d6ab600a0b6310eb16b24cb08524f57ae44a3f402f40ca81cc73965d4` |
| worktree_pool | `~/PvZ-search/.claude/worktrees/pool` | `archive/2026-10-24/worktrees/worktree_pool.MANIFEST.json` | `3a3208eef2238334ec373374e725129154fa737f` | 263/26219196 | `archive/2026-10-24/worktrees/worktree_pool.tar` 26900480 B / `90a455c4e1368b089e1819af64b7746589faa3bc65f0b63609aa5e0e6293748f` |
| worktree_qcalloc | `~/PvZ-search/.claude/worktrees/qcalloc` | `archive/2026-10-24/worktrees/worktree_qcalloc.MANIFEST.json` | `24001d3e6eead6c2cf7199d70ba211325a23d211` | 34/957705 | `archive/2026-10-24/worktrees/worktree_qcalloc.tar` 1044480 B / `e568b679d714e8dde828f8f694ded742c376e080f5741d6ec1f807976c04df6e` |
| worktree_qcbench | `~/PvZ-search/.claude/worktrees/qcbench` | `archive/2026-10-24/worktrees/worktree_qcbench.MANIFEST.json` | `efb777252643e418faab3390648b4c34ae8939db` | 4294/585443430 | `archive/2026-10-24/worktrees/worktree_qcbench.tar` 596633600 B / `018b5230194ae558c2cbc56c787e57c6d00fe94f2e8cef72e4b3a1e3d9ffeeeb` |
| worktree_qcpipe | `~/PvZ-search/.claude/worktrees/qcpipe` | `archive/2026-10-24/worktrees/worktree_qcpipe.MANIFEST.json` | `ce6d6043bfb317af276f7f037c471557c6f0dc57` | 1210/116463836 | `archive/2026-10-24/worktrees/worktree_qcpipe.tar` 119982080 B / `20b30e1b9ca646d5db4014495b1bd8800478c2ccc6751579763f26d9bd6dd91b` |
| worktree_recipe | `~/PvZ-search/.claude/worktrees/recipe` | `archive/2026-10-24/worktrees/worktree_recipe.MANIFEST.json` | `5d0738c5eba4c2dc7b2ffacb13c8e0b5a29f8e9c` | 882/102022827 | `archive/2026-10-24/worktrees/worktree_recipe.tar` 104120320 B / `f8afb37f7aa4717f820e16dbd4b570f9539007dcc4ead0975aa4a6c9806368bb` |
| worktree_roof | `~/PvZ-search/.claude/worktrees/roof` | `archive/2026-10-24/worktrees/worktree_roof.MANIFEST.json` | `d671c02fc8e8f115e36909cd3f471eba3ec5ba88` | 5103/538557744 | `archive/2026-10-24/worktrees/worktree_roof.tar` 550113280 B / `76910eeee12fedd03edfce1a3f53b0278016cf4731c7515689a316cb022ffc40` |
| worktree_teacher | `~/PvZ-search/.claude/worktrees/teacher` | `archive/2026-10-24/worktrees/worktree_teacher.MANIFEST.json` | `602de53fd8994cf8921515e61a0a9f59e12695df` | 199/24946736 | `archive/2026-10-24/worktrees/worktree_teacher.tar` 25456640 B / `acde91a7efdda8cb7cc1fe8049e9eb42c3b3e613dd682d545c9e6c9bc16015ca` |
| worktree_termmerge | `~/PvZ-search/.claude/worktrees/termmerge` | `archive/2026-10-24/worktrees/worktree_termmerge.MANIFEST.json` | `aee1892428775e211008f520c03b93c9afac8e52` | 268/5376320 | `archive/2026-10-24/worktrees/worktree_termmerge.tar` 6031360 B / `cb3155489cedb090b05d529f4686053968b35ef9f9050f78d919c987c81e8c6f` |
| worktree_trainmem | `~/PvZ-search/.claude/worktrees/trainmem` | `archive/2026-10-24/worktrees/worktree_trainmem.MANIFEST.json` | `5f826d3eca0e1fffc1ae3bef028a33afa8397f82` | 161/3325306 | `archive/2026-10-24/worktrees/worktree_trainmem.tar` 3676160 B / `4ce3a69cea72ef9c86c00a36152734bd73e2c82e0a8ac8c4f3c8f81303b14501` |
| worktree_vnet | `~/PvZ-search/.claude/worktrees/vnet` | `archive/2026-10-24/worktrees/worktree_vnet.MANIFEST.json` | `e3b3b66bf77884e997574fad144e2e28ab532664` | 1395/1206276479 | `archive/2026-10-24/worktrees/worktree_vnet.tar` 421990400 B / `669fccf6dba80380843ac455b1ac171de6625bd1af12bbc3fdd1c7f54a7c54df` |

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

### 可直接复用的下载、回验与恢复命令

从上表复制对应的完整 MANIFEST 路径及 revision 到两个变量。本机实测代理大文件传输显著降速，下例仅为这一条恢复命令清除代理变量、采用直连；不会修改 shell 配置或其他进程。其他机器按实际网络选择连接方式。此脚本强制下载、按部件顺序 SHA256/字节数回验，随后流式送入 tar 恢复到 `$HOME`，不会再拼出第二个完整 tar。恢复工作树的代码先按该归档 MANIFEST 的 `worktree_head` 用 Git 取回；该归档只恢复其 ignored/untracked 现场。

```bash
export ARCHIVE_MANIFEST='archive/2026-10-24/hard_branch.MANIFEST.json'
export ARCHIVE_REVISION='<上表对应的完整 revision>'
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY='*' no_proxy='*' python3 - <<'RESTORE_PY'
import os, json, hashlib, pathlib, shutil, subprocess
from huggingface_hub import hf_hub_download
stage = pathlib.Path('task62-restore')
kwargs = dict(repo_id=os.environ['PVZ_HF_REPO'], token=os.environ['HF_TOKEN'],
              revision=os.environ['ARCHIVE_REVISION'], local_dir=str(stage), force_download=True)
manifest = json.loads(pathlib.Path(hf_hub_download(filename=os.environ['ARCHIVE_MANIFEST'], **kwargs)).read_text())
def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()
parts = []
for item in manifest['parts']:
    part = pathlib.Path(hf_hub_download(filename=item['remote'], **kwargs))
    assert part.stat().st_size == item['bytes'] and sha(part) == item['sha256']
    parts.append(part)
home = pathlib.Path.home()
with subprocess.Popen(['tar', '-xpf', '-', '--exclude=TASK62_MANIFEST.json', '-C', str(home)], stdin=subprocess.PIPE) as tar:
    for part in parts:
        with part.open('rb') as stream:
            shutil.copyfileobj(stream, tar.stdin, 4 * 1024 * 1024)
    tar.stdin.close()
    assert tar.wait() == 0
for item in manifest['files']:
    target = home / item['path']
    if item['type'] == 'file':
        assert target.stat().st_size == item['bytes'] and sha(target) == item['sha256']
    elif item['type'] == 'symlink':
        assert target.is_symlink() and os.readlink(target) == item['target']
for item in manifest.get('excluded_shared', []):
    target, source = home / item['path'], home / item['shared_with']
    assert source.is_file() and source.stat().st_size == item['bytes']
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(source, target)
print('恢复完成；部件与普通文件 SHA256/字节数、软链目标已验证')
RESTORE_PY
```

完成恢复核验后，可按 MANIFEST 列出的明确路径删除下载暂存。`TASK62_MANIFEST.json` 是 tar 内附原始文件清单，完整部件 SHA256 以 HF 外置 MANIFEST 为准。

## 待完成：HF 私有存储额度阻塞

以下为本地已逐文件 SHA256/字节数验证的预备归档，**没有成功的 HF 提交和下载回验，不计入上方完成清单，不得据此删源数据**。临时 tar 副本已清，源证据和文件级 MANIFEST 保留，获得配额后可重建。vnet 采用 GNU sparse PAX 1.0，将零区表示为稀疏映射；原始逻辑文件完整，GNU tar 恢复时自动处理，SHA256 不变。

| 待传工作树 | 普通文件数 | 预备 tar B | SHA256（预备版，重建后更新） | 拟 HF 路径 | 本地文件级 MANIFEST |
|---|---:|---:|---|---|---|
| v5 | 11178 | 2543544320 | `aebaa6ac8148c8ceb7cdc20a598326e6c952b2cb34f62ce8019bf766ec6ff1cb` | `archive/2026-10-24/worktrees/worktree_v5.tar` | `/home/newbiexvwu/.task62-cleanup-state/worktree_v5.MANIFEST.json` |
| zones | 1559 | 267304960 | `c06dc2e0dc24a7a8c14890594b4309c27288acff827a5bbe97d1df7be9f95902` | `archive/2026-10-24/worktrees/worktree_zones.tar` | `/home/newbiexvwu/.task62-cleanup-state/worktree_zones.MANIFEST.json` |

31 组已完成归档的固定 revision 与回验记录均已核对；当前读回不受配额阻塞影响。v5 的部分训练数组链接指向已退役的 vnet，取回对应 vnet 归档后可恢复这些离线引用。

最新主控收口（2026-10-24T20:13:24+08:00）：停止上传重试并等待；额度升级或远端清理由用户决定。待上传未删的源路径与当前实占如下（整棵工作树，`du -s -B1`）：

| 保留的本地路径 | 实占 B | 实占 GiB | 状态 |
|---|---:|---:|---|
| `/home/newbiexvwu/PvZ-search/.claude/worktrees/v5` | 2,570,493,952 | 2.394 | 未上传成功、未下载回验；源证据与工作树均保留 |
| `/home/newbiexvwu/PvZ-search/.claude/worktrees/zones` | 293,486,592 | 0.273 | 未上传成功、未下载回验；源证据与工作树均保留 |
