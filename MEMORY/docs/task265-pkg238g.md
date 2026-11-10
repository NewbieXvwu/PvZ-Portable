# task265：GPT 包 238g

238g 复用 238f 的静态引擎、资源和题库，合入 `LLMPlay_2026-11-10_043258_HKT_out.zip.001` 的结论记录。修复恢复局的认领续期，补全 scenario 出题的巨人行限制。分支 `pkg238g`，代码提交 `eae76cc3`。

## 认领修复

认领键与 `workflow._slot(task_id, seed)` 保持一致：普通整局使用任务 id 与原局种子，resume 使用 `resume_id` 与原局种子，drill/scenario 使用题目 id 与 `future_seed`。`Game._claimed` 保存实际认领二元组，续期和释放共用它。

短题的原始存档只确定底局，题面中的默认未来可能与本次派发不同，因此在 `reseed(item['future_seed'])` 时绑定本次认领。已打过的短题存档及 `save()` 文件依据实际 reseed 操作恢复认领。resume 在 `load_archive()` 恢复元数据后立即续期。

续期从仅换波调用的 `_persist_best()` 移到 `_apply()`，同一波内每步推进也续期。导出、放弃和关闭沿用统一释放路径，内部试打 `_internal` 不认领。顶层 `llmplay.next_task(ttl=...)` 现直接透传参数，`__init__.py` 原有导出自动生效。

Mac 仅运行轻量定向单测：

```bash
ulimit -f 2000000
timeout 300 uv run --python 3.12 --with pytest --with numpy --with py7zr \
  python -m pytest python/test_llmplay_decks.py \
  -k 'task265 or restoring_claim or task248_concurrent_claim' -q
```

**5 项通过**。新增三类恢复题的假时钟回归，认领 TTL 与续期均为 10 秒，推进到第 12 秒后再次领题，当前题仍被保留。短题特意领取第二个未来种子，覆盖题面默认未来与原局种子均不能代替本次派发种子的情形，并覆盖关闭释放、保存恢复和内部试打。

## 屋顶 spawn 的根因与处理

类型 23 是巨人。`Board::EnvironmentCanSpawnZombie()`（`src/Lawn/Board.cpp:2693`）沿用原生出怪规则：非生存模式第 0 行禁止巨人及红眼巨人。L42 属于冒险模式，因此 `spawn(23, 0, 350)` 在创建僵尸前被拒绝。屋顶允许巨人，x=350 也合法。

台式机使用实际交付引擎复现 `scenario238_roof_032`、未来种子 `239000065`：`set_sun(325)=True`，r0 注入返回 `False`、id 为 `None`，同一起点改在 r1 注入返回 `True`、id 为 `66519049`。

scenario 底局没有固定的注入类型或行，威胁由 GPT 自己出题。任务书现规定冒险模式巨人避开 r0，屋顶使用 r1–r4，并要求注入成功后再声明目标。Python 未增加游戏规则判断，C++ 无需修改。

按 238f 题库扣除 11-10 交回，scenario 仅剩 **1 个底局、2 个种子项**：`scenario238_roof_032` 的 `239000064` 和 `239000065`。沿用 r0 巨人注入时这两项都会被拒绝，改用合法行即可继续，两项均保留未完成状态。

## carry 与打包

使用 `build_package.py --reuse-package` 复用 Mac `~/Downloads/pvz-llmplay-238f/llmplay_pkg/`，明确从该目录复制 resume、drill 和 scenario 存档。11-10 交回为单卷 ZIP，打包时通过 `.zip` 链接读取其进度及诊断，未解出或重放旧存档。合并后为 **15,594 条唯一存档记录**。

| 队列 | carry 已完成 | 238g 总数 | 剩余 |
|---|---:|---:|---:|
| drill | 2709 | 2781 | 72 |
| scenario | 598 | 600 | 2 |
| resume | 649 | 8457 | 7808 |

交回包的 drill 总数为 2709、resume 总数为 8392，238f 已新增 72 道雾关 drill 和 65 条 r160 resume，238g 保留这些新增题。队首为 `drill238_fog_174833d3`、未来种子 `263100009`，已完成条目由 carry 排除。

## 验收与交付

台式机启动前 `/mnt/c` 剩余 **30 GB**。在独立目录 `~/pvz-logs/task265/` 通过 `systemd-run --user` 后台运行，使用 `nice -n 19 ionice -c3`，内存上限 4 GB。后台任务继承现有代理后完成 HF 下载。

对从 HF 实际下载并解出的分卷运行一次 `accept_package.py`，退出码 **0**。`check_engine`、三类题、跨进程保存恢复和并发认领均通过。drill 等待至 900 tick 后判失败，导入器复验同判失败。scenario 正常导出，resume 推进到 tick 19140，连续两次导出前后均为 **15,596** 条记录。

接口包经 HF `transfer/task265/` 取回 Mac，两端取用后已删除远端。资源包从 238f 本地原样复制，全程未上传 HF。台式机独立目录和 Mac 打包临时目录已清理，未部署，未操作生产 `cq` 或采集冻结文件。

交付目录：`~/Downloads/pvz-llmplay-238g/`。两卷接口包需一起使用。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `llmplay_pkg.7z.001` | 20,000,000 | `d41366071936f2ab5335ab16df4ef8fca176f9289a546ddc1595838e8da622ec` |
| `llmplay_pkg.7z.002` | 285,359 | `e162344ee869e660300a4a07651a75f8a204df33a500fae31d3a600e0f5c5b2f` |
| `llmplay_res.zip` | 34,258,402 | `0005bb8e31ad2e2ca7649bc0b6128211ff13c8f1c333c2163984302d7d65d94e` |

静态引擎 SHA256 为 `1c3680f36aa5bd5fef97d035a97a1cdd9ca89fce2493a87f125c7bd507d29c07`，与 238f 相同。
