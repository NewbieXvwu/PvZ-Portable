# task258：生产状态文件瘦身

分支 `slimstate`，基于 `pkg238` 的 `4b983cb9`。仅修改代码并在 Mac 演练，未部署，未改台式机生产文件。

## 实际来源与改动

生产副本有 156 个 `training_jobs`，其顶层 `recipe` 全部已经是路径。约 388 MB 的任务数据主要来自 `timing.corpus_archives` 的历轮重复清单，合计 377,404,873 字节。修复针对这一实际来源，同时兼容旧的内嵌 recipe 格式。

所有任务配方路径读取统一经过 `job_recipe_path()`。加载旧状态时，内嵌 recipe 与对应轮次的 `recipe.json` 做一次内容比较，一致便改为路径，文件缺失则先原子写出。两份内容冲突时保留双方并报错，不覆盖现有配方。A/B 候选同样处理。

完整训练报告通过 `timing_path` 引用。已有 `complete.json` 与状态内报告一致时直接复用，缺失时先补写。如果两者不同，将状态中的版本保存为 `state_report.json`，保留完成文件。状态中的 `timing` 仅保留标量统计和门禁所需的 `version / approved / reasons / contract / winner_family / next_panel`。需要完整语料清单的晋升恢复通过 `job_report()` 按需读取，门禁证据和报告原文都保留在文件里。

状态中的 `log` 保留最近 20 轮。旧轮次写入各自的 `round_NNNNNN/collection.json`，已有相同文件直接复用，迁移中断后重启也不会重复追加历史。成功组的标签及首次出现轮次保存在精简的 `archived_labels`，保证完整训练标签集合和旧学生轮次的标签过滤结果保持一致。历史文件不参与每组状态重写。

保存时机和频率保持原样。首次加载完成迁移并保存，此后只处理新报告或刚退出最近 20 轮的日志，已经外置的报告不会在每组保存时重读或重写。

## 增长边界

`log` 的详细记录已限制为最近 20 轮。历史文件每轮写一次，其累计体积仍随轮次增加，但不再放大每组写入。

`archived_labels` 随实际采集标签增加，用于构建全部训练数据及恢复旧学生的标签范围，无法任意截断。它记录标签和最早轮次，没有历史组的完整字段。

`student_corpus_archives` 是当前学生实际训练过的档案集合，每次晋升替换，未逐轮追加。集合可能随训练语料增加而增长，不能截断，否则新旧语料的重放划分会改变。此次保留全部 110,933 项。其他任务的相同清单已移出状态，避免重复存储。当前这部分约 6.74 MB，仍是状态中最大的必要字段。

## 生产副本演练

台式机使用规定的 SSH/WSL 入口和 `nice -n 19`，只读生产状态与轮次报告，通过 HF `transfer/task258/` 传至 Mac。状态压缩包和报告包下载后均已删除远端文件。配方路径仅在 Mac 副本中重定位，报告内容保持原样。台式机没有运行测试。

| 项目 | 结果 |
|---|---:|
| 原始状态 | 448,465,824 字节（448.47 MB） |
| 迁移后状态 | 15,379,187 字节（15.38 MB，14.67 MiB） |
| 单次写入缩减 | 96.571% |
| `training_jobs` 迁移后 | 2,493,461 字节 |
| 最近 20 轮日志 | 1,901,042 字节 |
| 历史标签索引 | 326,725 字节 |
| 当前学生档案清单 | 6,738,457 字节 |
| Mac 首次迁移耗时 | 1.379 秒 |

156 个任务的轮次与状态等元数据保持一致，111 份完整报告逐份比较一致，门禁摘要字段一致。7,263 个训练标签及学生轮次过滤后的标签集合一致，当前学生 110,933 个档案保持一致，流水线生成的轮次配方字段一致。第二次加载返回相同状态，状态文件字节也完全相同。

从生产轮次目录取回的 61 份已有报告均未改写。副本补写了 51 份缺失历史报告，共 128,568,759 字节，另归档 138 轮日志，共 11,798,641 字节。这些是首次迁移的一次性写入。已有报告中有一份不属于该时刻状态里带 `timing` 的任务，所以报告文件总数与 111 份被引用报告相差一份。未遇到报告内容冲突。

台式机首次启动会进行同样的一次性比对与补写，实际耗时取决于当时磁盘负载。后续组边界只写瘦身后的状态，不重做这次历史迁移，也没有对局重放。

定向测试在 Mac 单进程执行，11 项通过，109 项未选中，耗时 2.04 秒：

```bash
ulimit -f 2000000
timeout 300 python3 -m pytest \
  python/test_slim_state.py python/test_curriculum_queue.py python/test_curriculum_pipeline.py \
  -k 'slim_state or pack_labels_union or run_reuses_trained_json or stale_comparator' -q
```

新增用例覆盖配方文件存在/缺失/冲突、报告文件存在/缺失/冲突、重复加载、重复保存不读取外置报告、日志归档中断后重启，以及从已批准任务恢复学生和完整语料清单。现有配方标签测试、训练快照复用测试与旧比较对象重新评估测试通过。

尝试收集 `test_qc_resilience.py` 时，基线在第 14 行导入了 `test_curriculum_pipeline` 中不存在的 `prepared_selection_without_real_games`，收集失败，未执行其中的用例。该问题未纳入本次改动。

## 上线

主控审查后，在台式机取回 `origin/slimstate`，使用现有 deploy26 的组边界部署入口：

```bash
cd ~/PvZ-search
git fetch origin slimstate
~/.venvs/ml/bin/python3 scripts/deploy26.py --revision origin/slimstate
```

新控制器首次加载自动迁移。保留 deploy26 输出的备份目录，并保留 `curriculum_state.pipeline/`：现在的历史报告与 `collection.json` 是恢复资料，不能作为临时缓存删除。首次迁移后查看状态文件大小即可，无需重放。

## 回滚

旧代码可以解析新状态的 JSON，但其 `pack_labels()` 不认识 `archived_labels`，会遗漏较早的训练标签，因此不能直接以新状态运行旧代码。deploy26 只恢复代码，不还原状态。应先在组边界停止控制器和看门狗，用当前状态及外置资料重建旧格式，再执行 deploy26 回滚。使用当前状态重建可保留上线后新增的种子进度，避免回退到部署前状态。

以下脚本在台式机执行，使用当前部署代码已有的停采函数和报告访问函数。它在全部资料读出后才原子替换状态文件。报告和历史文件继续保留。

```bash
cd ~/PvZ-search/.claude/worktrees/cq
nice -n 19 ~/.venvs/ml/bin/python3 - <<'PY'
import fcntl
import json
import sys
from pathlib import Path
sys.path[:0] = ['scripts', 'python']
import deploy26 as deployment
import curriculum_queue as queue

root = Path.home() / 'PvZ-search'
logs = Path.home() / 'pvz-logs'
worktree = root / '.claude/worktrees/cq'
with (logs / 'qc_deploy.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deployment.stop_watchdog(worktree, logs)
    deployment.stop_collection(worktree, logs, logs / 'task258-rollback-boundary.json')
    path = logs / 'curriculum_state.json'
    state = queue.read_json(path)
    for job in state.get('training_jobs', []):
        if job.get('timing_path'):
            job['timing'] = queue.job_report(job)
            del job['timing_path']
    directory = path.parent / (path.stem + '.pipeline')
    rounds = {}
    for history in directory.glob('round_*/collection.json'):
        entry = queue.read_json(history)
        rounds[entry['round']] = entry
    rounds.update({entry['round']: entry for entry in state['log']})
    state['log'] = [rounds[index] for index in sorted(rounds)]
    state.pop('archived_labels', None)
    temporary = path.with_suffix('.rollback-building')
    with temporary.open('w') as handle:
        json.dump(state, handle, indent=1)
        handle.write('\n')
    temporary.replace(path)
PY
~/.venvs/ml/bin/python3 scripts/deploy26.py --rollback ~/pvz-logs/deploy26_<此次部署的备份目录>
```

回滚重建会一次性恢复大状态文件，旧代码随之恢复原来的大文件写入行为。此流程仅用于撤回本次代码。
