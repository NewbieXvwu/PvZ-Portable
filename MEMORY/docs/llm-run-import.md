# LLM 存档一键入库（task89）结论

2026-10-26。入口：`scripts/import_llm_run.py`，测试：`python/test_import_llm_run.py`（接续复核后 30 项）。

## 用法

```bash
P=/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport
PVZ_NO_POPUP=1 /home/newbiexvwu/.venvs/ml/bin/python3 $P/scripts/import_llm_run.py /home/newbiexvwu/pvz-logs/task89/llm_free_out_numeric.zip --dest $P/artifacts/llm_demo/raw --executable $P/build/pvz-portable
```

常用可选项：`--report <md>`（报告落盘）、`--scan-out <json>`（task71b 扫描作业清单）、
默认扫描清单落在 `dest/scan_jobs_<日期>.json`；`--scan-out` 可覆盖。`--seed-ledger` 默认
`artifacts/llm_demo/scan_seed_ledger.json`，CLI 更换 dest 时仍共用此账本。
`--manifest`（任务清单，默认 `configs/human_demo_tasks.json`）、`--executable`（引擎二进制，
本工作树 `build/pvz-portable` 需先 `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`）、
`--no-verify`（跳过引擎校验）。依赖 `/home/newbiexvwu/.cache/pvz-research-resources`（main.pak + properties）。

## 流程与实现要点

1. **安全解压**：拒绝绝对路径、盘符、`..`、符号链接；输入只读，校验在 `_staging` 副本上做。
2. **发现**：rglob `meta.json`，`schema == episode_archive_v1` 且同目录有 `ticks.jsonl`。文件名不参与判断。
3. **自洽检查**：task vs environment_task 的 level/seed、task.id 编码的关卡号、deck 非空、
   outcome 胜负自洽、倍率一致、安全 task_id、`operations` 与 ticks.jsonl 行数、`frame_count` 与 frames.jsonl 行数、包内重复。
   原生传送带允许 `deck: []`，普通任务仍要求非空卡组。
4. **引擎校验**：`human_demo.verify(..., strict_hashes=False)` 重放（操作、每步 tick/ok、终局一致；不比逐位 state_hash，
   task82 已证差异来自未初始化字段）。实测 58k tick 的局约 3 秒，102 份约 8 分钟。
5. **分档（task81d）**：full = 通关且第 10 波检查点 ≥5/8；half = 通关不稳健；loss 只入报告。
   免费模型的存档没有 robustness 检查点记录 → 12 份通关全部 half（无 full），符合预期。
6. **新族登记**：种子不在原族 → `<base>_s<seed>`（task87 命名），卡组从 meta 读，按现有排版
   一行一个插进清单尾部，支持同批多个新族，并同步登记 DESIGN.md；已登记的族与种子幂等跳过（`registered` 为空）。
7. **扫描清单**：full/half 的局 × 12 决策点（覆盖前 94%，避开终局读档点）× 8 未来种子 ×
   老师/学生两臂。每个作业组含 8 个未来，同档不同点及两臂共用这 8 个未来。
   种子检查使用 task88 已提交的全机盘点快照（`MEMORY/docs/llm-import-seed-inventory.json`，含源提交，
   并保留新门禁池 [100M,101M)）、当前 configs/DESIGN 与本入口分配账本。
   每块单独避让（默认 84M 起，步长 1000、每档预留 96）；加锁、原子更新账本，跨导入不重叠，同档重跑复用原块。
   后续新增登记段应同步当前清单/盘点；不同机器使用同一账本。只生成不运行。
8. **报告**：逐份表（任务/种子/波/操作/verify/分档/稳健性/补救/新族）+ 各层汇总；
   另在 dest 写 `llm_import_<日期>.json` 索引；有失败/拒绝时退出码 1。

## 端到端实测（免费模型 12 份通关 + 90 份失败，共 102 份）

- 102/102 引擎校验通过；12 份通关入库 `dest/seed_<seed>/<task_id>/`，名单与包内 `report.md`
  及 task87 的表完全一致（L46–49 × 6 种子）。
- task87 已登记的 8 个 `_s2026` 族幂等跳过，不重复登记。
- 扫描作业 **288 组，2304 条续走**（12 份 × 12 点 × 2 臂，每组 8 个未来）。
- 坏输入：zip 路径穿越 → 解压前 raise，exit 1，零落盘；meta.operations 与 ticks 行数不符 →
  该存档标「拒绝」入报告，exit 1。

## Codex 接续复核

原 102/102 引擎实测复用 CodeBuddy 的日志和索引，保存在 ignored 的
`artifacts/task89/verification/`；没有重跑整个引擎验收。
对 SHA256 `d199f51129eec00b40d52caf37cbb2e03e84e93a90da97ddc4d89c67d1f7a801` 的原 zip
重新走更新后的发现/自洽/入库/分档/扫描路径（这一步跳过引擎），102 行任务、种子、波数、操作数、分档
逐项吻合旧索引，12 份 half / 90 份 loss，无新增族。12 个预留块逐对不相交，也与盘点所有登记段不相交。
随后用当前显式关闭哈希比较的验证调用抽验一胜一负，均通过；最多一个游戏进程，未运行接手扫描。
最终 CLI 检查：真实目录输入退出 0，并在不传 `--scan-out` 时保存 24 组扫描清单；
路径穿越 zip 和 meta/存档不一致各退出 1。30 项测试覆盖新族登记、分档、输入只读、种子冲突及幂等。
紧凑证据见 [llm-run-import-evidence.json](llm-run-import-evidence.json)。

## 已知边界

- verify 失败的通关局会被挪到 `dest/_rejected/` 并在报告中标注，不静默丢弃。
- 同 (族,种子,操作数,initial_state_hash) 的包内重复会标「拒绝」；仅前三个相同而 state hash
  不同的存档视为不同局（GPT Pro 交回的包里存在这种合法重复）。
- 目标已有存档时同时比较 ticks、frames 和 meta（只忽略验证器字段）；稳健性等 meta 改变会明确报冲突，避免报告分档与库内 meta 不一致。
- 引擎二进制不入 git（AGENTS.md），换机器要先编译；资源目录路径写死为默认值，可用
  `--resource-dir` 覆盖。
