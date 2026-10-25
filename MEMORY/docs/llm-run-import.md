# LLM 存档一键入库（task89）结论

2026-10-26。入口：`scripts/import_llm_run.py`，测试：`python/test_import_llm_run.py`（接续复核后 22 项）。

## 用法

```bash
python3 scripts/import_llm_run.py <out.zip 或解压目录> --dest artifacts/llm_demo/raw
```

常用可选项：`--report <md>`（报告落盘）、`--scan-out <json>`（task71b 扫描作业清单）、
`--manifest`（任务清单，默认 `configs/human_demo_tasks.json`）、`--executable`（引擎二进制，
本工作树 `build/pvz-portable` 需先 `cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build`）、
`--no-verify`（跳过引擎校验）。依赖 `/home/newbiexvwu/.cache/pvz-research-resources`（main.pak + properties）。

## 流程与实现要点

1. **安全解压**：拒绝绝对路径、盘符、`..`、符号链接；输入只读，校验在 `_staging` 副本上做。
2. **发现**：rglob `meta.json`，`schema == episode_archive_v1` 且同目录有 `ticks.jsonl`。文件名不参与判断。
3. **自洽检查**：task vs environment_task 的 level/seed、task.id 编码的关卡号、deck 非空、
   outcome 胜负自洽、倍率一致、安全 task_id、`operations` 与 ticks.jsonl 行数、`frame_count` 与 frames.jsonl 行数、包内重复。
   原生传送带允许 `deck: []`，普通任务仍要求非空卡组。
4. **引擎校验**：`human_demo.verify` 重放（操作、每步 tick/ok、终局一致；不比逐位 state_hash，
   task82 已证差异来自未初始化字段）。实测 58k tick 的局约 3 秒，102 份约 8 分钟。
5. **分档（task81d）**：full = 通关且第 10 波检查点 ≥5/8；half = 通关不稳健；loss 只入报告。
   免费模型的存档没有 robustness 检查点记录 → 12 份通关全部 half（无 full），符合预期。
6. **新族登记**：种子不在原族 → `<base>_s<seed>`（task87 命名），卡组从 meta 读，按现有排版
   一行一个插进清单尾部，支持同批多个新族，并同步登记 DESIGN.md；已登记的族与种子幂等跳过（`registered` 为空）。
7. **扫描清单**：full/half 的局 × 12 决策点（覆盖前 94%，避开终局读档点）× 8 未来种子 ×
   老师/学生两臂。未来种子块机械盘点 `configs/*.json` + DESIGN.md 的大数后自动分配
   （默认 84.0M 起、每份 stride 1000、块 96），实测与现有全部段不相交。只生成不运行。
8. **报告**：逐份表（任务/种子/波/操作/verify/分档/稳健性/补救/新族）+ 各层汇总；
   另在 dest 写 `llm_import_<日期>.json` 索引；有失败/拒绝时退出码 1。

## 端到端实测（免费模型 12 份通关 + 90 份失败，共 102 份）

- 102/102 引擎校验通过；12 份通关入库 `dest/seed_<seed>/<task_id>/`，名单与包内 `report.md`
  及 task87 的表完全一致（L46–49 × 6 种子）。
- task87 已登记的 8 个 `_s2026` 族幂等跳过，不重复登记。
- 扫描作业 288 条（12 份 × 12 × 8 × 2）。
- 坏输入：zip 路径穿越 → 解压前 raise，exit 1，零落盘；meta.operations 与 ticks 行数不符 →
  该存档标「拒绝」入报告，exit 1。

## 已知边界

- verify 失败的通关局会被挪到 `dest/_rejected/` 并在报告中标注，不静默丢弃。
- 同 (族,种子,操作数,initial_state_hash) 的包内重复会标「拒绝」；仅前三个相同而 state hash
  不同的存档视为不同局（GPT Pro 交回的包里存在这种合法重复）。
- 引擎二进制不入 git（AGENTS.md），换机器要先编译；资源目录路径写死为默认值，可用
  `--resource-dir` 覆盖。
