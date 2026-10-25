# 接手扫描与起点格登记自动化（task91）结论

> 2026-10-26。入口 `scripts/takeover_scan.py`（run / adaptive / register），测试
> `python/test_takeover_scan.py`。任务书：[task91-takeover-scan-runner.md](task91-takeover-scan-runner.md)。

## 用法

```bash
P=/home/newbiexvwu/PvZ-search/.claude/worktrees/llmimport
PY=/home/newbiexvwu/.venvs/ml/bin/python3

# 1) task89 入库（若尚未做）会顺带生成扫描清单 scan_jobs_*.json
# 2) 按清单跑两臂（断点续跑：重跑同一命令只补缺的未来）
$PY $P/scripts/takeover_scan.py run  $P/artifacts/llm_demo/raw/scan_jobs_20261026.json
# 2') 自适应取点（默认）：4 等距老师点 → 跨 50% 二分 → [50% 线, 顶档] 补齐网格档，
#     学生臂只打 50% 线与顶档两端。所有点都落在 task89 的 12 点网格上，可与 12 点结果直接比档。
$PY $P/scripts/takeover_scan.py adaptive $P/artifacts/llm_demo/raw/scan_jobs_20261026.json
# 3) 判读登记：老师 ≥50% 的存档生成阶梯写 demo_start；独立加锁账本分配 256 种子块；
#    既有起点格不覆盖（task87 的 _llm 族等），只核对与报告；同步 DESIGN 与测试期望。
$PY $P/scripts/takeover_scan.py register $P/artifacts/llm_demo/scan_out/results --dry-run   # 先看
$PY $P/scripts/takeover_scan.py register $P/artifacts/llm_demo/scan_out/results
```

生产老师参数与 qc 同款：学生 r11（`curriculum_state.pipeline/round_000011/student/model.pt`）、
叶值 v5、静态网 branch1、`--verify-samples 9 --student-candidates 2`；老师臂 β=1，学生臂 β=0（fast 臂）。
结果按存档写 `scan_out/results/<task_id>_s<seed>.json`，每局含胜负、末波、破防行与僵尸类型。

## 实现要点

- **断点续跑三层**：run 的 per-(存档,点,臂) 输出目录里，search_play 自己按 (arm, future_seed)
  跳过已完局；takeover_scan 再按结果文件里已有的完整点跳过整点。实测重跑 0.08s 零重跑。
- **登记族解析**（`resolve_target_task`）：按存档路径匹配清单里已有的 demo_start 最可靠；
  本机种子 LLM 局与人类示范共用 base task_id 时落 `base_llm` 族（task87 先例）；base 被别的
  存档占用则报错留人工，绝不静默覆盖既有起点格。
- **登记种子账本**：`artifacts/llm_demo/register_seed_ledger.json`，与 task89 扫描账本
  （`scan_seed_ledger.json`，84M 起、每份 1000 步长 96 块）同格式但独立文件；分配时合并
  清单/盘点/两个账本的所有已占段，登记块从 86M 起步长 1M。
- **阶梯规则**（task71b 同款）：档位全部 ≥50%（4/8 恰过线）；顶档取最晚的 8/8 稳定点，
  没有 8/8 就取最晚 ≥50% 点（报告注明）；顶档以上即使 ≥50% 也不入（不如稳定点）。
- **测试**：`python/test_takeover_scan.py` 17 项（阶梯规则、自适应取点顺序的合成驱动、
  族名解析、清单/测试期望的文件更新）。

## 回测（自适应 vs task87 的 12 点全量）

（进行中，本节跑完后补数）

## 端到端（task89 包 2 份，独立 dest/账本/临时清单，未来块 90M 起）

- **import**：全包 102 行走完，12 份通关入 `artifacts/e2e/dest/`，扫描清单 288 组、独立账本
  `artifacts/e2e/scan_seed_ledger.json`（90,000,000 起、每份 1000 步长 96），生产清单零改动。
- **run**：2 份 × 2 点 × 2 未来 × 两臂（`--decisions 1,906,996,1089` 按各存档网格取交集）。
- **register**（临时清单副本 + 临时 DESIGN/测试/账本）：两族写入 `demo_start`
  （base 被人类示范占用 → 正确回退 `_llm`；阶梯 [996,906]（d906 1/2 恰过线）与 [1089]，
  种子块 86M/87M）；`update_manifest` / `update_test_expectations`（期望集合插入）/
  `update_design`（§14.35 自动编号）产物逐项核对通过；登记账本两条 256 块记录正确。
- run 的 `--decisions` 现按每份存档的网格取交集（否则第二份会 KeyError）。
