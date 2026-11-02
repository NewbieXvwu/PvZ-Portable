# task223：门禁 v8 审计（校准跟随现任 + 胜局回归守卫）

分支 `gatev8`，基线 `0201635f`（含 task222 与 deploy26）。**未部署**，没有改生产状态。

## 根因

1. **校准跟着上一任学生走。** `next_panel` 原来只读 `outcomes['incumbent']`。
   r94 的 B 晋升后，写进 state 的 fog `low_win/rates` 仍是 r52 的（rate 0.0）；
   r95 开评时冻结的 calibration 就是这份旧数据：fog 8 个任务里 7 个 `low_win=True`、
   incumbent rate 0.0。r95 的现任实际是 r94，fog 已经不是 0%。
2. **波数终点盖住了胜局回归。** r95 的 96 个 fog 对里 84 个按 `final_wave/wave_count`
   计分；胜局 A 16 vs 现任 35（gain 4 / harm 23，精确 McNemar 单侧 p=1.55e-4），
   但旧 `decide` 只要 `wave_pairs>0` 就改用波数 Wilcoxon，胜局 p 没有进区级检验，
   于是 fog 没被否决。

> 任务书里的 “A 16 vs incumbent 29 / gain 6 harm 19 / p=.0073” 是 `ab_result.json`
> 的 **A−B** 对比；门禁晋升用的是 **A vs 现任（r94）**，存储 decision 为 16 vs 35、
> gain 4 / harm 23、p=1.55e-4。两者都显示 fog 显著退步，下面按门禁口径计算。

## 修复（同一套实现内）

- `pvz_promotion.next_panel(..., incumbent_side)`：晋升后使用**被晋升候选项在同一批门禁种子上的
  行**来更新 `low_win` 和倍率；未晋升时仍用 `incumbent`。`promotion_gate.evaluate` 把
  `winner` 传进去。没有新增对局，直接复用门禁自己的 games。
- `recalibrate_panel(pre_calibration, post_calibration, incumbent_side)`：只用 state 里已存的
  `rates` 和门禁前冻结的倍率重做一步楼梯；不重放、不重跑。
- `curriculum_queue._recalibrate_current_incumbent`：启动迁移。若 `gate_panel_student_round`
  不是当前 `student_round`，用该轮已批准 job 的 `contract.calibration` + `gate_panel.rates`
  重算，并写 marker。r95 的旧 state 在控制器重启后会从旧口径切到 r95（A）口径：
  fog L31/L34 变 `low_win=True`（A 0.0），L33/L37/L39 变 `False`（A .25/.58/.25）；
  倍率同步按 A 的同一批 rate 走一步。
- `decide` 的区级规则：非屋顶区同时有胜局和波数对时，若胜局分歧对 ≥
  `ZONE_WIN_GUARD_DISCORDANT=6`，以胜局 McNemar 为准；胜局 p <
  `ZONE_WIN_GUARD_ALPHA=.05` 直接否决（`significant_win_regression`），不再被平波的
  波数分数救回。只有胜局分歧对不足 6 时，波数才替代胜局。屋顶仍只用波数 Wilcoxon。

## r95 离线复算（无新对局）

用 r95 evaluation 目录里已存的 `manifest.json` + A/B/incumbent `games.jsonl` 重建配对，
调用修后的 `decide`：

| 口径 | 旧 | 新 |
|---|---|---|
| A vs r94 总体 | approved=True, +2.40pp | approved=False |
| fog 胜局 | 16 vs 35, gain 4/harm 23, p=1.55e-4 | 同左 |
| fog 波数 | 84 对, Wilcoxon p=4.37e-3 | 仍记录，但胜局分歧对 27 ≥ 6，端点用胜局 |
| 否决理由 | 无 | `significant_win_regression:fog`, `significant_zone_regression:fog` |
| Holm | fog 0.0175 > α=.00625，未触发 | fog 0.00062 < α=.00625，触发 |

结论：**A 在修后的规则下不会通过 r95**。B 的离线复算仍是总体负、不通过。
复算只读 JSON/JSONL，没有起引擎；临时 bundle 与脚本已删。

## 测试

`ulimit -f 2000000`、`timeout 300`、Mac 单 pytest：

```
python3 -m pytest python/test_gate_v8_panel.py python/test_deploy21.py \
  python/test_curriculum_pipeline.py -q -k 'not deploy21_e2e'
```

33 passed, 1 deselected。新增覆盖：晋升后跟随 A 的最优 side、stored-rate 迁移、
启动迁移 marker、胜局显著损失 + 平波仍否决、分歧对不足 6 时波数仍可替代。

## 边界

- 不改 r95 历史 `decision.json`；是否回滚/重评由主控决定。
- 不部署；控制器重启时 state 迁移自动生效（`gate_panel_student_round`）。
- 区级 win guard 是额外的安全否决，仍保留原 Holm 家族；两者可能同时给理由。
