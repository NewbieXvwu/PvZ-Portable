# task223b：roof 也改为胜局优先

分支 `gatev8`，基线 `fbd4b534`（task223 已部署）。**未部署**，没有改生产状态。

## 改动

`pvz_promotion.decide()` 不再对 roof 无条件跑波数 Wilcoxon：

- 胜局分歧对 ≥ `ZONE_WIN_GUARD_DISCORDANT=6`：以胜局 McNemar 为准；
  `p < ZONE_WIN_GUARD_ALPHA=.05` 直接否决（`significant_win_regression:roof`）。
- 胜局分歧对 < 6：仍用原来的 roof 波数 Wilcoxon，α=`ROOF_WAVE_ALPHA/(4 looks × 2 arms)`，
  理由 `significant_roof_wave_regression:roof`。这时才是“胜局太少，波数替代”。
- `policy()['roof_veto']` 增加 `win_first` 说明；区级 Holm 仍只含四个非屋顶区。

## 三次离线复算（只读存储 manifest/games，无新对局）

| 轮次 | 旧结论 | 新 A | 新 B | 说明 |
|---|---|---|---|---|
| r94 | B 通过 | 拒绝：night 胜局回归 | 通过 | 与旧赢家一致（B） |
| r95 | A 通过 | 拒绝：fog 胜局回归 | 拒绝：总体负 | 修后无候选人通过 |
| r96 | 无候选人 | **拒绝：roof 胜局回归** | 拒绝：总体/pool/roof | 仍无候选人，但理由从波数变成胜局 |

r96 的关键数字（A vs 现任 r95，不是 A vs B）：

- 总体 236 vs 205，+4.65pp，score_ci [-0.91pp, +10.20pp]（总体本身不否决）。
- roof 胜局 **31 vs 42**，gain 12 / harm 23，win discordant 35，
  McNemar 单侧 p=0.0448 < .05 → `significant_win_regression:roof`。
- 同轮 `ab_result` 的 roof 31 vs 30 是 **A−B**；门禁晋升比的是 A vs 现任 r95，
  后者 roof 是 31 vs 42。任务书里“A 在每个区胜局都更多”的数字来自 A−B 口径，
  不是晋升口径。roof 现在按胜局判，A 输在真实胜局上，不是波数。
  作为交叉核对：如果把 B 当成“现任”，A−B 的 roof 31 vs 30、p=.64，A 会通过；
  但那不是晋升门禁的比较。

r94/r95 的旧/新对照：r94 A 旧已因 night 被拒、B 通过；r95 A 旧通过，新因 fog
16 vs 35（p=1.55e-4，discordant 27）被拒；B 仍总体负。与 task223 的结论一致。

## 测试

Mac、`ulimit -f 2000000`、`timeout 300`、单 pytest：

```
python3 -m pytest python/test_gate_v8_panel.py python/test_deploy21.py \
  python/test_curriculum_pipeline.py -q -k 'not deploy21_e2e'
```

36 passed, 1 deselected。新增三条 roof 测试：胜局足够时忽略显著波数差、
胜局太少时回退波数、胜局显著损失且平波时否决。

## 边界

- 不改 r94/r95/r96 历史 decision；不部署，由控制器决定是否重启。
- 生产 state 未写；离线 bundle 与脚本已删。
