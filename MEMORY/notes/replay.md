# 回放 RL 检查点的硬规则 / 任务名陷阱

## 回放 RL 检查点的两条硬规则

1. **卡组只能来自训练任务清单，不能用 `deck_for_level()`。** 训练是 `train_pvz_ppo.py:64` =
   `env.reset(deck=task["deck"], ...)`，清单在 `experiments/t7/*/train.json`。卡槽错位时**这一局会照常跑完
   并给出结果**（静默）。`policy=ppo` 不带 deck 现在直接报错；用 `capture --task-manifest <清单> --task-id <任务>`
   一次拿全 level/deck/cap/mult/**预种植物**。
2. **`provenance.fingerprints.simulator` 是训练时的 `build/pvz-portable`**（`evaluate_full_acceptance.py` 拿它做
   硬门禁）。本机 `2056e13e` / 主线 `dc8747f4` / 老 v4 `371a75d7` 都不同。**跨平台编译同源码字节就不同 →
   "不一致"≠行为变了**；`python/*.py`+`scripts/*.py` 的**源码指纹**才是语义判据。

## 任务名陷阱：`heldout_*` ≠ `train_*`

`heldout_roof_1` = 第 45 关 / cap3 / deck `[0,1,3,4,7,33]`；`train_roof_1` = 第 42 关 / cap1 / deck
`[0,1,2,3,4,33]`。早先 `lookup_task()` 剥掉 `heldout_` 前缀回退到 `train_*`，静默拼错。**已修**：先收
`experiments/t7/*/progress_evaluation.json`（精确键），**只允许 `validation_X → train_X` 一种别名**，
`heldout_X` 一律不回退。**推广规则：任何"按名字回退匹配"的查表都必须显式白名单别名。**
另：**屋顶卡组含花盆（33）**，草地 vs 屋顶不是同一动作空间。
