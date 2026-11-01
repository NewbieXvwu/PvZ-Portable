# task212：A 槽改为 B 的兄弟（旧行 10%）

- 新配方 `recipes/dw4i2_v2timer_clean_old10.json` 复制 `clean_b`，仅 `source_weight.default` 0 → 0.1；`qc_launch.py` 的 `--recipe` 默认指向它。
- `curriculum_queue.py` 用同一 `_b_parent` 给 A/B 写 init/epochs（首轮 `gptbc_s1`/8，之后上一轮 B/4）和同一 replay 窗口；A、B 生成配方只有 `source_weight` 不同。
- 训练前 `_a_recipe`、`_b_recipe` 再按同一 B 标记刷新 init/epochs；谱系仍只由 seed 2（B）推进，A 不入谱系，两臂继续共用同一个训练包。
- 用 `TrainingPipeline.enqueue`（mock executor，不训练）打印：round 1 init 相同、epochs 8；有上一轮 B 后 init 同为该 B、epochs 4；两次 pack key 均相同（夹具数据 `49b8c94da89a13c3`）。
- `pytest -k` 通过：`test_deploy21 clean_b_lineage`、`test_ab_queue recipes_share`、`test_multiseed_production two_recipes`、`test_llm_demo_rows clean_b`。
