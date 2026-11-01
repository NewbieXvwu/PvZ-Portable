# task216：A 槽每轮从 GPT 行为克隆重新起步

- A/B 继续共用同一个训练包和 `source_weight`；只有 init/epochs 不同。
- `curriculum_queue.py --a-restart`（默认开，`--no-a-restart` 回滚）把 `a_restart` 冻结进每个 training job。
- A 复用 `_b_parent(0)` + `_lineage`：每轮 init=`--b-init`（gptbc_s1）、epochs=`b_initial_epochs`（8），不读 B 谱系。
- B 不变：首轮 gptbc_s1/8，之后上一轮 B/`b_epochs`（4），仍只由 seed 2 推进。
- `qc_launch.py` 默认转发 `--a-restart`；`deploy26.py` 回滚显式传 `--no-a-restart`。
- 验证：两轮 `enqueue` 计划 A 恒为 gptbc_s1/8，B 第二轮为上一轮 B/4；两轮 source_weight 与 pack key 均相同。
